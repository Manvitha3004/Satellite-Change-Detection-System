"""
Memory-efficient inference for large satellite images.
Processes images tile-by-tile to run on CPU with 16GB RAM.
"""

import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.siamese_unet import create_model
from data.preprocessing import normalize_bands, read_and_stack_bands


class TiledPredictor:
    """
    Tile-based predictor for large images.
    Processes images in small tiles to avoid memory overflow.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        tile_size: int = 256,
        overlap: int = 64,
        batch_size: int = 1,
        threshold: float = 0.5
    ):
        """
        Initialize tiled predictor.
        
        Args:
            model: Trained change detection model
            device: Device for inference
            tile_size: Size of tiles (pixels)
            overlap: Overlap between tiles (pixels)
            batch_size: Number of tiles to process at once
            threshold: Threshold for binary prediction
        """
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.batch_size = batch_size
        self.threshold = threshold
        
        self.model.eval()
        self.model.to(device)
    
    def generate_tiles(
        self,
        height: int,
        width: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate tile coordinates covering the entire image.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            List of (row, col, tile_height, tile_width) tuples
        """
        tiles = []
        
        for row in range(0, height, self.stride):
            for col in range(0, width, self.stride):
                tile_h = min(self.tile_size, height - row)
                tile_w = min(self.tile_size, width - col)
                
                # Pad to tile_size if at edge
                if tile_h < self.tile_size or tile_w < self.tile_size:
                    # Calculate padding needed
                    pad_h = self.tile_size - tile_h
                    pad_w = self.tile_size - tile_w
                    tiles.append((row, col, tile_h, tile_w, pad_h, pad_w))
                else:
                    tiles.append((row, col, tile_h, tile_w, 0, 0))
        
        return tiles
    
    @torch.no_grad()
    def predict_tile(
        self,
        tile1: np.ndarray,
        tile2: np.ndarray
    ) -> np.ndarray:
        """
        Predict change mask for a single tile.
        
        Args:
            tile1: First temporal tile (C, H, W)
            tile2: Second temporal tile (C, H, W)
            
        Returns:
            Predicted change mask (H, W)
        """
        # Normalize
        tile1 = normalize_bands(tile1, method='percentile')
        tile2 = normalize_bands(tile2, method='percentile')
        
        # Convert to tensors
        tile1_tensor = torch.from_numpy(tile1).unsqueeze(0).float().to(self.device)
        tile2_tensor = torch.from_numpy(tile2).unsqueeze(0).float().to(self.device)
        
        # Predict
        output = self.model(tile1_tensor, tile2_tensor)
        
        # Convert to numpy
        pred = output.squeeze().cpu().numpy()
        
        return pred
    
    def predict_large_image(
        self,
        image1_path: Path,
        image2_path: Path,
        output_path: Path,
        band_indices: Optional[List[int]] = None,
        progress: bool = True
    ) -> np.ndarray:
        """
        Predict change mask for large image using tiled inference.
        
        Args:
            image1_path: Path to first temporal image
            image2_path: Path to second temporal image
            output_path: Path to save output mask
            band_indices: Indices of bands to use (None = all)
            progress: Show progress bar
            
        Returns:
            Full prediction mask
        """
        # Open images
        with rasterio.open(image1_path) as src1:
            height, width = src1.height, src1.width
            profile = src1.profile.copy()
            transform = src1.transform
            crs = src1.crs
        
        # Update profile for output
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw',
            'nodata': None
        })
        
        # Create output arrays
        prediction = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        
        # Generate tiles
        tiles = self.generate_tiles(height, width)
        
        if progress:
            print(f"Processing {len(tiles)} tiles...")
            tiles_iter = tqdm(tiles, desc="Predicting")
        else:
            tiles_iter = tiles
        
        # Process tiles
        with rasterio.open(image1_path) as src1, \
             rasterio.open(image2_path) as src2:
            
            for tile_info in tiles_iter:
                if len(tile_info) == 6:
                    row, col, tile_h, tile_w, pad_h, pad_w = tile_info
                else:
                    row, col, tile_h, tile_w = tile_info
                    pad_h, pad_w = 0, 0
                
                # Create window
                window = Window(col, row, tile_w, tile_h)
                
                # Read tiles
                if band_indices:
                    tile1 = src1.read(band_indices, window=window)
                    tile2 = src2.read(band_indices, window=window)
                else:
                    tile1 = src1.read(window=window)
                    tile2 = src2.read(window=window)
                
                # Pad if necessary
                if pad_h > 0 or pad_w > 0:
                    tile1 = np.pad(
                        tile1,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode='reflect'
                    )
                    tile2 = np.pad(
                        tile2,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode='reflect'
                    )
                
                # Predict
                pred_tile = self.predict_tile(tile1, tile2)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    pred_tile = pred_tile[:tile_h, :tile_w]
                
                # Accumulate predictions (overlap averaging)
                prediction[row:row+tile_h, col:col+tile_w] += pred_tile
                counts[row:row+tile_h, col:col+tile_w] += 1.0
        
        # Average overlapping predictions
        prediction = np.divide(
            prediction,
            counts,
            out=np.zeros_like(prediction),
            where=counts != 0
        )
        
        # Binarize
        prediction_binary = (prediction >= self.threshold).astype(np.uint8)
        
        # Save output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction_binary, 1)
        
        if progress:
            print(f"✓ Saved prediction to {output_path}")
        
        return prediction_binary


def load_model_for_inference(
    checkpoint_path: Path,
    device: torch.device = torch.device('cpu'),
    config: Optional[dict] = None
) -> nn.Module:
    """
    Load trained model for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for inference
        config: Optional config dict (will load from checkpoint if not provided)
        
    Returns:
        Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if config is None:
        config = checkpoint.get('config', {})
    
    # Auto-detect encoder from checkpoint state_dict keys
    # (CPU training script didn't save encoder_name in config)
    sample_keys = list(checkpoint['model_state_dict'].keys())
    
    # ResNet uses: conv1, bn1, layer1, layer2, etc.
    # EfficientNet uses: _conv_stem, _bn0, _blocks, etc.
    if any('model.encoder.layer1' in k for k in sample_keys):
        encoder_name = 'resnet18'
        print("  Auto-detected encoder: resnet18")
    elif any('model.encoder._blocks' in k for k in sample_keys):
        encoder_name = 'efficientnet-b0'
        print("  Auto-detected encoder: efficientnet-b0")
    else:
        # Fallback: check config if it has model.encoder_name
        model_cfg = config.get('model', {})
        if isinstance(model_cfg, dict):
            encoder_name = model_cfg.get('encoder_name', 'resnet18')
        else:
            encoder_name = 'resnet18'
        print(f"  Using encoder: {encoder_name}")
    
    # Create model
    model = create_model(
        model_type='simple',
        encoder_name=encoder_name,
        encoder_weights=None,  # Don't load pretrained weights
        in_channels=config.get('model', {}).get('in_channels', 3) if isinstance(config.get('model'), dict) else 3,
        classes=config.get('model', {}).get('classes', 1) if isinstance(config.get('model'), dict) else 1,
        activation=config.get('model', {}).get('activation', 'sigmoid') if isinstance(config.get('model'), dict) else 'sigmoid'
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"[OK] Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    best_metric = checkpoint.get('best_val_metric', checkpoint.get('best_val_iou', 'unknown'))
    if isinstance(best_metric, (int, float)):
        print(f"  Best val metric: {best_metric:.4f}")
    else:
        print(f"  Best val metric: {best_metric}")
    
    return model


def predict_from_band_files(
    model: nn.Module,
    band_paths1: List[Path],
    band_paths2: List[Path],
    output_path: Path,
    device: torch.device = torch.device('cpu'),
    tile_size: int = 256,
    overlap: int = 64,
    threshold: float = 0.5,
    band_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Predict change mask from separate band files (ResourceSat-2 format).
    
    Args:
        model: Trained model
        band_paths1: List of band file paths for T1
        band_paths2: List of band file paths for T2
        output_path: Path to save output
        device: Device for inference
        tile_size: Size of tiles
        overlap: Overlap between tiles
        threshold: Binary threshold
        band_indices: Indices of bands to use
        
    Returns:
        Prediction mask
    """
    # Read and stack bands
    print("Reading and stacking bands...")
    image1, meta1 = read_and_stack_bands(band_paths1, target_bands=band_indices)
    image2, meta2 = read_and_stack_bands(band_paths2, target_bands=band_indices)
    
    height1, width1 = image1.shape[1:]
    height2, width2 = image2.shape[1:]
    
    if (height1, width1) != (height2, width2):
        min_height = min(height1, height2)
        min_width = min(width1, width2)
        print(f"Shape mismatch detected: T1={height1}x{width1}, T2={height2}x{width2}")
        print(f"Cropping both stacks to common {min_height} x {min_width} area")
        image1 = image1[:, :min_height, :min_width]
        image2 = image2[:, :min_height, :min_width]
        height, width = min_height, min_width
        if meta1 is not None:
            meta1 = meta1.copy()
            meta1['shape'] = (height, width)
    else:
        height, width = height1, width1
    
    print(f"Image size: {height} x {width}")
    print(f"Bands: {image1.shape[0]}")
    
    # Create predictor
    predictor = TiledPredictor(
        model=model,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        threshold=threshold
    )
    
    # Create output arrays
    prediction = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    # Generate tiles
    tiles = predictor.generate_tiles(height, width)
    
    print(f"Processing {len(tiles)} tiles...")
    
    # Process tiles
    for tile_info in tqdm(tiles, desc="Predicting"):
        row, col, tile_h, tile_w, pad_h, pad_w = tile_info
        
        # Extract tiles
        tile1 = image1[:, row:row+tile_h, col:col+tile_w].copy()
        tile2 = image2[:, row:row+tile_h, col:col+tile_w].copy()
        
        # Pad to tile_size if necessary
        if pad_h > 0 or pad_w > 0:
            tile1 = np.pad(
                tile1,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )
            tile2 = np.pad(
                tile2,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )
        
        # Verify tile sizes match
        assert tile1.shape == tile2.shape, f"Tile shape mismatch: {tile1.shape} vs {tile2.shape}"
        assert tile1.shape[1] == tile_size and tile1.shape[2] == tile_size, \
            f"Tile not padded to {tile_size}x{tile_size}: got {tile1.shape}"
        
        # Predict
        pred_tile = predictor.predict_tile(tile1, tile2)
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            pred_tile = pred_tile[:tile_h, :tile_w]
        
        # Accumulate
        prediction[row:row+tile_h, col:col+tile_w] += pred_tile
        counts[row:row+tile_h, col:col+tile_w] += 1.0
    
    # Average overlapping predictions
    prediction = np.divide(
        prediction,
        counts,
        out=np.zeros_like(prediction),
        where=counts != 0
    )
    
    # Binarize
    prediction_binary = (prediction >= threshold).astype(np.uint8)
    
    # Save output with georeference
    profile = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': width,
        'height': height,
        'count': 1,
        'crs': meta1['crs'],
        'transform': meta1['transform'],
        'compress': 'lzw',
        'nodata': None
    }
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction_binary, 1)
    
    print(f"[OK] Saved prediction to {output_path}")
    print(f"  Change pixels: {np.sum(prediction_binary):,}")
    
    return prediction_binary


if __name__ == '__main__':
    # Test inference
    print("Inference module loaded successfully")
