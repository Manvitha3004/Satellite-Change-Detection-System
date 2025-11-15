"""
Memory-efficient patch generator for large satellite imagery.
Processes ResourceSat-2 TIF files using windowed reading to stay under 16GB RAM.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from tqdm import tqdm
import warnings


class PatchGenerator:
    """
    Generate patches from large satellite images using memory-efficient windowed reading.
    
    Features:
    - Windowed reading (never loads full image into memory)
    - Configurable patch size and overlap
    - Filtering of invalid patches (no-data, zeros)
    - Metadata preservation (CRS, transform, coordinates)
    - Progress tracking with tqdm
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        overlap: int = 64,
        min_valid_ratio: float = 0.2,
        max_nodata_ratio: float = 0.8,
        target_bands: Optional[List[int]] = None
    ):
        """
        Initialize patch generator.
        
        Args:
            patch_size: Size of square patches (pixels)
            overlap: Overlap between adjacent patches (pixels)
            min_valid_ratio: Minimum ratio of valid (non-zero) pixels
            max_nodata_ratio: Maximum ratio of no-data pixels
            target_bands: List of band indices to use (None = all bands)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.min_valid_ratio = min_valid_ratio
        self.max_nodata_ratio = max_nodata_ratio
        self.target_bands = target_bands
        
    def generate_sliding_windows(
        self, 
        height: int, 
        width: int
    ) -> List[Window]:
        """
        Generate list of sliding windows covering the entire image.
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            
        Returns:
            List of rasterio Window objects
        """
        windows = []
        
        for row in range(0, height, self.stride):
            for col in range(0, width, self.stride):
                # Calculate window dimensions
                win_height = min(self.patch_size, height - row)
                win_width = min(self.patch_size, width - col)
                
                # Only use full-sized patches (skip edge patches)
                if win_height == self.patch_size and win_width == self.patch_size:
                    window = Window(col, row, win_width, win_height)
                    windows.append(window)
        
        return windows
    
    def is_valid_patch(
        self, 
        patch: np.ndarray,
        nodata_value: Optional[float] = None
    ) -> bool:
        """
        Check if patch contains enough valid data.
        
        Args:
            patch: Patch array (H, W, C) or (C, H, W)
            nodata_value: No-data value to check for
            
        Returns:
            True if patch is valid, False otherwise
        """
        # Handle both (H, W, C) and (C, H, W) formats
        if patch.ndim == 3 and patch.shape[0] <= 10:  # Likely (C, H, W)
            patch = np.moveaxis(patch, 0, -1)  # Convert to (H, W, C)
        
        total_pixels = patch.shape[0] * patch.shape[1]
        
        # Check for all-zero patches
        non_zero_pixels = np.count_nonzero(patch)
        non_zero_ratio = non_zero_pixels / (total_pixels * patch.shape[-1])
        
        if non_zero_ratio < self.min_valid_ratio:
            return False
        
        # Check for no-data values if specified
        if nodata_value is not None:
            nodata_pixels = np.sum(patch == nodata_value)
            nodata_ratio = nodata_pixels / (total_pixels * patch.shape[-1])
            
            if nodata_ratio > self.max_nodata_ratio:
                return False
        
        return True
    
    def read_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """
        Read image metadata without loading data.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with metadata (CRS, transform, shape, etc.)
        """
        with rasterio.open(image_path) as src:
            metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'shape': (src.height, src.width),
                'count': src.count,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'bounds': src.bounds,
                'res': src.res
            }
        return metadata
    
    def generate_patches_from_image(
        self,
        image_path: Path,
        output_dir: Path,
        prefix: str = 'patch',
        verbose: bool = True
    ) -> int:
        """
        Generate patches from a single image file.
        
        Args:
            image_path: Path to input image (TIF)
            output_dir: Directory to save patches
            prefix: Prefix for patch filenames
            verbose: Show progress bar
            
        Returns:
            Number of valid patches generated
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read metadata
        metadata = self.read_image_metadata(image_path)
        height, width = metadata['shape']
        nodata = metadata['nodata']
        
        if verbose:
            print(f"Processing {image_path.name}")
            print(f"Image size: {height} x {width} pixels")
            print(f"Bands: {metadata['count']}")
        
        # Generate sliding windows
        windows = self.generate_sliding_windows(height, width)
        
        if verbose:
            print(f"Total possible patches: {len(windows)}")
        
        valid_count = 0
        
        # Open file once and read patches
        with rasterio.open(image_path) as src:
            # Select bands
            band_indices = self.target_bands if self.target_bands else list(range(1, src.count + 1))
            
            # Process patches with progress bar
            iterator = tqdm(windows, desc="Generating patches") if verbose else windows
            
            for idx, window in enumerate(iterator):
                try:
                    # Read patch from window (only loads small region)
                    patch = src.read(band_indices, window=window)
                    
                    # Validate patch
                    if not self.is_valid_patch(patch, nodata):
                        continue
                    
                    # Get window transform for georeferencing
                    window_transform = src.window_transform(window)
                    
                    # Get window bounds in geographic coordinates
                    window_bounds = rasterio.windows.bounds(window, src.transform)
                    
                    # Save patch with metadata
                    patch_filename = output_dir / f"{prefix}_{valid_count:06d}.npz"
                    
                    np.savez_compressed(
                        patch_filename,
                        data=patch.astype(np.float32),
                        transform=np.array(window_transform.to_gdal()),
                        crs=str(metadata['crs']),
                        bounds=window_bounds,
                        window_col=window.col_off,
                        window_row=window.row_off,
                        source_image=str(image_path.name)
                    )
                    
                    valid_count += 1
                    
                except Exception as e:
                    warnings.warn(f"Error processing window {idx}: {str(e)}")
                    continue
        
        if verbose:
            print(f"Generated {valid_count} valid patches")
        
        return valid_count
    
    def generate_patches_from_pair(
        self,
        image1_path: Path,
        image2_path: Path,
        output_dir: Path,
        mask_path: Optional[Path] = None,
        verbose: bool = True
    ) -> int:
        """
        Generate aligned patches from an image pair (T1, T2) and optional mask.
        
        Args:
            image1_path: Path to first temporal image
            image2_path: Path to second temporal image
            output_dir: Directory to save patches
            mask_path: Optional path to change mask
            verbose: Show progress bar
            
        Returns:
            Number of valid patch pairs generated
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Read metadata from both images
        meta1 = self.read_image_metadata(image1_path)
        meta2 = self.read_image_metadata(image2_path)
        
        # Handle shape mismatch by cropping to minimum dimensions
        if meta1['shape'] != meta2['shape']:
            if verbose:
                print(f"  ⚠ Shape mismatch: {meta1['shape']} vs {meta2['shape']}")
                print(f"  → Cropping to overlapping region...")
            # Use minimum dimensions
            min_height = min(meta1['shape'][0], meta2['shape'][1])
            min_width = min(meta1['shape'][1], meta2['shape'][1])
            height, width = min_height, min_width
        else:
            height, width = meta1['shape']
        
            height, width = meta1['shape']
        
        if verbose:
            print(f"Processing image pair:")
            print(f"  T1: {image1_path.name}")
            print(f"  T2: {image2_path.name}")
            if mask_path:
                print(f"  Mask: {mask_path.name}")
            print(f"  Using dimensions: {height} x {width} pixels")        # Generate sliding windows
        windows = self.generate_sliding_windows(height, width)
        
        if verbose:
            print(f"Total possible patches: {len(windows)}")
        
        valid_count = 0
        
        # Open all files
        with rasterio.open(image1_path) as src1, \
             rasterio.open(image2_path) as src2:
            
            # Open mask if provided
            mask_src = rasterio.open(mask_path) if mask_path else None
            
            try:
                # Select bands
                band_indices = self.target_bands if self.target_bands else list(range(1, src1.count + 1))
                
                # Process patches
                iterator = tqdm(windows, desc="Generating patch pairs") if verbose else windows
                
                for window in iterator:
                    try:
                        # Read patches from both images
                        patch1 = src1.read(band_indices, window=window)
                        patch2 = src2.read(band_indices, window=window)
                        
                        # Validate both patches
                        if not self.is_valid_patch(patch1, meta1['nodata']):
                            continue
                        if not self.is_valid_patch(patch2, meta2['nodata']):
                            continue
                        
                        # Read mask patch if available
                        mask_patch = None
                        if mask_src:
                            mask_patch = mask_src.read(1, window=window)
                            # Convert to binary (0 or 1)
                            mask_patch = (mask_patch > 0).astype(np.uint8)
                        
                        # Get window metadata
                        window_transform = src1.window_transform(window)
                        window_bounds = rasterio.windows.bounds(window, src1.transform)
                        
                        # Save patch pair
                        patch_filename = output_dir / f"pair_{valid_count:06d}.npz"
                        
                        save_dict = {
                            'image1': patch1.astype(np.float32),
                            'image2': patch2.astype(np.float32),
                            'transform': np.array(window_transform.to_gdal()),
                            'crs': str(meta1['crs']),
                            'bounds': window_bounds,
                            'window_col': window.col_off,
                            'window_row': window.row_off,
                            'source_image1': str(image1_path.name),
                            'source_image2': str(image2_path.name)
                        }
                        
                        if mask_patch is not None:
                            save_dict['mask'] = mask_patch
                        
                        np.savez_compressed(patch_filename, **save_dict)
                        
                        valid_count += 1
                        
                    except Exception as e:
                        warnings.warn(f"Error processing window: {str(e)}")
                        continue
            
            finally:
                if mask_src:
                    mask_src.close()
        
        if verbose:
            print(f"Generated {valid_count} valid patch pairs")
        
        return valid_count
    
    def generate_patches_from_stacked_bands(
        self,
        band_files1: List[Path],
        band_files2: List[Path],
        output_dir: Path,
        mask_path: Optional[Path] = None,
        verbose: bool = True
    ) -> int:
        """
        Generate patches from separate band files (stacks them first).
        Handles ResourceSat-2 format where each band is a separate TIF file.
        
        Args:
            band_files1: List of band file paths for T1 (e.g., BAND2, BAND3, BAND4)
            band_files2: List of band file paths for T2
            output_dir: Directory to save patches
            mask_path: Optional path to change mask
            verbose: Show progress bar
            
        Returns:
            Number of valid patch pairs generated
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(band_files1) != len(band_files2):
            raise ValueError(f"Band count mismatch: {len(band_files1)} vs {len(band_files2)}")
        
        # Read metadata from first band of each image
        with rasterio.open(band_files1[0]) as src1, rasterio.open(band_files2[0]) as src2:
            shape1 = src1.shape
            shape2 = src2.shape
            crs1 = src1.crs
            nodata1 = src1.nodata
            
            # Handle shape mismatch by cropping to minimum dimensions
            if shape1 != shape2:
                if verbose:
                    print(f"  Shape mismatch: {shape1} vs {shape2}")
                    print(f"  Cropping to minimum dimensions...")
                min_height = min(shape1[0], shape2[0])
                min_width = min(shape1[1], shape2[1])
                height, width = min_height, min_width
            else:
                height, width = shape1
            
            if verbose:
                print(f"  Using dimensions: {height} x {width} pixels")
                print(f"  Bands: {len(band_files1)}")
        
        # Generate sliding windows
        windows = self.generate_sliding_windows(height, width)
        
        if verbose:
            print(f"  Total possible patches: {len(windows)}")
        
        valid_count = 0
        
        # Open all band files
        srcs1 = [rasterio.open(bf) for bf in band_files1]
        srcs2 = [rasterio.open(bf) for bf in band_files2]
        mask_src = rasterio.open(mask_path) if mask_path else None
        
        try:
            # Process patches
            iterator = tqdm(windows, desc="Generating patches") if verbose else windows
            
            for window in iterator:
                try:
                    # Read all bands for both images
                    patches1 = []
                    patches2 = []
                    
                    for src in srcs1:
                        band_data = src.read(1, window=window)  # Read band 1 (single band per file)
                        patches1.append(band_data)
                    
                    for src in srcs2:
                        band_data = src.read(1, window=window)
                        patches2.append(band_data)
                    
                    # Stack bands
                    patch1 = np.stack(patches1, axis=0)  # Shape: (bands, height, width)
                    patch2 = np.stack(patches2, axis=0)
                    
                    # Validate both patches
                    if not self.is_valid_patch(patch1, nodata1):
                        continue
                    if not self.is_valid_patch(patch2, nodata1):
                        continue
                    
                    # Read mask patch if available
                    mask_patch = None
                    if mask_src:
                        mask_patch = mask_src.read(1, window=window)
                        mask_patch = (mask_patch > 0).astype(np.uint8)
                    else:
                        # Create dummy mask (all zeros for unsupervised)
                        mask_patch = np.zeros(patch1.shape[1:], dtype=np.uint8)
                    
                    # Get window metadata
                    window_transform = srcs1[0].window_transform(window)
                    window_bounds = rasterio.windows.bounds(window, srcs1[0].transform)
                    
                    # Save patch pair
                    patch_filename = output_dir / f"patch_{valid_count:06d}.npz"
                    
                    np.savez_compressed(
                        patch_filename,
                        image1=patch1.astype(np.float32),
                        image2=patch2.astype(np.float32),
                        mask=mask_patch,
                        transform=np.array(window_transform.to_gdal()),
                        crs=str(crs1),
                        bounds=window_bounds,
                        window_col=window.col_off,
                        window_row=window.row_off
                    )
                    
                    valid_count += 1
                    
                except Exception as e:
                    if verbose:
                        warnings.warn(f"Error processing window: {str(e)}")
                    continue
        
        finally:
            # Close all files
            for src in srcs1:
                src.close()
            for src in srcs2:
                src.close()
            if mask_src:
                mask_src.close()
        
        if verbose:
            print(f"  ✓ Generated {valid_count} valid patches")
        
        return valid_count
    
    def split_patches(
        self,
        patches_dir: Path,
        output_base_dir: Path,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[int, int]:
        """
        Split patches into train and validation sets.
        
        Args:
            patches_dir: Directory containing patches
            output_base_dir: Base directory for train/val splits
            train_ratio: Ratio of training patches
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_count, val_count)
        """
        np.random.seed(seed)
        
        # Get all patch files
        patch_files = sorted(list(Path(patches_dir).glob('*.npz')))
        
        if len(patch_files) == 0:
            raise ValueError(f"No patches found in {patches_dir}")
        
        # Shuffle and split
        indices = np.random.permutation(len(patch_files))
        split_idx = int(len(patch_files) * train_ratio)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create output directories
        train_dir = Path(output_base_dir) / 'train'
        val_dir = Path(output_base_dir) / 'val'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to train directory
        print(f"Copying {len(train_indices)} patches to train set...")
        for idx in tqdm(train_indices):
            src_file = patch_files[idx]
            dst_file = train_dir / src_file.name
            # Load and resave (could also use shutil.copy)
            data = np.load(src_file)
            np.savez_compressed(dst_file, **{k: data[k] for k in data.files})
        
        # Copy files to val directory
        print(f"Copying {len(val_indices)} patches to val set...")
        for idx in tqdm(val_indices):
            src_file = patch_files[idx]
            dst_file = val_dir / src_file.name
            data = np.load(src_file)
            np.savez_compressed(dst_file, **{k: data[k] for k in data.files})
        
        return len(train_indices), len(val_indices)


def find_resourcesat2_bands(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Find ResourceSat-2 band files in the dataset directory.
    
    Args:
        data_dir: Root dataset directory
        
    Returns:
        Dictionary mapping date identifiers to lists of band paths
    """
    data_dir = Path(data_dir)
    image_pairs = {}
    
    # Find all ResourceSat-2 directories
    for scene_dir in data_dir.glob('R2F*'):
        # Navigate to the nested directory with the same name
        inner_dir = scene_dir / scene_dir.name
        
        if not inner_dir.exists():
            continue
        
        # Find BAND files
        band_files = sorted(inner_dir.glob('BAND*.tif'))
        
        if len(band_files) >= 3:  # Need at least 3 bands (Green, Red, NIR)
            # Extract date from directory name (e.g., 18MAR2020)
            parts = scene_dir.name.split('F')
            if len(parts) >= 2:
                date_str = parts[1][:11]  # e.g., "18MAR2020"
                image_pairs[date_str] = band_files
    
    return image_pairs
