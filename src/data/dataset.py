"""
PyTorch Dataset for change detection with augmentation support.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):
    """
    PyTorch Dataset for satellite change detection.
    
    Loads pre-generated patches from .npz files and applies augmentation.
    """
    
    def __init__(
        self,
        patches_dir: Path,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        augment: bool = True,
        return_metadata: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            patches_dir: Directory containing .npz patch files
            transform: Optional transform/augmentation pipeline
            normalize: Normalize images to [0, 1]
            augment: Apply data augmentation (only for training)
            return_metadata: Return metadata (CRS, transform) with samples
        """
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.normalize = normalize
        self.augment = augment
        self.return_metadata = return_metadata
        
        # Find all patch files
        self.patch_files = sorted(list(self.patches_dir.glob('*.npz')))
        
        if len(self.patch_files) == 0:
            raise ValueError(f"No patch files found in {patches_dir}")
        
        # Set up default transform if none provided
        if self.transform is None and self.augment:
            self.transform = self.get_default_augmentation()
    
    def __len__(self) -> int:
        return len(self.patch_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (image1, image2, mask) or (image1, image2, mask, metadata)
        """
        # Load patch
        patch_file = self.patch_files[idx]
        data = np.load(patch_file)
        
        # Extract images
        if 'image1' in data and 'image2' in data:
            # Paired patch
            image1 = data['image1']  # (C, H, W)
            image2 = data['image2']  # (C, H, W)
        else:
            # Single patch (use for both)
            image = data['data']
            image1 = image
            image2 = image
        
        # Extract mask if available
        if 'mask' in data:
            mask = data['mask']  # (H, W)
        else:
            # Create dummy mask (all zeros)
            mask = np.zeros(image1.shape[1:], dtype=np.uint8)
        
        # Convert to (H, W, C) for albumentations
        image1 = np.moveaxis(image1, 0, -1)  # (C, H, W) -> (H, W, C)
        image2 = np.moveaxis(image2, 0, -1)
        
        # Normalize to [0, 1] if needed
        if self.normalize:
            image1 = self._normalize(image1)
            image2 = self._normalize(image2)
        
        # Apply augmentation
        if self.transform is not None:
            # Apply same transform to both images and mask
            transformed = self.transform(
                image=image1,
                image2=image2,
                mask=mask
            )
            image1 = transformed['image']
            image2 = transformed['image2']
            mask = transformed['mask']
        else:
            # Convert to tensors
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        if self.return_metadata:
            metadata = {
                'transform': data.get('transform', None),
                'crs': str(data.get('crs', '')),
                'bounds': data.get('bounds', None),
                'source_image1': str(data.get('source_image1', '')),
                'source_image2': str(data.get('source_image2', ''))
            }
            return image1, image2, mask, metadata
        
        return image1, image2, mask
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        # Handle different data types
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Assume already in reasonable range, apply percentile normalization
            p2, p98 = np.percentile(image, [2, 98])
            if p98 > p2:
                image = np.clip((image - p2) / (p98 - p2), 0, 1)
            return image.astype(np.float32)
    
    @staticmethod
    def get_default_augmentation(
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        brightness_limit: float = 0.1,
        contrast_limit: float = 0.1
    ) -> A.Compose:
        """
        Get default augmentation pipeline.
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            rotation_prob: Probability of rotation
            brightness_limit: Brightness adjustment limit
            contrast_limit: Contrast adjustment limit
            
        Returns:
            Albumentations Compose object
        """
        transform = A.Compose([
            # Geometric augmentations (applied to all images and mask)
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.VerticalFlip(p=vertical_flip_prob),
            A.RandomRotate90(p=rotation_prob),
            
            # Photometric augmentations (only applied to images, not mask)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=1.0
                ),
            ], p=0.3),
            
            # Noise augmentation
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            
            # Convert to tensor
            ToTensorV2()
        ], additional_targets={'image2': 'image'})
        
        return transform
    
    @staticmethod
    def get_validation_transform() -> A.Compose:
        """
        Get validation transform (no augmentation, only tensor conversion).
        
        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            ToTensorV2()
        ], additional_targets={'image2': 'image'})


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Directory with training patches
        val_dir: Directory with validation patches
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        augment_train: Apply augmentation to training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = ChangeDetectionDataset(
        patches_dir=train_dir,
        augment=augment_train,
        normalize=True
    )
    
    val_dataset = ChangeDetectionDataset(
        patches_dir=val_dir,
        transform=ChangeDetectionDataset.get_validation_transform(),
        normalize=True,
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


class InferenceDataset(Dataset):
    """
    Dataset for inference on large images using sliding window.
    Generates patches on-the-fly without saving to disk.
    """
    
    def __init__(
        self,
        image1_array: np.ndarray,
        image2_array: np.ndarray,
        windows: List[Tuple[int, int, int, int]],
        normalize: bool = True
    ):
        """
        Initialize inference dataset.
        
        Args:
            image1_array: First image array (C, H, W)
            image2_array: Second image array (C, H, W)
            windows: List of windows (row, col, height, width)
            normalize: Normalize images
        """
        self.image1 = image1_array
        self.image2 = image2_array
        self.windows = windows
        self.normalize = normalize
        
        self.transform = A.Compose([
            ToTensorV2()
        ], additional_targets={'image2': 'image'})
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int, int]]:
        """
        Get a patch for inference.
        
        Returns:
            Tuple of (image1_patch, image2_patch, window_coords)
        """
        row, col, height, width = self.windows[idx]
        
        # Extract patches
        patch1 = self.image1[:, row:row+height, col:col+width]
        patch2 = self.image2[:, row:row+height, col:col+width]
        
        # Convert to (H, W, C)
        patch1 = np.moveaxis(patch1, 0, -1)
        patch2 = np.moveaxis(patch2, 0, -1)
        
        # Normalize
        if self.normalize:
            patch1 = self._normalize(patch1)
            patch2 = self._normalize(patch2)
        
        # Apply transform (tensor conversion)
        transformed = self.transform(image=patch1, image2=patch2)
        patch1 = transformed['image']
        patch2 = transformed['image2']
        
        return patch1, patch2, (row, col, height, width)
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        p2, p98 = np.percentile(image, [2, 98])
        if p98 > p2:
            image = np.clip((image - p2) / (p98 - p2), 0, 1)
        return image.astype(np.float32)
