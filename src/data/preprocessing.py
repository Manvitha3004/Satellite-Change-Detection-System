"""
Image preprocessing utilities for satellite imagery.
Handles normalization, band selection, and image alignment.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as ResamplingEnum
from pathlib import Path
from typing import Tuple, Optional, List, Union
import warnings


def normalize_bands(
    image: np.ndarray,
    method: str = 'minmax',
    percentile_range: Tuple[float, float] = (2, 98),
    clip: bool = True
) -> np.ndarray:
    """
    Normalize image bands to [0, 1] range.
    
    Args:
        image: Input image array (C, H, W) or (H, W, C)
        method: Normalization method ('minmax', 'percentile', 'standardize')
        percentile_range: Percentile range for 'percentile' method
        clip: Clip values to [0, 1] range
        
    Returns:
        Normalized image array
    """
    image = image.astype(np.float32)
    
    # Determine axis for normalization
    if image.ndim == 3:
        # Assume (C, H, W) if first dimension is small (<= 10)
        if image.shape[0] <= 10:
            axis = (1, 2)  # Normalize each channel independently
        else:
            axis = (0, 1)  # Assume (H, W, C)
            image = np.moveaxis(image, -1, 0)  # Convert to (C, H, W)
    else:
        axis = None  # Normalize entire 2D image
    
    if method == 'minmax':
        # Min-max normalization
        if axis is not None:
            mins = np.min(image, axis=axis, keepdims=True)
            maxs = np.max(image, axis=axis, keepdims=True)
        else:
            mins = np.min(image)
            maxs = np.max(image)
        
        # Avoid division by zero
        range_vals = maxs - mins
        range_vals[range_vals == 0] = 1.0
        
        normalized = (image - mins) / range_vals
    
    elif method == 'percentile':
        # Percentile-based normalization (robust to outliers)
        if axis is not None:
            p_low = np.percentile(image, percentile_range[0], axis=axis, keepdims=True)
            p_high = np.percentile(image, percentile_range[1], axis=axis, keepdims=True)
        else:
            p_low = np.percentile(image, percentile_range[0])
            p_high = np.percentile(image, percentile_range[1])
        
        range_vals = p_high - p_low
        range_vals[range_vals == 0] = 1.0
        
        normalized = (image - p_low) / range_vals
    
    elif method == 'standardize':
        # Z-score normalization
        if axis is not None:
            mean = np.mean(image, axis=axis, keepdims=True)
            std = np.std(image, axis=axis, keepdims=True)
        else:
            mean = np.mean(image)
            std = np.std(image)
        
        std[std == 0] = 1.0
        normalized = (image - mean) / std
        
        # Rescale to [0, 1] for standardize
        if clip:
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Clip to valid range
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized


def preprocess_image(
    image: np.ndarray,
    normalize: bool = True,
    normalization_method: str = 'percentile',
    target_dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Apply preprocessing pipeline to satellite image.
    
    Args:
        image: Input image array
        normalize: Apply normalization
        normalization_method: Normalization method to use
        target_dtype: Target data type
        
    Returns:
        Preprocessed image
    """
    # Convert to float
    image = image.astype(target_dtype)
    
    # Handle no-data values (replace with 0)
    image[~np.isfinite(image)] = 0
    
    # Normalize if requested
    if normalize:
        image = normalize_bands(image, method=normalization_method)
    
    return image


def read_and_stack_bands(
    band_paths: List[Path],
    target_bands: Optional[List[int]] = None,
    window: Optional[rasterio.windows.Window] = None
) -> Tuple[np.ndarray, dict]:
    """
    Read and stack multiple band files into a single array.
    
    Args:
        band_paths: List of paths to band files (BAND2.tif, BAND3.tif, BAND4.tif)
        target_bands: Indices of bands to read (None = all)
        window: Optional window to read subset
        
    Returns:
        Tuple of (stacked_array, metadata)
    """
    if target_bands is not None:
        band_paths = [band_paths[i] for i in target_bands]
    
    bands = []
    metadata = None
    
    for band_path in band_paths:
        with rasterio.open(band_path) as src:
            # Read band data
            if window is not None:
                data = src.read(1, window=window)
            else:
                data = src.read(1)
            
            bands.append(data)
            
            # Store metadata from first band
            if metadata is None:
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform if window is None else src.window_transform(window),
                    'shape': data.shape,
                    'dtype': src.dtypes[0],
                    'nodata': src.nodata,
                    'bounds': src.bounds
                }
    
    # Stack bands (C, H, W)
    stacked = np.stack(bands, axis=0)
    
    return stacked, metadata


def align_images(
    image1_path: Path,
    image2_path: Path,
    output1_path: Optional[Path] = None,
    output2_path: Optional[Path] = None,
    target_crs: Optional[str] = None,
    target_resolution: Optional[float] = None,
    resampling_method: str = 'cubic'
) -> Tuple[bool, str]:
    """
    Check and optionally align two images to the same CRS and resolution.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        output1_path: Path to save aligned first image (if None, only check)
        output2_path: Path to save aligned second image (if None, only check)
        target_crs: Target CRS (if None, use image1 CRS)
        target_resolution: Target resolution in meters (if None, use image1 resolution)
        resampling_method: Resampling method ('nearest', 'bilinear', 'cubic', 'lanczos')
        
    Returns:
        Tuple of (aligned, message)
    """
    # Map resampling methods
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos
    }
    
    resampling = resampling_map.get(resampling_method, Resampling.cubic_spline)
    
    with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
        # Check CRS
        crs_match = src1.crs == src2.crs
        
        # Check resolution
        res_match = np.allclose(src1.res, src2.res, atol=0.1)
        
        # Check shape
        shape_match = src1.shape == src2.shape
        
        if crs_match and res_match and shape_match:
            return True, "Images are already aligned"
        
        # Determine target CRS and resolution
        dst_crs = target_crs if target_crs else src1.crs
        dst_res = target_resolution if target_resolution else src1.res[0]
        
        # If only checking, return result
        if output1_path is None or output2_path is None:
            msg = []
            if not crs_match:
                msg.append(f"CRS mismatch: {src1.crs} vs {src2.crs}")
            if not res_match:
                msg.append(f"Resolution mismatch: {src1.res} vs {src2.res}")
            if not shape_match:
                msg.append(f"Shape mismatch: {src1.shape} vs {src2.shape}")
            
            return False, "; ".join(msg)
        
        # Reproject and align images
        # This is a placeholder for full reprojection logic
        # In practice, you'd use rasterio.warp.reproject
        
        warnings.warn(
            "Image reprojection not implemented. "
            "Please ensure images are pre-aligned."
        )
        
        return False, "Reprojection required but not implemented"


def calculate_ndvi(
    red_band: np.ndarray,
    nir_band: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    
    Args:
        red_band: Red band array
        nir_band: NIR band array
        epsilon: Small value to avoid division by zero
        
    Returns:
        NDVI array (values in [-1, 1])
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    numerator = nir - red
    denominator = nir + red + epsilon
    
    ndvi = numerator / denominator
    
    # Clip to valid range
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    return ndvi


def calculate_ndwi(
    green_band: np.ndarray,
    nir_band: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index (NDWI).
    
    Args:
        green_band: Green band array
        nir_band: NIR band array
        epsilon: Small value to avoid division by zero
        
    Returns:
        NDWI array (values in [-1, 1])
    """
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    numerator = green - nir
    denominator = green + nir + epsilon
    
    ndwi = numerator / denominator
    
    # Clip to valid range
    ndwi = np.clip(ndwi, -1.0, 1.0)
    
    return ndwi


def augment_with_indices(
    image: np.ndarray,
    add_ndvi: bool = True,
    add_ndwi: bool = True
) -> np.ndarray:
    """
    Augment image with spectral indices (NDVI, NDWI).
    Assumes image has 3 bands: [Green, Red, NIR]
    
    Args:
        image: Input image (3, H, W) with bands [Green, Red, NIR]
        add_ndvi: Add NDVI as additional band
        add_ndwi: Add NDWI as additional band
        
    Returns:
        Augmented image with additional bands
    """
    if image.shape[0] != 3:
        warnings.warn(f"Expected 3 bands, got {image.shape[0]}. Returning original image.")
        return image
    
    green = image[0]
    red = image[1]
    nir = image[2]
    
    bands = [image]
    
    if add_ndvi:
        ndvi = calculate_ndvi(red, nir)
        bands.append(ndvi[np.newaxis, :, :])
    
    if add_ndwi:
        ndwi = calculate_ndwi(green, nir)
        bands.append(ndwi[np.newaxis, :, :])
    
    augmented = np.concatenate(bands, axis=0)
    
    return augmented


def verify_image_pair(
    image1_path: Path,
    image2_path: Path,
    tolerance: float = 0.1
) -> Tuple[bool, str]:
    """
    Verify that two images are suitable for change detection.
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        tolerance: Tolerance for resolution matching (meters)
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
            issues = []
            
            # Check CRS
            if src1.crs != src2.crs:
                issues.append(f"CRS mismatch: {src1.crs} vs {src2.crs}")
            
            # Check resolution
            if not np.allclose(src1.res, src2.res, atol=tolerance):
                issues.append(f"Resolution mismatch: {src1.res} vs {src2.res}")
            
            # Check shape
            if src1.shape != src2.shape:
                issues.append(f"Shape mismatch: {src1.shape} vs {src2.shape}")
            
            # Check band count
            if src1.count != src2.count:
                issues.append(f"Band count mismatch: {src1.count} vs {src2.count}")
            
            if issues:
                return False, "; ".join(issues)
            
            return True, "Images are compatible for change detection"
    
    except Exception as e:
        return False, f"Error reading images: {str(e)}"
