"""
Post-processing utilities for refining binary change masks.
Applies morphological operations and connected component filtering.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage import morphology, measure
from typing import Tuple, Optional
import warnings


def create_morphology_kernel(
    kernel_size: int = 3,
    kernel_shape: str = 'ellipse'
) -> np.ndarray:
    """
    Create morphological kernel.
    
    Args:
        kernel_size: Size of kernel
        kernel_shape: Shape of kernel ('rect', 'ellipse', 'cross', 'diamond')
        
    Returns:
        Binary kernel array
    """
    if kernel_shape == 'rect':
        return np.ones((kernel_size, kernel_size), dtype=np.uint8)
    elif kernel_shape == 'ellipse':
        return morphology.disk(kernel_size // 2)
    elif kernel_shape == 'cross':
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        center = kernel_size // 2
        kernel[center, :] = 1
        kernel[:, center] = 1
        return kernel
    elif kernel_shape == 'diamond':
        return morphology.diamond(kernel_size // 2)
    else:
        raise ValueError(f"Unknown kernel shape: {kernel_shape}")


def apply_morphological_operations(
    mask: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: str = 'ellipse',
    opening_iterations: int = 1,
    closing_iterations: int = 1
) -> np.ndarray:
    """
    Apply morphological opening and closing to remove noise and fill holes.
    
    Args:
        mask: Binary mask (H, W)
        kernel_size: Size of morphological kernel
        kernel_shape: Shape of kernel
        opening_iterations: Number of opening iterations
        closing_iterations: Number of closing iterations
        
    Returns:
        Refined binary mask
    """
    # Ensure binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Create kernel
    kernel = create_morphology_kernel(kernel_size, kernel_shape)
    
    # Apply opening (erosion followed by dilation) to remove small objects
    if opening_iterations > 0:
        for _ in range(opening_iterations):
            mask_binary = binary_opening(mask_binary, structure=kernel).astype(np.uint8)
    
    # Apply closing (dilation followed by erosion) to fill small holes
    if closing_iterations > 0:
        for _ in range(closing_iterations):
            mask_binary = binary_closing(mask_binary, structure=kernel).astype(np.uint8)
    
    return mask_binary


def remove_small_components(
    mask: np.ndarray,
    min_size: int = 100,
    connectivity: int = 2
) -> np.ndarray:
    """
    Remove small connected components from binary mask.
    
    Args:
        mask: Binary mask (H, W)
        min_size: Minimum size of components to keep (pixels)
        connectivity: Connectivity for component analysis (1 or 2)
        
    Returns:
        Filtered binary mask
    """
    # Ensure binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Remove small objects
    cleaned = morphology.remove_small_objects(
        mask_binary.astype(bool),
        min_size=min_size,
        connectivity=connectivity
    ).astype(np.uint8)
    
    return cleaned


def fill_holes(
    mask: np.ndarray,
    max_hole_size: Optional[int] = None
) -> np.ndarray:
    """
    Fill holes in binary mask.
    
    Args:
        mask: Binary mask (H, W)
        max_hole_size: Maximum size of holes to fill (None = fill all holes)
        
    Returns:
        Mask with holes filled
    """
    # Ensure binary
    mask_binary = (mask > 0).astype(bool)
    
    if max_hole_size is None:
        # Fill all holes
        filled = binary_fill_holes(mask_binary).astype(np.uint8)
    else:
        # Fill only small holes
        # Invert mask to treat holes as objects
        inverted = ~mask_binary
        
        # Remove small objects (holes) from inverted mask
        cleaned_inverted = morphology.remove_small_objects(
            inverted,
            min_size=max_hole_size,
            connectivity=2
        )
        
        # Invert back
        filled = (~cleaned_inverted).astype(np.uint8)
    
    return filled


def filter_by_area(
    mask: np.ndarray,
    min_area: int = 100,
    max_area: Optional[int] = None
) -> np.ndarray:
    """
    Filter connected components by area.
    
    Args:
        mask: Binary mask (H, W)
        min_area: Minimum area (pixels)
        max_area: Maximum area (pixels, None = no maximum)
        
    Returns:
        Filtered mask
    """
    # Label connected components
    labeled = measure.label(mask > 0, connectivity=2)
    
    # Get region properties
    regions = measure.regionprops(labeled)
    
    # Create filtered mask
    filtered = np.zeros_like(mask, dtype=np.uint8)
    
    for region in regions:
        area = region.area
        
        # Check area constraints
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        
        # Keep this component
        coords = region.coords
        filtered[coords[:, 0], coords[:, 1]] = 1
    
    return filtered


def filter_by_shape(
    mask: np.ndarray,
    min_solidity: float = 0.5,
    min_extent: float = 0.3
) -> np.ndarray:
    """
    Filter connected components by shape metrics.
    
    Args:
        mask: Binary mask (H, W)
        min_solidity: Minimum solidity (area / convex_area)
        min_extent: Minimum extent (area / bounding_box_area)
        
    Returns:
        Filtered mask
    """
    # Label connected components
    labeled = measure.label(mask > 0, connectivity=2)
    
    # Get region properties
    regions = measure.regionprops(labeled)
    
    # Create filtered mask
    filtered = np.zeros_like(mask, dtype=np.uint8)
    
    for region in regions:
        # Calculate shape metrics
        solidity = region.solidity
        extent = region.extent
        
        # Check shape constraints
        if solidity < min_solidity:
            continue
        if extent < min_extent:
            continue
        
        # Keep this component
        coords = region.coords
        filtered[coords[:, 0], coords[:, 1]] = 1
    
    return filtered


def refine_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: str = 'ellipse',
    opening_iterations: int = 1,
    closing_iterations: int = 1,
    min_component_size: int = 100,
    max_component_size: Optional[int] = None,
    fill_small_holes: bool = True,
    max_hole_size: Optional[int] = None
) -> np.ndarray:
    """
    Complete refinement pipeline for binary change masks.
    
    Args:
        mask: Binary mask (H, W)
        kernel_size: Size of morphological kernel
        kernel_shape: Shape of kernel
        opening_iterations: Number of opening iterations
        closing_iterations: Number of closing iterations
        min_component_size: Minimum size of components to keep
        max_component_size: Maximum size of components to keep
        fill_small_holes: Whether to fill small holes
        max_hole_size: Maximum size of holes to fill
        
    Returns:
        Refined binary mask
    """
    # Ensure binary
    mask_refined = (mask > 0).astype(np.uint8)
    
    # Step 1: Morphological operations
    mask_refined = apply_morphological_operations(
        mask_refined,
        kernel_size=kernel_size,
        kernel_shape=kernel_shape,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations
    )
    
    # Step 2: Fill holes
    if fill_small_holes:
        mask_refined = fill_holes(mask_refined, max_hole_size=max_hole_size)
    
    # Step 3: Remove small components
    if min_component_size > 0:
        mask_refined = filter_by_area(
            mask_refined,
            min_area=min_component_size,
            max_area=max_component_size
        )
    
    return mask_refined


def get_component_statistics(mask: np.ndarray) -> dict:
    """
    Calculate statistics about connected components in mask.
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        Dictionary with statistics
    """
    # Label components
    labeled = measure.label(mask > 0, connectivity=2)
    n_components = labeled.max()
    
    if n_components == 0:
        return {
            'n_components': 0,
            'total_area': 0,
            'mean_area': 0,
            'median_area': 0,
            'min_area': 0,
            'max_area': 0
        }
    
    # Get region properties
    regions = measure.regionprops(labeled)
    areas = [r.area for r in regions]
    
    stats = {
        'n_components': n_components,
        'total_area': sum(areas),
        'mean_area': np.mean(areas),
        'median_area': np.median(areas),
        'min_area': min(areas),
        'max_area': max(areas)
    }
    
    return stats


def smooth_boundaries(
    mask: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Smooth component boundaries using Gaussian filter.
    
    Args:
        mask: Binary mask (H, W)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed binary mask
    """
    # Convert to float
    mask_float = mask.astype(np.float32)
    
    # Apply Gaussian smoothing
    smoothed = ndimage.gaussian_filter(mask_float, sigma=sigma)
    
    # Re-binarize
    binary = (smoothed > 0.5).astype(np.uint8)
    
    return binary


if __name__ == '__main__':
    # Test post-processing
    mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
    
    print("Original mask:")
    stats = get_component_statistics(mask)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    refined = refine_binary_mask(mask, min_component_size=50)
    
    print("\nRefined mask:")
    stats = get_component_statistics(refined)
    for key, value in stats.items():
        print(f"  {key}: {value}")
