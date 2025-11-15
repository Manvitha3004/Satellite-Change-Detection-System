"""
Visualization utilities for training and inference.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Union
import seaborn as sns


def plot_predictions(
    image1: Union[np.ndarray, torch.Tensor],
    image2: Union[np.ndarray, torch.Tensor],
    pred_mask: Union[np.ndarray, torch.Tensor],
    true_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (20, 5),
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Visualize change detection predictions.
    
    Args:
        image1: First temporal image (C, H, W) or (H, W, C)
        image2: Second temporal image (C, H, W) or (H, W, C)
        pred_mask: Predicted change mask (H, W) or (1, H, W)
        true_mask: Ground truth mask (optional)
        threshold: Threshold for binarizing predictions
        figsize: Figure size
        save_path: Path to save figure
        title: Figure title
    """
    # Convert tensors to numpy
    if isinstance(image1, torch.Tensor):
        image1 = image1.detach().cpu().numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.detach().cpu().numpy()
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if true_mask is not None and isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.detach().cpu().numpy()
    
    # Handle different input shapes
    if image1.ndim == 3 and image1.shape[0] <= 10:
        image1 = np.moveaxis(image1, 0, -1)  # (C, H, W) -> (H, W, C)
    if image2.ndim == 3 and image2.shape[0] <= 10:
        image2 = np.moveaxis(image2, 0, -1)
    
    # Remove channel dimension from mask if present
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0] if pred_mask.shape[0] == 1 else pred_mask
    if true_mask is not None and true_mask.ndim == 3:
        true_mask = true_mask[0] if true_mask.shape[0] == 1 else true_mask
    
    # Binarize prediction
    pred_binary = (pred_mask >= threshold).astype(np.uint8)
    
    # Normalize images for display
    image1_display = normalize_for_display(image1)
    image2_display = normalize_for_display(image2)
    
    # Create figure
    n_cols = 4 if true_mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    
    # Plot T1 image
    axes[0].imshow(image1_display)
    axes[0].set_title('Time 1 (T1)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot T2 image
    axes[1].imshow(image2_display)
    axes[1].set_title('Time 2 (T2)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(pred_binary, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title('Predicted Change', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Plot ground truth if available
    if true_mask is not None:
        axes[3].imshow(true_mask, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[3].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[3].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: dict,
    metrics: List[str] = ['loss', 'iou', 'f1'],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            axes[idx].plot(history[train_key], label='Train', linewidth=2)
        if val_key in history:
            axes[idx].plot(history[val_key], label='Validation', linewidth=2)
        
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric.upper(), fontsize=12)
        axes[idx].set_title(f'{metric.upper()} History', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None
):
    """
    Plot confusion matrix.
    
    Args:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
        figsize: Figure size
        save_path: Path to save figure
    """
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Change', 'Change'],
        yticklabels=['No Change', 'Change'],
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_overlay(
    image: Union[np.ndarray, torch.Tensor],
    mask: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
    color: str = 'red',
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Plot image with mask overlay.
    
    Args:
        image: Background image (C, H, W) or (H, W, C)
        mask: Binary mask (H, W)
        alpha: Transparency of overlay
        color: Color of overlay ('red', 'blue', 'green', 'yellow')
        figsize: Figure size
        save_path: Path to save figure
        title: Figure title
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Handle different input shapes
    if image.ndim == 3 and image.shape[0] <= 10:
        image = np.moveaxis(image, 0, -1)  # (C, H, W) -> (H, W, C)
    
    # Remove channel dimension from mask if present
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask
    
    # Normalize image for display
    image_display = normalize_for_display(image)
    
    # Create colored overlay
    overlay = np.zeros_like(image_display)
    color_map = {
        'red': [1, 0, 0],
        'blue': [0, 0, 1],
        'green': [0, 1, 0],
        'yellow': [1, 1, 0]
    }
    
    color_rgb = color_map.get(color, [1, 0, 0])
    for i in range(3):
        overlay[:, :, i] = mask * color_rgb[i]
    
    # Blend image and overlay
    blended = image_display * (1 - alpha * mask[:, :, np.newaxis]) + overlay * alpha
    blended = np.clip(blended, 0, 1)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(blended)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def normalize_for_display(
    image: np.ndarray,
    percentile: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    Normalize image for display.
    
    Args:
        image: Input image
        percentile: Percentile range for normalization
        
    Returns:
        Normalized image in [0, 1] range
    """
    # Handle grayscale
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    
    # Normalize each channel
    normalized = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[2]):
        channel = image[:, :, i].astype(np.float32)
        p_low, p_high = np.percentile(channel, percentile)
        
        if p_high > p_low:
            channel = np.clip((channel - p_low) / (p_high - p_low), 0, 1)
        else:
            channel = channel / (channel.max() + 1e-8)
        
        normalized[:, :, i] = channel
    
    # Convert to RGB if single channel
    if normalized.shape[2] == 1:
        normalized = np.repeat(normalized, 3, axis=2)
    
    # Use first 3 channels if more than 3
    if normalized.shape[2] > 3:
        # For multispectral: use NIR-R-G as false color
        if normalized.shape[2] >= 3:
            normalized = normalized[:, :, :3]
    
    return normalized


def create_comparison_grid(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 15),
    save_path: Optional[Path] = None
):
    """
    Create a grid comparing multiple samples.
    
    Args:
        samples: List of (image1, image2, pred, target) tuples
        titles: List of titles for each sample
        figsize: Figure size
        save_path: Path to save figure
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=figsize)
    
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    for idx, (img1, img2, pred, target) in enumerate(samples):
        # Normalize images
        img1_display = normalize_for_display(img1)
        img2_display = normalize_for_display(img2)
        
        # Plot T1
        axes[idx, 0].imshow(img1_display)
        axes[idx, 0].set_title('T1' if idx == 0 else '', fontsize=12)
        axes[idx, 0].axis('off')
        
        # Plot T2
        axes[idx, 1].imshow(img2_display)
        axes[idx, 1].set_title('T2' if idx == 0 else '', fontsize=12)
        axes[idx, 1].axis('off')
        
        # Plot prediction
        axes[idx, 2].imshow(pred, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[idx, 2].set_title('Prediction' if idx == 0 else '', fontsize=12)
        axes[idx, 2].axis('off')
        
        # Plot ground truth
        axes[idx, 3].imshow(target, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[idx, 3].set_title('Ground Truth' if idx == 0 else '', fontsize=12)
        axes[idx, 3].axis('off')
        
        # Add sample title on the left
        if titles and idx < len(titles):
            axes[idx, 0].set_ylabel(titles[idx], fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_prediction_as_rgb(
    pred_mask: np.ndarray,
    output_path: Path,
    colormap: str = 'jet'
):
    """
    Save prediction mask as RGB image.
    
    Args:
        pred_mask: Prediction mask (H, W)
        output_path: Path to save image
        colormap: Matplotlib colormap name
    """
    # Normalize to [0, 1]
    if pred_mask.max() > 1:
        pred_mask = pred_mask / pred_mask.max()
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(pred_mask)
    
    # Convert to uint8 RGB
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    # Save
    from PIL import Image
    Image.fromarray(rgb).save(output_path)


if __name__ == '__main__':
    # Test visualization
    img1 = np.random.rand(256, 256, 3)
    img2 = np.random.rand(256, 256, 3)
    pred = np.random.rand(256, 256)
    target = np.random.randint(0, 2, (256, 256))
    
    plot_predictions(img1, img2, pred, target, title='Test Prediction')
