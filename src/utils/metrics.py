"""
Evaluation metrics for change detection.
"""

import numpy as np
import torch
from typing import Tuple, Union, Optional
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, precision_score, recall_score


def calculate_iou(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        eps: Small epsilon for numerical stability
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Binarize predictions
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = (target >= threshold).astype(np.uint8)
    
    # Calculate intersection and union
    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary | target_binary)
    
    # Handle edge case: both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = (intersection + eps) / (union + eps)
    
    return float(iou)


def calculate_dice(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> float:
    """
    Calculate Dice coefficient (F1 score for segmentation).
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        eps: Small epsilon for numerical stability
        
    Returns:
        Dice coefficient (0.0 to 1.0)
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Binarize predictions
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = (target >= threshold).astype(np.uint8)
    
    # Calculate intersection
    intersection = np.sum(pred_binary & target_binary)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + eps) / (np.sum(pred_binary) + np.sum(target_binary) + eps)
    
    return float(dice)


def calculate_f1(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> float:
    """
    Calculate F1 score.
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Binarize predictions
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    
    # Calculate F1 score
    try:
        f1 = f1_score(target_binary, pred_binary, zero_division=0)
    except:
        # Fallback calculation
        precision = calculate_precision(pred, target, threshold)
        recall = calculate_recall(pred, target, threshold)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return float(f1)


def calculate_precision(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> float:
    """
    Calculate precision (positive predictive value).
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        Precision score (0.0 to 1.0)
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Binarize predictions
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    
    # Calculate precision
    try:
        precision = precision_score(target_binary, pred_binary, zero_division=0)
    except:
        # Manual calculation
        tp = np.sum(pred_binary & target_binary)
        fp = np.sum(pred_binary & ~target_binary)
        
        if tp + fp == 0:
            return 0.0
        
        precision = tp / (tp + fp)
    
    return float(precision)


def calculate_recall(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> float:
    """
    Calculate recall (sensitivity, true positive rate).
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Binarize predictions
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    
    # Calculate recall
    try:
        recall = recall_score(target_binary, pred_binary, zero_division=0)
    except:
        # Manual calculation
        tp = np.sum(pred_binary & target_binary)
        fn = np.sum(~pred_binary & target_binary)
        
        if tp + fn == 0:
            return 0.0
        
        recall = tp / (tp + fn)
    
    return float(recall)


def calculate_precision_recall(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Calculate both precision and recall.
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        Tuple of (precision, recall)
    """
    precision = calculate_precision(pred, target, threshold)
    recall = calculate_recall(pred, target, threshold)
    
    return precision, recall


def calculate_confusion_matrix(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Tuple[int, int, int, int]:
    """
    Calculate confusion matrix components.
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    # Convert to numpy if torch tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Flatten arrays
    pred = pred.flatten()
    target = target.flatten()
    
    # Binarize predictions
    pred_binary = (pred >= threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)
    
    # Calculate confusion matrix
    tn = np.sum((~pred_binary) & (~target_binary))
    tp = np.sum(pred_binary & target_binary)
    fn = np.sum((~pred_binary) & target_binary)
    fp = np.sum(pred_binary & (~target_binary))
    
    return int(tp), int(tn), int(fp), int(fn)


def calculate_all_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> dict:
    """
    Calculate all evaluation metrics at once.
    
    Args:
        pred: Predicted probabilities or binary mask
        target: Ground truth binary mask
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'iou': calculate_iou(pred, target, threshold),
        'dice': calculate_dice(pred, target, threshold),
        'f1': calculate_f1(pred, target, threshold),
        'precision': calculate_precision(pred, target, threshold),
        'recall': calculate_recall(pred, target, threshold)
    }
    
    # Add confusion matrix
    tp, tn, fp, fn = calculate_confusion_matrix(pred, target, threshold)
    metrics.update({
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    })
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    metrics['accuracy'] = accuracy
    
    return metrics


class MetricsCalculator:
    """
    Class for calculating and accumulating metrics across batches.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Threshold for binarizing predictions
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.f1_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.count = 0
    
    def update(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        target: Union[np.ndarray, torch.Tensor]
    ):
        """
        Update metrics with new batch.
        
        Args:
            pred: Predicted probabilities or binary mask
            target: Ground truth binary mask
        """
        metrics = calculate_all_metrics(pred, target, self.threshold)
        
        self.iou_sum += metrics['iou']
        self.dice_sum += metrics['dice']
        self.f1_sum += metrics['f1']
        self.precision_sum += metrics['precision']
        self.recall_sum += metrics['recall']
        self.count += 1
    
    def compute(self) -> dict:
        """
        Compute average metrics.
        
        Returns:
            Dictionary with average metrics
        """
        if self.count == 0:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        return {
            'iou': self.iou_sum / self.count,
            'dice': self.dice_sum / self.count,
            'f1': self.f1_sum / self.count,
            'precision': self.precision_sum / self.count,
            'recall': self.recall_sum / self.count
        }


if __name__ == '__main__':
    # Test metrics
    pred = np.random.rand(256, 256)
    target = np.random.randint(0, 2, (256, 256))
    
    metrics = calculate_all_metrics(pred, target)
    
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
