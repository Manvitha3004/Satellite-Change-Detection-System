"""
Loss functions for change detection.
Implements Dice Loss, Focal Loss, and Combined Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Optimizes for overlap between prediction and ground truth.
    """
    
    def __init__(
        self,
        mode: str = 'binary',
        smooth: float = 1.0,
        eps: float = 1e-7
    ):
        """
        Initialize Dice Loss.
        
        Args:
            mode: Loss mode ('binary' or 'multiclass')
            smooth: Smoothing constant to avoid division by zero
            eps: Small epsilon for numerical stability
        """
        super(DiceLoss, self).__init__()
        self.mode = mode
        self.smooth = smooth
        self.eps = eps
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, C, H, W) or (B, H, W)
            
        Returns:
            Dice loss value
        """
        # Ensure target has same shape as pred
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Flatten predictions and targets
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1).float()
        
        # Compute Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        
        # Dice loss is 1 - Dice coefficient
        dice_loss = 1.0 - dice_coeff
        
        return dice_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses on hard-to-classify examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for class balance
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, C, H, W) or (B, H, W)
            
        Returns:
            Focal loss value
        """
        # Ensure target has same shape as pred
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Compute binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Compute focal term: (1 - pt)^gamma
        pt = torch.exp(-bce)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_term * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined Dice + Focal Loss for change detection.
    Balances region-based (Dice) and pixel-wise (Focal) optimization.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        dice_smooth: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Initialize Combined Loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            dice_smooth: Smoothing constant for Dice loss
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
        """
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss(mode='binary', smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted probabilities (B, 1, H, W)
            target: Ground truth masks (B, 1, H, W) or (B, H, W)
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        combined = self.dice_weight * dice + self.focal_weight * focal
        
        return combined


class SMPCombinedLoss(nn.Module):
    """
    Combined loss using segmentation_models_pytorch implementations.
    More optimized and battle-tested than custom implementations.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        mode: str = 'binary',
        smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        """
        Initialize SMP Combined Loss.
        
        Args:
            dice_weight: Weight for Dice loss
            focal_weight: Weight for Focal loss
            mode: Loss mode ('binary' or 'multiclass')
            smooth: Smoothing for Dice loss
            alpha: Alpha for Focal loss
            gamma: Gamma for Focal loss
        """
        super(SMPCombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # Use SMP loss implementations
        self.dice_loss = smp.losses.DiceLoss(mode=mode, smooth=smooth)
        self.focal_loss = smp.losses.FocalLoss(mode=mode, alpha=alpha, gamma=gamma)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits or probabilities (B, C, H, W)
            target: Ground truth masks (B, C, H, W) or (B, H, W)
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        combined = self.dice_weight * dice + self.focal_weight * focal
        
        return combined


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with asymmetric penalties.
    Useful for handling false positives and false negatives differently.
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0
    ):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing constant
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky Loss.
        
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, C, H, W) or (B, H, W)
            
        Returns:
            Tversky loss value
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum(dim=1)
        fp = ((1 - target) * pred).sum(dim=1)
        fn = (target * (1 - pred)).sum(dim=1)
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return (1 - tversky).mean()


def create_loss(
    loss_type: str = 'combined',
    dice_weight: float = 0.5,
    focal_weight: float = 0.5,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss ('dice', 'focal', 'combined', 'tversky')
        dice_weight: Weight for Dice loss (if using combined)
        focal_weight: Weight for Focal loss (if using combined)
        **kwargs: Additional loss-specific parameters
        
    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'combined':
        return SMPCombinedLoss(
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            **kwargs
        )
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    pred = torch.sigmoid(torch.randn(batch_size, 1, 256, 256))
    target = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(pred, target).item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(pred, target).item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss()
    print(f"Combined Loss: {combined_loss(pred, target).item():.4f}")
    
    # Test SMP Combined Loss
    smp_loss = SMPCombinedLoss()
    print(f"SMP Combined Loss: {smp_loss(pred, target).item():.4f}")
