"""
Siamese U-Net architecture for change detection.
Uses EfficientNet-B0 backbone with shared weights and U-Net decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Optional, List


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for bi-temporal change detection.
    
    Architecture:
    - Shared EfficientNet-B0 encoder for both time points
    - Feature concatenation for fusion
    - U-Net decoder with skip connections
    - Final sigmoid activation for binary change mask
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b0',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = 'sigmoid',
        decoder_channels: List[int] = [256, 128, 64, 32, 16]
    ):
        """
        Initialize Siamese U-Net.
        
        Args:
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Output activation function
            decoder_channels: List of decoder channel sizes
        """
        super(SiameseUNet, self).__init__()
        
        self.encoder_name = encoder_name
        self.in_channels = in_channels
        self.classes = classes
        
        # Create shared encoder using segmentation_models_pytorch
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_channels=decoder_channels
        )
        
        # For feature extraction, we'll use the encoder part
        # and create a custom fusion + decoder
        
        # Alternative: Use full U-Net for each branch and fuse features
        self.branch1 = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=128,  # Feature extraction
            activation=None
        )
        
        # Share weights between branches
        self.branch2 = self.branch1
        
        # Fusion and final prediction layers
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classes, kernel_size=1)
        )
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, name: Optional[str]) -> Optional[nn.Module]:
        """Get activation function by name."""
        if name is None or name == 'identity':
            return nn.Identity()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'softmax':
            return nn.Softmax(dim=1)
        elif name == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x1: First temporal image (B, C, H, W)
            x2: Second temporal image (B, C, H, W)
            
        Returns:
            Change probability map (B, 1, H, W)
        """
        # Extract features from both time points
        features1 = self.branch1(x1)
        features2 = self.branch2(x2)
        
        # Concatenate features
        fused = torch.cat([features1, features2], dim=1)
        
        # Final prediction
        output = self.fusion(fused)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
        
        return output


class SimpleSiameseUNet(nn.Module):
    """
    Simplified Siamese U-Net using segmentation_models_pytorch directly.
    
    This version processes concatenated images through a single U-Net,
    which is more memory efficient than true Siamese architecture.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b0',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = 'sigmoid',
        decoder_channels: List[int] = [256, 128, 64, 32, 16]
    ):
        """
        Initialize Simplified Siamese U-Net.
        
        Args:
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels per image
            classes: Number of output classes
            activation: Output activation function
            decoder_channels: List of decoder channel sizes
        """
        super(SimpleSiameseUNet, self).__init__()
        
        # Create U-Net that takes concatenated images (2 * in_channels)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,  # Can't use pretrained with 2x channels
            in_channels=in_channels * 2,  # Concatenate both temporal images
            classes=classes,
            activation=activation,
            decoder_channels=decoder_channels
        )
        
        # If pretrained weights requested, load them for first branch
        if encoder_weights is not None:
            self._init_with_pretrained(encoder_name, encoder_weights, in_channels)
    
    def _init_with_pretrained(
        self, 
        encoder_name: str, 
        encoder_weights: str,
        in_channels: int
    ):
        """Initialize encoder with pretrained weights."""
        try:
            # Create temporary model with pretrained weights
            temp_model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=1
            )
            
            # Copy pretrained weights to first half of input layer
            encoder_weights_dict = temp_model.encoder.state_dict()
            
            # Get first conv layer name (varies by encoder)
            first_conv_keys = [k for k in encoder_weights_dict.keys() if 'conv' in k.lower() and 'weight' in k]
            
            if first_conv_keys:
                first_key = first_conv_keys[0]
                pretrained_conv = encoder_weights_dict[first_key]
                
                # Initialize our model's first conv with pretrained + duplicated weights
                our_conv_key = first_key
                if our_conv_key in self.model.encoder.state_dict():
                    our_conv = self.model.encoder.state_dict()[our_conv_key]
                    # Copy weights for first temporal image
                    our_conv[:, :in_channels, :, :] = pretrained_conv
                    # Copy same weights for second temporal image
                    our_conv[:, in_channels:, :, :] = pretrained_conv
            
            print(f"Initialized encoder with {encoder_weights} pretrained weights")
        
        except Exception as e:
            print(f"Warning: Could not initialize with pretrained weights: {e}")
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x1: First temporal image (B, C, H, W)
            x2: Second temporal image (B, C, H, W)
            
        Returns:
            Change probability map (B, 1, H, W)
        """
        # Concatenate temporal images along channel dimension
        x = torch.cat([x1, x2], dim=1)
        
        # Forward through U-Net
        output = self.model(x)
        
        return output


class EarlyFusionSiameseUNet(nn.Module):
    """
    Siamese U-Net with early fusion strategy.
    Computes difference and concatenation of features.
    """
    
    def __init__(
        self,
        encoder_name: str = 'efficientnet-b0',
        encoder_weights: Optional[str] = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = 'sigmoid'
    ):
        """
        Initialize Early Fusion Siamese U-Net.
        
        Args:
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights
            in_channels: Number of input channels per image
            classes: Number of output classes
            activation: Output activation function
        """
        super(EarlyFusionSiameseUNet, self).__init__()
        
        # Create encoder for feature extraction
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        
        # Get encoder output channels
        encoder_channels = self.encoder.out_channels
        
        # Create decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=[256, 128, 64, 32, 16],
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=classes,
            activation=activation,
            kernel_size=3
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            x1: First temporal image (B, C, H, W)
            x2: Second temporal image (B, C, H, W)
            
        Returns:
            Change probability map (B, 1, H, W)
        """
        # Extract features from both branches (shared encoder)
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)
        
        # Compute difference and concatenation at each level
        fused_features = []
        for f1, f2 in zip(features1, features2):
            diff = torch.abs(f1 - f2)
            concat = torch.cat([f1, f2], dim=1)
            # Average pooling to reduce channels
            fused = F.adaptive_avg_pool2d(concat, f1.shape[2:])
            fused_features.append(fused)
        
        # Decode
        decoder_output = self.decoder(*fused_features)
        
        # Segmentation head
        output = self.segmentation_head(decoder_output)
        
        return output


def create_model(
    model_type: str = 'simple',
    encoder_name: str = 'efficientnet-b0',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    classes: int = 1,
    activation: str = 'sigmoid'
) -> nn.Module:
    """
    Factory function to create change detection model.
    
    Args:
        model_type: Type of model ('simple', 'siamese', 'early_fusion')
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels per image
        classes: Number of output classes
        activation: Output activation function
        
    Returns:
        Change detection model
    """
    if model_type == 'simple':
        return SimpleSiameseUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif model_type == 'siamese':
        return SiameseUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    elif model_type == 'early_fusion':
        return EarlyFusionSiameseUNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation
    model = create_model(model_type='simple', in_channels=3)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    with torch.no_grad():
        output = model(x1, x2)
    
    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {output.shape}")
