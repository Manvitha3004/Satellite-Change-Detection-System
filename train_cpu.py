"""
CPU-optimized training script for local development.
Uses smaller batch size and simpler architecture for faster training on CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from models.siamese_unet import SimpleSiameseUNet
from models.losses import CombinedLoss
from utils.metrics import calculate_iou, calculate_f1

# Simple dataset class
class ChangeDetectionDataset(Dataset):
    def __init__(self, patches_dir):
        self.patches_dir = Path(patches_dir)
        self.patch_files = sorted(list(self.patches_dir.glob('*.npz')))
        print(f"Loaded {len(self.patch_files)} patches from {patches_dir}")
    
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        data = np.load(self.patch_files[idx])
        
        # Get images
        image1 = data['image1']  # (C, H, W)
        image2 = data['image2']
        mask = data.get('mask', np.zeros(image1.shape[1:], dtype=np.float32))
        
        # Normalize
        image1 = self._normalize(image1)
        image2 = self._normalize(image2)
        
        # Convert to tensors
        image1 = torch.from_numpy(image1).float()
        image2 = torch.from_numpy(image2).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image1, image2, mask
    
    def _normalize(self, img):
        """Percentile normalization"""
        img = img.astype(np.float32)
        p2 = np.percentile(img, 2)
        p98 = np.percentile(img, 98)
        img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1)
        return img

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(loader, desc='Training')
    for img1, img2, mask in pbar:
        img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        iou = calculate_iou((output > 0.5).float().cpu().numpy(), mask.cpu().numpy())
        total_iou += iou
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    return total_loss / len(loader), total_iou / len(loader)

@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    
    pbar = tqdm(loader, desc='Validation')
    for img1, img2, mask in pbar:
        img1, img2, mask = img1.to(device), img2.to(device), mask.to(device)
        
        output = model(img1, img2)
        loss = criterion(output, mask)
        
        pred_binary = (output > 0.5).float().cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        iou = calculate_iou(pred_binary, mask_np)
        f1 = calculate_f1(pred_binary, mask_np)
        
        total_loss += loss.item()
        total_iou += iou
        total_f1 += f1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iou': f'{iou:.4f}'})
    
    return total_loss / len(loader), total_iou / len(loader), total_f1 / len(loader)

def main():
    print("="*60)
    print("CHANGE DETECTION MODEL TRAINING (CPU)")
    print("="*60)
    
    # Configuration
    config = {
        'batch_size': 4,  # Small batch for CPU
        'epochs': 30,
        'learning_rate': 0.001,
        'device': 'cpu',
        'patience': 10
    }
    
    device = torch.device(config['device'])
    print(f"\nDevice: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    train_dataset = ChangeDetectionDataset('data/patches/train')
    val_dataset = ChangeDetectionDataset('data/patches/val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    model = SimpleSiameseUNet(
        encoder_name='resnet18',  # Lighter encoder for CPU
        in_channels=3,
        classes=1
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [], 'val_f1': []
    }
    best_val_iou = 0
    patience_counter = 0
    
    # Create output directory
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"★ New best model! Val IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Plot training history
    print("\n" + "="*60)
    print("Plotting training history...")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU
    axes[1].plot(history['train_iou'], label='Train IoU', linewidth=2)
    axes[1].plot(history['val_iou'], label='Val IoU', linewidth=2)
    axes[1].plot(history['val_f1'], label='Val F1', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training History - Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Saved training_history.png")
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Final Val F1: {history['val_f1'][-1]:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pth'}")
    print("\nNext: Run inference with scripts/3_inference_local.py")
    print("="*60)

if __name__ == '__main__':
    main()
