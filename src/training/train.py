"""
Training script for change detection model.
Supports training on GPU (Kaggle/Colab) with full logging and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import time
from typing import Dict, Optional
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import ChangeDetectionDataset, create_dataloaders
from models.siamese_unet import create_model, count_parameters
from models.losses import create_loss
from utils.metrics import MetricsCalculator, calculate_all_metrics
from utils.visualization import plot_predictions, plot_training_history


class Trainer:
    """
    Trainer class for change detection model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        device: torch.device,
        config: dict
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Training device
            config: Configuration dictionary
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Mixed precision training
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_f1': [],
            'val_f1': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.early_stopping_patience = config['training'].get('early_stopping_patience', 15)
        self.early_stopping_counter = 0
        
        # Paths
        self.output_dir = Path(config['paths'].get('output_dir', 'outputs'))
        self.models_dir = Path(config['paths'].get('models_dir', 'models'))
        self.logs_dir = Path(config['paths'].get('logs_dir', 'logs'))
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        metrics_calc = MetricsCalculator()
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, (img1, img2, mask) in enumerate(pbar):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(img1, img2)
                    loss = self.criterion(output, mask)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(img1, img2)
                loss = self.criterion(output, mask)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            metrics_calc.update(output.detach(), mask.detach())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Compute average metrics
        avg_loss = epoch_loss / len(train_loader)
        metrics = metrics_calc.compute()
        
        return {
            'loss': avg_loss,
            'iou': metrics['iou'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
    
    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        epoch_loss = 0.0
        metrics_calc = MetricsCalculator()
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        for batch_idx, (img1, img2, mask) in enumerate(pbar):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            output = self.model(img1, img2)
            loss = self.criterion(output, mask)
            
            # Update metrics
            epoch_loss += loss.item()
            metrics_calc.update(output, mask)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute average metrics
        avg_loss = epoch_loss / len(val_loader)
        metrics = metrics_calc.compute()
        
        return {
            'loss': avg_loss,
            'iou': metrics['iou'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        }
    
    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric': self.best_val_metric,
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_path = self.models_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.models_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            val_metric = val_metrics['iou']  # Use IoU as primary metric
            is_best = val_metric > self.best_val_metric
            
            if is_best:
                self.best_val_metric = val_metric
                self.early_stopping_counter = 0
                print(f"  ★ New best validation IoU: {val_metric:.4f}")
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_frequency', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best=is_best)
            elif is_best:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best=True)
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"  Best validation IoU: {self.best_val_metric:.4f}")
                break
        
        # Save final checkpoint
        self.save_checkpoint('final_model.pth')
        
        # Plot training history
        plot_training_history(
            self.history,
            metrics=['loss', 'iou', 'f1'],
            save_path=self.output_dir / 'training_history.png'
        )
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation IoU: {self.best_val_metric:.4f}")
        print(f"Models saved to: {self.models_dir}")
        print(f"{'='*60}\n")


def create_optimizer(
    model: nn.Module,
    config: dict
) -> optim.Optimizer:
    """
    Create optimizer from config.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer
    """
    optimizer_type = config['training'].get('optimizer', 'adamw').lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 1e-5)
    
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=config['training'].get('betas', [0.9, 0.999]),
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
    num_epochs: int
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler from config.
    
    Args:
        optimizer: Optimizer
        config: Configuration dictionary
        num_epochs: Total number of epochs
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config['training'].get('scheduler', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('step_size', 10),
            gamma=config['training'].get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def main(config_path: Path, resume: Optional[Path] = None):
    """
    Main training function.
    
    Args:
        config_path: Path to configuration file
        resume: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_dir = Path(config['paths']['patches_dir']) / 'train'
    val_dir = Path(config['paths']['patches_dir']) / 'val'
    
    print(f"\nLoading data from:")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")
    
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=config['training'].get('pin_memory', True),
        augment_train=True
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} patches")
    print(f"  Val: {len(val_loader.dataset)} patches")
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(
        model_type='simple',
        encoder_name=config['model']['encoder_name'],
        encoder_weights=config['model']['encoder_weights'],
        in_channels=config['model']['in_channels'],
        classes=config['model']['classes'],
        activation=config['model'].get('activation', 'sigmoid')
    )
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function
    criterion = create_loss(
        loss_type='combined',
        dice_weight=config['loss']['dice_weight'],
        focal_weight=config['loss']['focal_weight'],
        focal_alpha=config['loss'].get('focal_alpha', 0.25),
        focal_gamma=config['loss'].get('focal_gamma', 2.0)
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    num_epochs = config['training']['epochs']
    scheduler = create_scheduler(optimizer, config, num_epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Resume from checkpoint if specified
    if resume:
        trainer.load_checkpoint(resume)
    
    # Train
    trainer.train(train_loader, val_loader, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train change detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    main(Path(args.config), Path(args.resume) if args.resume else None)
