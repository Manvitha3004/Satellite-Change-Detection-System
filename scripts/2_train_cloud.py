"""
Script 2: Training script for cloud platforms (Kaggle/Colab).
Standalone script with minimal dependencies on local paths.

Usage (Kaggle/Colab):
    python 2_train_cloud.py --patches_dir /kaggle/input/patches --output_dir /kaggle/working
"""

import argparse
import sys
from pathlib import Path

# Import training module
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from training.train import main as train_main

def main():
    parser = argparse.ArgumentParser(description='Train change detection model on cloud')
    parser.add_argument(
        '--patches_dir',
        type=str,
        required=True,
        help='Directory containing train/val patches'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='Directory for outputs'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Modify config paths for cloud environment
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override paths
    config['paths']['patches_dir'] = args.patches_dir
    config['paths']['output_dir'] = args.output_dir
    config['paths']['models_dir'] = str(Path(args.output_dir) / 'models')
    config['paths']['logs_dir'] = str(Path(args.output_dir) / 'logs')
    
    # Override training parameters if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Save modified config
    temp_config = Path(args.output_dir) / 'config_cloud.yaml'
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    train_main(temp_config, Path(args.resume) if args.resume else None)

if __name__ == '__main__':
    main()
