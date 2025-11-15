"""
Script 1: Generate patches from ResourceSat-2 image pairs.
Processes large images in memory-efficient windowed mode.

Usage:
    python scripts/1_generate_patches.py --data_dir Dataset --output_dir data/patches --config configs/config.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.patch_generator import PatchGenerator, find_resourcesat2_bands
from data.preprocessing import verify_image_pair


def main():
    parser = argparse.ArgumentParser(description='Generate patches from satellite image pairs')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='Dataset',
        help='Directory containing satellite data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/patches',
        help='Directory to save generated patches'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=None,
        help='Training split ratio (overrides config)'
    )
    parser.add_argument(
        '--skip_split',
        action='store_true',
        help='Skip train/val splitting'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters from config
    patch_size = config['data']['patch_size']
    overlap = config['data']['overlap']
    train_split = args.train_split if args.train_split else config['data']['train_split']
    
    print(f"\n{'='*60}")
    print("PATCH GENERATION FOR CHANGE DETECTION")
    print(f"{'='*60}\n")
    print(f"Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Overlap: {overlap} pixels")
    print(f"  Train split: {train_split}")
    print(f"{'='*60}\n")
    
    # Find ResourceSat-2 image pairs
    data_dir = Path(args.data_dir)
    print("Searching for ResourceSat-2 data...")
    
    image_pairs = find_resourcesat2_bands(data_dir)
    
    if len(image_pairs) == 0:
        print("❌ No ResourceSat-2 data found!")
        print("Expected directory structure: Dataset/R2F**/R2F**/BAND*.tif")
        return
    
    print(f"Found {len(image_pairs)} time points:")
    for date, bands in image_pairs.items():
        print(f"  {date}: {len(bands)} bands")
    
    # Verify we have at least 2 time points for change detection
    if len(image_pairs) < 2:
        print("\n⚠ Warning: Only one time point found!")
        print("Change detection requires at least 2 temporal images.")
        print("Generating patches for single time point...")
        
        # Process single time point
        date, band_files = list(image_pairs.items())[0]
        
        # Create patch generator
        generator = PatchGenerator(
            patch_size=patch_size,
            overlap=overlap,
            min_valid_ratio=config['data'].get('min_valid_pixels_ratio', 0.2),
            max_nodata_ratio=config['data'].get('max_nodata_ratio', 0.8)
        )
        
        # Generate patches for each band
        output_dir = Path(args.output_dir) / 'single'
        for band_file in band_files:
            generator.generate_patches_from_image(
                image_path=band_file,
                output_dir=output_dir,
                prefix=f"{date}_{band_file.stem}",
                verbose=True
            )
        
        print(f"\n✓ Patches saved to {output_dir}")
        return
    
    # Sort dates
    dates = sorted(image_pairs.keys())
    
    print(f"\n{'='*60}")
    print("PROCESSING IMAGE PAIRS")
    print(f"{'='*60}\n")
    
    # Create patch generator
    generator = PatchGenerator(
        patch_size=patch_size,
        overlap=overlap,
        min_valid_ratio=config['data'].get('min_valid_pixels_ratio', 0.2),
        max_nodata_ratio=config['data'].get('max_nodata_ratio', 0.8),
        target_bands=[0, 1, 2]  # Use all 3 bands (Green, Red, NIR)
    )
    
    # Process pairs: consecutive dates
    output_dir = Path(args.output_dir) / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_patches = 0
    
    for i in range(len(dates) - 1):
        date1 = dates[i]
        date2 = dates[i + 1]
        
        print(f"\nProcessing pair: {date1} -> {date2}")
        
        # Get band files (need to stack them into single image)
        # For ResourceSat-2, we have BAND2, BAND3, BAND4 as separate files
        # We'll use the first band file as reference
        band_files1 = image_pairs[date1]
        band_files2 = image_pairs[date2]
        
        # Use first band as reference (they should all have same dimensions)
        image1_path = band_files1[0]
        image2_path = band_files2[0]
        
        # Verify images are compatible (allow shape mismatch - will be cropped)
        is_valid, message = verify_image_pair(image1_path, image2_path)
        if not is_valid and "Shape mismatch" not in message:
            print(f"  ⚠ Warning: {message}")
            print(f"  Skipping this pair...")
            continue
        elif not is_valid:
            print(f"  ⚠ Info: {message} - will crop to overlapping region")
        
        # For ResourceSat-2 with separate bands, we need to stack them into one multi-band image
        # We'll process all bands together
        print(f"  Stacking {len(band_files1)} bands and generating patches...")
        
        # Generate patches from stacked bands
        n_patches = generator.generate_patches_from_stacked_bands(
            band_files1=band_files1,
            band_files2=band_files2,
            output_dir=output_dir / f"pair_{date1}_{date2}",
            mask_path=None,  # No ground truth masks available
            verbose=True
        )
        
        total_patches += n_patches
        print(f"  Generated {n_patches} patches")
    
    print(f"\n{'='*60}")
    print(f"✓ Generated {total_patches} total patches")
    print(f"{'='*60}\n")
    
    # Split into train/val if requested
    if not args.skip_split and total_patches > 0:
        print("Splitting patches into train/val sets...")
        
        try:
            train_count, val_count = generator.split_patches(
                patches_dir=output_dir,
                output_base_dir=Path(args.output_dir),
                train_ratio=train_split,
                seed=42
            )
            
            print(f"\n✓ Split complete:")
            print(f"  Training patches: {train_count}")
            print(f"  Validation patches: {val_count}")
            print(f"  Train ratio: {train_count / (train_count + val_count):.2%}")
        
        except Exception as e:
            print(f"\n⚠ Error during splitting: {e}")
            print(f"Raw patches are still available in: {output_dir}")
    
    print(f"\n{'='*60}")
    print("PATCH GENERATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Output directory: {args.output_dir}")
    print(f"Next step: Upload patches to Kaggle/Colab and run training")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
