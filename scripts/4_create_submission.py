"""
Script 4: Create submission package.
Packages predictions into competition format.

Usage:
    python scripts/4_create_submission.py --predictions_dir outputs/predictions --output_zip submission.zip
"""

import argparse
from pathlib import Path
import zipfile
from datetime import datetime
import shutil


def main():
    parser = argparse.ArgumentParser(description='Create submission package')
    parser.add_argument(
        '--predictions_dir',
        type=str,
        required=True,
        help='Directory containing predictions'
    )
    parser.add_argument(
        '--output_zip',
        type=str,
        default=None,
        help='Output zip filename'
    )
    parser.add_argument(
        '--team_name',
        type=str,
        default='TeamAI',
        help='Team name for submission'
    )
    
    args = parser.parse_args()
    
    predictions_dir = Path(args.predictions_dir)
    
    if not predictions_dir.exists():
        print(f"❌ Predictions directory not found: {predictions_dir}")
        return
    
    # Generate output filename if not specified
    if args.output_zip:
        output_zip = Path(args.output_zip)
    else:
        date_str = datetime.now().strftime('%d-%b-%Y').upper()
        output_zip = Path(f"PS10_{date_str}_{args.team_name}.zip")
    
    print(f"\n{'='*60}")
    print("CREATING SUBMISSION PACKAGE")
    print(f"{'='*60}\n")
    print(f"Predictions directory: {predictions_dir}")
    print(f"Output: {output_zip}")
    print(f"Team: {args.team_name}")
    print(f"{'='*60}\n")
    
    # Find all prediction files
    tif_files = list(predictions_dir.glob('*.tif'))
    shp_files = list(predictions_dir.glob('*.shp'))
    shx_files = list(predictions_dir.glob('*.shx'))
    dbf_files = list(predictions_dir.glob('*.dbf'))
    prj_files = list(predictions_dir.glob('*.prj'))
    
    all_files = tif_files + shp_files + shx_files + dbf_files + prj_files
    
    if len(all_files) == 0:
        print("❌ No prediction files found!")
        return
    
    print(f"Found {len(all_files)} files:")
    print(f"  GeoTIFF: {len(tif_files)}")
    print(f"  Shapefiles: {len(shp_files)}")
    print(f"  Supporting: {len(shx_files + dbf_files + prj_files)}")
    
    # Create zip file
    print(f"\nCreating zip archive...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in all_files:
            zipf.write(file, arcname=file.name)
            print(f"  Added: {file.name}")
    
    # Get zip file size
    zip_size = output_zip.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n{'='*60}")
    print("✓ SUBMISSION PACKAGE CREATED")
    print(f"{'='*60}\n")
    print(f"File: {output_zip}")
    print(f"Size: {zip_size:.2f} MB")
    print(f"Files: {len(all_files)}")
    print(f"\nReady for submission!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
