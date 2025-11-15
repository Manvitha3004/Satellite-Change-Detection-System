"""
Script 3: Local inference on CPU.
Process image pairs and generate change masks + shapefiles.

Usage:
    python scripts/3_inference_local.py --model models/best_model.pth --image1_dir Dataset/R2F18MAR2020... --image2_dir Dataset/R2F27JAN2025... --output_dir outputs/predictions
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference.predict import load_model_for_inference, predict_from_band_files
from postprocessing.refine_mask import refine_binary_mask
from postprocessing.vectorize import raster_to_vector_pipeline
from data.preprocessing import read_and_stack_bands
import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser(description='Run inference on image pair')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--image1_dir',
        type=str,
        required=True,
        help='Directory containing T1 band files'
    )
    parser.add_argument(
        '--image2_dir',
        type=str,
        required=True,
        help='Directory containing T2 band files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/predictions',
        help='Output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output filename prefix (default: auto-generated)'
    )
    parser.add_argument(
        '--no_postprocess',
        action='store_true',
        help='Skip post-processing'
    )
    parser.add_argument(
        '--no_vectorize',
        action='store_true',
        help='Skip vectorization'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print("CHANGE DETECTION INFERENCE (CPU)")
    print(f"{'='*60}\n")
    
    # Setup paths
    image1_dir = Path(args.image1_dir)
    image2_dir = Path(args.image2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find band files
    print("Finding band files...")
    band_files1 = sorted(list(image1_dir.glob('BAND*.tif')))
    band_files2 = sorted(list(image2_dir.glob('BAND*.tif')))
    
    if len(band_files1) == 0 or len(band_files2) == 0:
        print("❌ No band files found!")
        print(f"T1 directory: {image1_dir}")
        print(f"T2 directory: {image2_dir}")
        return
    
    print(f"  T1: {len(band_files1)} bands")
    print(f"  T2: {len(band_files2)} bands")
    
    # Generate output name
    if args.output_name:
        output_name = args.output_name
    else:
        # Extract dates from directory names
        date1 = image1_dir.parent.name.split('F')[1][:11] if 'F' in image1_dir.parent.name else 'T1'
        date2 = image2_dir.parent.name.split('F')[1][:11] if 'F' in image2_dir.parent.name else 'T2'
        output_name = f"Change_Mask_{date1}_{date2}"
    
    raster_output = output_dir / f"{output_name}.tif"
    vector_output = output_dir / f"{output_name}.shp"
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    device = torch.device('cpu')
    model = load_model_for_inference(Path(args.model), device, config)
    
    # Run inference
    print(f"\nRunning inference...")
    prediction = predict_from_band_files(
        model=model,
        band_paths1=band_files1,
        band_paths2=band_files2,
        output_path=raster_output,
        device=device,
        tile_size=config['inference']['tile_size'],
        overlap=config['inference']['tile_overlap'],
        threshold=config['inference']['threshold'],
        band_indices=None  # Use all bands
    )
    
    print(f"✓ Saved raw prediction to {raster_output}")
    
    # Post-processing
    if not args.no_postprocess:
        print(f"\nApplying post-processing...")
        
        prediction_refined = refine_binary_mask(
            mask=prediction,
            kernel_size=config['postprocessing']['morphology_kernel_size'],
            kernel_shape=config['postprocessing'].get('morphology_kernel_shape', 'ellipse'),
            opening_iterations=config['postprocessing'].get('opening_iterations', 1),
            closing_iterations=config['postprocessing'].get('closing_iterations', 1),
            min_component_size=config['postprocessing']['min_component_size'],
            max_component_size=config['postprocessing'].get('max_component_size', None)
        )
        
        # Save refined mask
        refined_output = output_dir / f"{output_name}_refined.tif"
        
        with rasterio.open(raster_output) as src:
            profile = src.profile.copy()
        
        with rasterio.open(refined_output, 'w', **profile) as dst:
            dst.write(prediction_refined, 1)
        
        print(f"✓ Saved refined mask to {refined_output}")
        
        # Use refined mask for vectorization
        vector_input = refined_output
    else:
        vector_input = raster_output
    
    # Vectorization
    if not args.no_vectorize:
        print(f"\nVectorizing mask...")
        
        gdf = raster_to_vector_pipeline(
            raster_path=vector_input,
            output_shapefile=vector_output,
            simplify_tolerance=config['postprocessing']['simplify_tolerance'],
            min_area=config['postprocessing'].get('min_area_sqm', 25.0),
            buffer_distance=config['postprocessing'].get('buffer_distance', 0.0)
        )
        
        if len(gdf) > 0:
            print(f"✓ Detected {len(gdf)} change polygons")
        else:
            print(f"⚠ No significant changes detected")
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}\n")
    print(f"Outputs:")
    print(f"  Raster: {raster_output}")
    if not args.no_postprocess:
        print(f"  Refined: {refined_output}")
    if not args.no_vectorize:
        print(f"  Vector: {vector_output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
