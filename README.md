# Satellite Change Detection System

Production-ready deep learning system for detecting changes in bi-temporal satellite imagery using ResourceSat-2 (5m resolution) and Sentinel-2 (10m resolution) data.

## Project Overview

This system implements a **Siamese U-Net** architecture (default encoder: ResNet-18; auto-detected from your checkpoint) for semantic change detection in satellite imagery. It's designed to:

- **Train on GPU** (Kaggle/Google Colab) with mixed precision
- **Run inference on CPU** (16GB RAM, no GPU required)
- **Process large images** (18K×17K pixels) using memory-efficient tiling
- **Generate geospatial outputs** (GeoTIFF + Shapefiles with CRS preservation)

### Key Features

Memory-efficient windowed processing (never loads full 18K×17K images)  
CPU-optimized inference with tile-based prediction  
Combined Dice + Focal loss for handling class imbalance  
Morphological post-processing and vectorization  
Complete training pipeline with validation and checkpointing  
Mixed precision training (AMP) for faster GPU training  
CRS and geotransform preservation throughout pipeline  
Robust vector export using Fiona (NumPy 2.0 compatible)  
Unsupervised fallback (CVA + ΔNDVI + Otsu) when model predicts empty masks

---

## Project Structure

```
PS-10/
├── Dataset/                          # Your satellite data (ResourceSat-2, Sentinel-2)
│   ├── R2F18MAR2020.../             # ResourceSat-2 March 2020
│   │   └── BAND2.tif, BAND3.tif, BAND4.tif
│   └── R2F27JAN2025.../             # ResourceSat-2 January 2025
│       └── BAND2.tif, BAND3.tif, BAND4.tif
│
├── src/
│   ├── data/
│   │   ├── patch_generator.py       # Memory-efficient patch generation
│   │   ├── dataset.py               # PyTorch Dataset with augmentation
│   │   └── preprocessing.py         # Normalization and alignment
│   ├── models/
│   │   ├── siamese_unet.py          # Siamese U-Net (ResNet-18 default)
│   │   └── losses.py                # Dice + Focal loss
│   ├── training/
│   │   └── train.py                 # Complete training loop
│   ├── inference/
│   │   └── predict.py               # CPU-optimized tiled inference + unsupervised fallback
│   ├── postprocessing/
│   │   ├── refine_mask.py           # Morphological operations
│   │   └── vectorize.py             # Raster→Vector via Fiona (Shapefile/GeoJSON)
│   └── utils/
│       ├── metrics.py               # IoU, F1, precision, recall
│       └── visualization.py         # Plotting utilities
│
├── configs/
│   └── config.yaml                  # All hyperparameters
│
├── scripts/
│   ├── 1_generate_patches.py        # Run locally to prepare data
│   ├── 2_train_cloud.py             # Upload to Kaggle/Colab
│   ├── 3_inference_local.py         # Run locally on test images
│   └── 4_create_submission.py       # Generate final submission
│
├── notebooks/
│   └── kaggle_training.ipynb        # Ready-to-upload training notebook
│
├── requirements.txt                 # CPU dependencies (local)
├── requirements_cloud.txt           # GPU dependencies (Kaggle/Colab)
└── README.md                        # This file
```

---

## Quick Start

### 1. Installation (Local - Windows)

```powershell
# Clone/download project
cd D:\PS-10

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (CPU-only stack)
pip install -r requirements.txt

# If Torch fails on your platform, try the PyTorch index (CPU):
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Generate Training Patches (Local)

```powershell
# Process satellite image pairs into 256×256 patches
# Auto-detects ResourceSat-2 or Sentinel-2 from folder structure
python scripts\1_generate_patches.py `
    --data_dir Dataset `
    --output_dir data\patches `
    --config configs\config.yaml
```
Model Architecture
Siamese U-Net
Input: T1, T2 images
↓
Shared ResNet-18 Encoder
↓
Feature Fusion
↓
U-Net Decoder with skip connections
↓
Binary change mask

**Supported structures:**
- ResourceSat-2: `Dataset/R2F.../R2F.../BAND*.tif`
- Sentinel-2: `Dataset/*.SAFE/`

**Output:**
- `data/patches/train/` - Training patches (80%)
- `data/patches/val/` - Validation patches (20%)
- Each patch: 256×256 pixels, 3 bands (Green, Red, NIR)

**Memory usage:** <4GB RAM (processes one tile at a time)

### 3. Train Model (Kaggle/Colab)

#### Option A: Upload to Kaggle

1. Zip your patches:
   ```powershell
   Compress-Archive -Path data\patches -DestinationPath patches.zip
   ```

2. Upload `patches.zip` to Kaggle as a dataset

3. Create new Kaggle notebook and install dependencies:
   ```python
   !pip install segmentation-models-pytorch timm albumentations
   ```

4. Run training:
   ```python
   !python scripts/2_train_cloud.py \
       --patches_dir /kaggle/input/patches \
       --output_dir /kaggle/working \
       --epochs 50 \
       --batch_size 16
   ```

5. Download `best_model.pth` from output

#### Option B: Google Colab

1. Upload to Google Drive
2. Mount drive in Colab
3. Run training script

**Training time:** ~2-3 hours on Kaggle T4 GPU (50 epochs)

### 4. Run Inference (Local - CPU)

#### ResourceSat-2 Data

```powershell
# Predict changes between two time points
python scripts\3_inference_local.py `
    --model models\best_model.pth `
    --image1_dir Dataset\R2F18MAR2020046249009500049SSANSTUC00GTDA\R2F18MAR2020046249009500049SSANSTUC00GTDA `
    --image2_dir Dataset\R2F27JAN2025071483009500049SSANSTUC00GTDA\R2F27JAN2025071483009500049SSANSTUC00GTDA `
    --output_dir outputs\predictions
```

#### Sentinel-2 Data

```powershell
# Predict changes from Sentinel-2 .SAFE products
python scripts\3_inference_local.py `
    --model models\best_model.pth `
    --image1_dir Dataset\S2A_MSIL2A_20200328T053641_N0500_R005_T43RCM_20230601T154807.SAFE `
    --image2_dir Dataset\S2B_MSIL2A_20250307T053649_N0511_R005_T43RCM_20250307T082113.SAFE `
    --output_dir outputs\predictions
```

**Auto-detection:** The script automatically detects whether you're using ResourceSat-2 or Sentinel-2 data based on folder structure.

**Output:**
- `Change_Mask_*.tif` - Binary change mask (GeoTIFF)
- `Change_Mask_*_refined.tif` - Post-processed mask
- `Change_Mask_*.shp` - Vector polygons (+ .shx, .dbf, .prj)
- `Change_Mask_*.geojson` - Optional GeoJSON (see below)

**Memory usage:** <16GB RAM (tile-based processing)  
**Speed:** ~10-15 minutes for 18K×17K image on Intel i7-13620H (CPU)

#### Export GeoJSON alongside Shapefile

```powershell
python -c "import sys; from pathlib import Path; sys.path.append('src'); from postprocessing.vectorize import raster_to_vector_pipeline; p=Path('results/Change_Mask_18MAR202004_27JAN202507_refined.tif'); raster_to_vector_pipeline(p, Path('results/Change_Mask_18MAR202004_27JAN202507.shp'), output_geojson=Path('results/Change_Mask_18MAR202004_27JAN202507.geojson'))"
```

#### Zip outputs (rasters + vectors)

```powershell
if (Test-Path "results\Change_Mask_18MAR202004_27JAN202507_package.zip") { Remove-Item -Force "results\Change_Mask_18MAR202004_27JAN202507_package.zip" }
Compress-Archive -Path "results\Change_Mask_18MAR202004_27JAN202507.tif","results\Change_Mask_18MAR202004_27JAN202507_refined.tif","results\Change_Mask_18MAR202004_27JAN202507.shp","results\Change_Mask_18MAR202004_27JAN202507.shx","results\Change_Mask_18MAR202004_27JAN202507.dbf","results\Change_Mask_18MAR202004_27JAN202507.prj","results\Change_Mask_18MAR202004_27JAN202507.cpg","results\Change_Mask_18MAR202004_27JAN202507.geojson" -DestinationPath "results\Change_Mask_18MAR202004_27JAN202507_package.zip"
```

### 5. Create Submission

```powershell
python scripts\4_create_submission.py `
    --predictions_dir outputs\predictions `
    --team_name YourTeamName
```

**Output:** `PS10_14-NOV-2024_YourTeamName.zip`

---

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

### Data Settings
```yaml
data:
  patch_size: 256          # Patch dimensions
  overlap: 64              # Overlap for inference
  train_split: 0.8         # 80% training, 20% validation
  bands: [0, 1, 2]         # Green, Red, NIR
```

### Model Settings
```yaml
model:
  architecture: resnet18            # Default; auto-detected from checkpoint when possible
  encoder_weights: imagenet
  in_channels: 3           # RGB or multispectral
  classes: 1               # Binary change detection
```

### Training Settings
```yaml
training:
  batch_size: 16           # For GPU (reduce if OOM)
  epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.00001
  optimizer: adamw
  scheduler: cosine
  early_stopping_patience: 15
  mixed_precision: true    # Enable AMP
```

### Inference Settings
```yaml
inference:
  device: cpu              # Use 'cuda' if GPU available
  threshold: 0.5           # Binary threshold
  tile_size: 256
  tile_overlap: 64
  overlap_mode: average    # Blend overlapping tiles
```

### Post-processing
```yaml
postprocessing:
  morphology_kernel_size: 3
  min_component_size: 100  # Remove objects <100 pixels
  simplify_tolerance: 0.5  # Simplify polygons (meters)
  min_area_sqm: 25.0       # Minimum polygon area
```

---

## 📊 Model Architecture

### Siamese U-Net
```
Input: Two temporal images (T1, T2)
├── Shared Encoder: ResNet-18 (ImageNet pretrained by default)
│   ├── Branch 1: Extract features from T1
│   └── Branch 2: Extract features from T2 (shared weights)
├── Feature Fusion: Concatenate T1 and T2 features
└── U-Net Decoder: Reconstruct change mask
    ├── Skip connections from encoder
    ├── Decoder channels: [256, 128, 64, 32, 16]
    └── Output: Single-channel binary mask
```

**Parameters:** ~14M (fits in 16GB RAM for inference)

### Loss Function
```
Combined Loss = 0.5 × Dice Loss + 0.5 × Focal Loss

- Dice Loss: Optimizes region overlap (good for segmentation)
- Focal Loss: Handles class imbalance (focuses on hard examples)
```

---

## 🔬 Dataset Information

### ResourceSat-2 (LISS-4 MX Sensor)
- **Resolution:** 5.8m (resampled to 5m)
- **Bands:** 3 (BAND2=Green, BAND3=Red, BAND4=NIR)
- **Format:** Separate GeoTIFF per band
- **Structure:** `Dataset/R2FDDMMMYYYYHHMMSS.../R2FDDMMMYYYYHHMMSS.../BAND*.tif`
- **Size:** 18,000 × 17,000 pixels (~620MB per band)
- **CRS:** UTM Zone 43N, WGS84

### Sentinel-2 (MSI Sensor)
- **Resolution:** 10m (bands B02, B03, B04, B08)
- **Bands:** B03=Green, B04=Red, B08=NIR (used for change detection)
- **Format:** JPEG2000 (.jp2) in .SAFE structure
- **Structure:** `Dataset/*.SAFE/GRANULE/*/IMG_DATA/`
- **Products:** L1C (Top-of-Atmosphere) and L2A (Bottom-of-Atmosphere)
- **Size:** ~10,000 × 10,000 pixels per tile
- **CRS:** UTM (zone-dependent), WGS84

**Note:** The system auto-detects which data source you're using based on folder structure.

### Preprocessing Pipeline
1. **Windowed Reading:** Load 256×256 tiles (never full image)
2. **Normalization:** Percentile-based (2nd-98th percentile)
3. **Augmentation:** Flips, rotations, brightness/contrast
4. **Filtering:** Remove patches with >80% no-data

---

## Expected Performance

### Training Metrics (50 epochs)
- **Best Validation IoU:** 0.65-0.75 (depends on data quality)
- **Best Validation F1:** 0.70-0.80
- **Training Time:** 2-3 hours (Kaggle T4 GPU)

### Inference Performance
- **Speed:** ~10-15 minutes for 18K×17K image (CPU)
- **Memory:** <16GB RAM (tile-based processing)
- **Output:** Binary mask + shapefile with preserved CRS

---

## Troubleshooting

### Issue: Out of Memory during training
**Solution:** Reduce `batch_size` in config.yaml (try 8 or 4)

### Issue: Patches generation runs out of memory
**Solution:** Already optimized with windowed reading. Should use <4GB RAM.

### Issue: Model training is slow
**Solution:**
- Enable mixed precision: `mixed_precision: true`
- Increase `num_workers` (but not too high on CPU)
- Use GPU (Kaggle/Colab)

### Issue: No changes detected in output
**Solution:**
- The pipeline includes an unsupervised fallback (CVA + ΔNDVI + Otsu) that activates automatically when the model output is empty or nearly empty.
- You can tune `postprocessing` thresholds and morphology to refine results.

### Issue: Shapefile has wrong coordinates
**Solution:** CRS is automatically preserved from input GeoTIFF. Verify input files have correct CRS.

### Issue: Crash when exporting shapefile on NumPy 2.0
**Symptoms:** ValueError from GeoPandas when calling `to_file` or `.apply()` on a GeoSeries.

**Solution:** The vector export uses Fiona directly and converts Polygon→MultiPolygon safely without `.apply()`, avoiding NumPy 2.0 internals. This is implemented in `src/postprocessing/vectorize.py` and requires no user action.

### Issue: PyTorch installation fails
**Solution:**
```powershell
# Install specific version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

---

## Technical Details

### Memory Management
- **Patch Generation:** Reads 256×256 windows, processes immediately, never accumulates
- **Training:** Uses PyTorch DataLoader with `num_workers=2`
- **Inference:** Tile-based with overlap averaging, writes progressively to disk

### Coordinate Reference Systems
- **Input:** UTM Zone 43N (EPSG:32643)
- **Processing:** Preserves CRS throughout pipeline
- **Output:** Same CRS as input (GeoTIFF + .prj for shapefile)

### File Formats
- **Input:** GeoTIFF (.tif)
- **Output:**
  - Raster: GeoTIFF with LZW compression
  - Vector: ESRI Shapefile (.shp + .shx + .dbf + .prj)

---

## Citation

If you use this code, please cite:

```bibtex
@software{satellite_change_detection,
  title={Satellite Change Detection System},
  author={Team AI},
  year={2024},
  url={https://github.com/Manvitha3004/PS-10}
}
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## Contact

For questions or issues:
- Open a GitHub issue
- Email: manvithareddy3004@gmail.com

---

## Version History

- **v1.0.0** (November 2024) - Initial release
  - Siamese U-Net architecture
  - CPU-optimized inference
  - ResourceSat-2 support
  - Memory-efficient processing

---

**Happy Change Detecting! 🛰️🔍**
