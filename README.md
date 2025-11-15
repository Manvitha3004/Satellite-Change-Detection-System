# Satellite Change Detection System

Production-ready deep learning system for detecting changes in bi-temporal satellite imagery using ResourceSat-2 (5m resolution) and Sentinel-2 (10m resolution) data.

## 🎯 Project Overview

This system implements a **Siamese U-Net** architecture with EfficientNet-B0 backbone for semantic change detection in satellite imagery. It's designed to:

- **Train on GPU** (Kaggle/Google Colab) with mixed precision
- **Run inference on CPU** (16GB RAM, no GPU required)
- **Process large images** (18K×17K pixels) using memory-efficient tiling
- **Generate geospatial outputs** (GeoTIFF + Shapefiles with CRS preservation)

### Key Features

✅ Memory-efficient windowed processing (never loads full 18K×17K images)  
✅ CPU-optimized inference with tile-based prediction  
✅ Combined Dice + Focal loss for handling class imbalance  
✅ Morphological post-processing and vectorization  
✅ Complete training pipeline with validation and checkpointing  
✅ Mixed precision training (AMP) for faster GPU training  
✅ CRS and geotransform preservation throughout pipeline  

---

## 📁 Project Structure

```
satellite-change-detection/
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
│   │   ├── siamese_unet.py          # EfficientNet-B0 + U-Net architecture
│   │   └── losses.py                # Dice + Focal loss
│   ├── training/
│   │   └── train.py                 # Complete training loop
│   ├── inference/
│   │   └── predict.py               # CPU-optimized tile-based inference
│   ├── postprocessing/
│   │   ├── refine_mask.py           # Morphological operations
│   │   └── vectorize.py             # Raster to shapefile conversion
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

## 🚀 Quick Start

### 1. Installation (Local - Windows)

```powershell
# Clone/download project
cd D:\PS-10

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Patches (Local)

```powershell
# Process ResourceSat-2 image pairs into 256×256 patches
python scripts\1_generate_patches.py `
    --data_dir Dataset `
    --output_dir data\patches `
    --config configs\config.yaml
```

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

```powershell
# Predict changes between two time points
python scripts\3_inference_local.py `
    --model models\best_model.pth `
    --image1_dir Dataset\R2F18MAR2020046249009500049SSANSTUC00GTDA\R2F18MAR2020046249009500049SSANSTUC00GTDA `
    --image2_dir Dataset\R2F27JAN2025071483009500049SSANSTUC00GTDA\R2F27JAN2025071483009500049SSANSTUC00GTDA `
    --output_dir outputs\predictions
```

**Output:**
- `Change_Mask_*.tif` - Binary change mask (GeoTIFF)
- `Change_Mask_*_refined.tif` - Post-processed mask
- `Change_Mask_*.shp` - Vector polygons (+ .shx, .dbf, .prj)

**Memory usage:** <16GB RAM (tile-based processing)  
**Speed:** ~5-10 minutes for 18K×17K image on Intel i7-13620H

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
  architecture: efficientnet-b0
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
├── Shared Encoder: EfficientNet-B0 (ImageNet pretrained)
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
- **Bands:** 3 (Green, Red, NIR)
- **Format:** Separate GeoTIFF per band (BAND2.tif, BAND3.tif, BAND4.tif)
- **Size:** 18,000 × 17,000 pixels (~620MB per band)
- **CRS:** UTM Zone 43N, WGS84
- **Dates:** March 2020, January 2025 (~5 years apart)

### Preprocessing Pipeline
1. **Windowed Reading:** Load 256×256 tiles (never full image)
2. **Normalization:** Percentile-based (2nd-98th percentile)
3. **Augmentation:** Flips, rotations, brightness/contrast
4. **Filtering:** Remove patches with >80% no-data

---

## 📈 Expected Performance

### Training Metrics (50 epochs)
- **Best Validation IoU:** 0.65-0.75 (depends on data quality)
- **Best Validation F1:** 0.70-0.80
- **Training Time:** 2-3 hours (Kaggle T4 GPU)

### Inference Performance
- **Speed:** ~5-10 minutes for 18K×17K image (CPU)
- **Memory:** <16GB RAM (tile-based processing)
- **Output:** Binary mask + shapefile with preserved CRS

---

## 🛠️ Troubleshooting

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
- Check `threshold` in config (try 0.3 or 0.7)
- Verify images are from different time points
- Reduce `min_component_size` in post-processing

### Issue: Shapefile has wrong coordinates
**Solution:** CRS is automatically preserved from input GeoTIFF. Verify input files have correct CRS.

### Issue: PyTorch installation fails
**Solution:**
```powershell
# Install specific version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

---

## 📚 Technical Details

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

## 🎓 Citation

If you use this code, please cite:

```bibtex
@software{satellite_change_detection,
  title={Satellite Change Detection System},
  author={Team AI},
  year={2024},
  url={https://github.com/yourusername/satellite-change-detection}
}
```

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## 📧 Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com

---

## 🔄 Version History

- **v1.0.0** (November 2024) - Initial release
  - Siamese U-Net architecture
  - CPU-optimized inference
  - ResourceSat-2 support
  - Memory-efficient processing

---

**Happy Change Detecting! 🛰️🔍**
