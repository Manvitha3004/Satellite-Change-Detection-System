from pathlib import Path
import shutil
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Get all patches
patches_dir = Path('data/patches/raw/pair_18MAR202004_27JAN202507')
patches = sorted(list(patches_dir.glob('*.npz')))

print(f"Found {len(patches)} patches")

# Shuffle and split
indices = np.random.permutation(len(patches))
split_idx = int(len(patches) * 0.8)

train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

# Create output directories
train_dir = Path('data/patches/train')
val_dir = Path('data/patches/val')
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# Copy files
print("Copying training patches...")
for i in train_indices:
    shutil.copy(patches[i], train_dir / patches[i].name)

print("Copying validation patches...")
for i in val_indices:
    shutil.copy(patches[i], val_dir / patches[i].name)

print(f"\n✓ Split complete:")
print(f"  Training patches: {len(list(train_dir.glob('*.npz')))}")
print(f"  Validation patches: {len(list(val_dir.glob('*.npz')))}")
