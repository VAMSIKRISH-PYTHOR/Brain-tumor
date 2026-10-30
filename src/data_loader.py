"""
PyTorch Dataset and Data Loader Module with Data Augmentation Pipeline.

Handles:
1. Loading preprocessed MRI images from CSV index splits.
2. Training-only data augmentations (Rotation ±15°, Horizontal Flip, Zoom, Brightness/Contrast Jitter).
3. Unaugmented transformations for Validation and Testing splits.
4. Optional ROI Segmentation cropping pipeline integration.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    PROJECT_ROOT,
    DATA_SPLITS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    CLASSES,
    CLASS_TO_IDX,
    FIGURES_DIR,
    ENABLE_SEGMENTATION_ROI
)
from src.segmentation import apply_segmentation_pipeline


def get_transforms(split: str):
    """
    Returns image transformation pipelines for PyTorch.
    Augmentations are strictly restricted to 'train' split.
    """
    if split == "train":
        transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=15),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.9, 1.1), ratio=(0.95, 1.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet standard normalization
        ])
    else:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for Brain Tumor MRI Scans.
    Reads image paths and class labels from split CSV files.
    """
    def __init__(self, csv_file: str, split: str = "train", apply_roi: bool = ENABLE_SEGMENTATION_ROI):
        self.csv_path = Path(csv_file)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Split CSV file not found: {csv_file}")
            
        self.df = pd.read_csv(self.csv_path)
        self.split = split
        self.apply_roi = apply_roi
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["filepath"]
        full_path = PROJECT_ROOT / rel_path

        # Read image in RGB
        img_bgr = cv2.imread(str(full_path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {full_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply ROI Segmentation if enabled
        if self.apply_roi:
            img_rgb = apply_segmentation_pipeline(img_rgb, enable_roi=True)

        # Apply torchvision transforms
        tensor_img = self.transform(img_rgb)
        label = int(row["label"])

        return tensor_img, label, str(rel_path)


def get_dataloaders(batch_size: int = BATCH_SIZE, num_workers: int = 0, apply_roi: bool = ENABLE_SEGMENTATION_ROI):
    """
    Creates and returns PyTorch DataLoaders for train, val, and test splits.
    """
    train_csv = DATA_SPLITS_DIR / "train_split.csv"
    val_csv = DATA_SPLITS_DIR / "val_split.csv"
    test_csv = DATA_SPLITS_DIR / "test_split.csv"

    train_ds = BrainTumorDataset(train_csv, split="train", apply_roi=apply_roi)
    val_ds = BrainTumorDataset(val_csv, split="val", apply_roi=apply_roi)
    test_ds = BrainTumorDataset(test_csv, split="test", apply_roi=apply_roi)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def main():
    print("Testing Stage 4: Data Augmentation & DataLoader...")

    train_csv = DATA_SPLITS_DIR / "train_split.csv"
    if not train_csv.exists():
        print(f"[ERROR] Train split CSV missing: {train_csv}")
        return

    train_ds = BrainTumorDataset(train_csv, split="train", apply_roi=False)
    print(f"Loaded Train Dataset: {len(train_ds)} samples.")

    # Visualize augmentations on a single sample
    img_bgr = cv2.imread(str(PROJECT_ROOT / train_ds.df.iloc[0]["filepath"]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    aug_transform = get_transforms("train")
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original Preprocessed MRI")
    axes[0, 0].axis("off")

    for i in range(1, 8):
        row_idx = i // 4
        col_idx = i % 4
        # Apply transformation
        aug_tensor = aug_transform(img_rgb)
        # Denormalize tensor for display
        aug_np = aug_tensor.numpy().transpose(1, 2, 0)
        aug_np = aug_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        aug_np = np.clip(aug_np, 0, 1)

        axes[row_idx, col_idx].imshow(aug_np)
        axes[row_idx, col_idx].set_title(f"Augmented Variant {i}")
        axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    out_fig = FIGURES_DIR / "augmentation_demo.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()

    print(f"Augmentation demonstration figure saved to {out_fig}")

    # Verify DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    for images, labels, paths in train_loader:
        print(f"Train Batch - Images Tensor: {images.shape}, Labels Tensor: {labels.shape}")
        break

    print("Stage 4 completed successfully!")


if __name__ == "__main__":
    main()
