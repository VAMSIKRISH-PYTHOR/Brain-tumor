"""
Preprocessing and Stratified Dataset Splitting Module.

Performs:
1. Image resizing to 224x224 pixels.
2. Median filter denoising (3x3 kernel).
3. Contrast Limited Adaptive Histogram Equalization (CLAHE) enhancement.
4. Min-max normalization [0, 1].
5. Saving processed images into data/processed/ preserving class folders.
6. Generating stratified Train (70%) / Val (15%) / Test (15%) splits saved as CSV index files in data/splits/.
"""

import os
import sys
import glob
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SPLITS_DIR,
    CLASSES,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MEDIAN_BLUR_KERNEL,
    SPLIT_RATIOS,
    RANDOM_SEED,
)


def preprocess_single_image(img_input, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Applies the full preprocessing pipeline to a single image (file path or BGR/RGB numpy array):
    - Reads image in grayscale (or converts in-memory array to grayscale).
    - Resizes to target_size (224x224).
    - Applies median filter denoising.
    - Applies CLAHE contrast enhancement.
    - Normalizes to [0, 1] float32 array.
    - Converts to 3-channel format for CNN backbone compatibility.
    """
    if isinstance(img_input, (str, Path)):
        img = cv2.imread(str(img_input), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image from path: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY) if len(img_input.shape) == 3 else img_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(img_input)}")

    # Resize image
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # Apply Median Filter Denoising
    img_denoised = cv2.medianBlur(img_resized, MEDIAN_BLUR_KERNEL)

    # Apply CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    img_clahe = clahe.apply(img_denoised)

    # Min-Max Normalization to [0, 1] float32
    img_norm = img_clahe.astype(np.float32) / 255.0

    # Stack to 3-channel BGR format
    img_3channel = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return img_3channel


def process_and_save_all_images():
    """
    Processes all raw images across Training and Testing raw folders,
    saves preprocessed images to data/processed/<class>/, and returns a DataFrame of records.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    print("Starting Image Preprocessing Pipeline (Resize, Denoise, CLAHE, Normalization)...")

    # Collect all image files from data/raw/
    raw_files = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    for root, _, files in os.walk(DATA_RAW_DIR):
        for f in files:
            if Path(f).suffix.lower() in valid_exts:
                full_path = Path(root) / f
                # Determine class name from parent directory name
                parent_name = full_path.parent.name.lower()
                matched_class = None
                for cls in CLASSES:
                    if cls.lower() == parent_name:
                        matched_class = cls
                        break
                if matched_class:
                    raw_files.append((full_path, matched_class))

    print(f"Found {len(raw_files)} raw images across all classes.")

    for raw_path, cls_name in tqdm(raw_files, desc="Processing Images"):
        # Create output directory for class in data/processed/
        cls_out_dir = DATA_PROCESSED_DIR / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        # Unique filename using parent split name and original stem
        split_prefix = raw_path.parent.parent.name.lower()
        out_filename = f"{split_prefix}_{raw_path.stem}.png"
        out_path = cls_out_dir / out_filename

        try:
            processed_img = preprocess_single_image(str(raw_path))
            # Save as PNG image
            cv2.imwrite(str(out_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
            
            # Store relative path for split CSVs
            rel_path = str(out_path.relative_to(Path(__file__).resolve().parent.parent))
            records.append({
                "filepath": rel_path,
                "class_name": cls_name,
                "label": CLASS_TO_IDX[cls_name]
            })
        except Exception as e:
            print(f"Error processing {raw_path}: {e}")

    df_dataset = pd.DataFrame(records)
    print(f"Successfully processed and saved {len(df_dataset)} images into {DATA_PROCESSED_DIR}")
    return df_dataset


def create_stratified_splits(df: pd.DataFrame):
    """
    Creates stratified Train (70%), Validation (15%), and Test (15%) splits
    and saves train_split.csv, val_split.csv, test_split.csv in data/splits/.
    """
    DATA_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Stratified Train (70%) / Val (15%) / Test (15%) Splits...")

    # First split: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]),
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    # Second split: Split 30% Temp equally into 15% Val and 15% Test
    val_ratio_in_temp = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    # Save to CSV
    train_path = DATA_SPLITS_DIR / "train_split.csv"
    val_path = DATA_SPLITS_DIR / "val_split.csv"
    test_path = DATA_SPLITS_DIR / "test_split.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {len(train_df)} training records to {train_path}")
    print(f"Saved {len(val_df)} validation records to {val_path}")
    print(f"Saved {len(test_df)} test records to {test_path}")

    # Summary table per split and class
    summary_rows = []
    for name, split_df in [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]:
        counts = split_df["class_name"].value_counts().to_dict()
        row = {"Split": name}
        for cls in CLASSES:
            row[cls] = counts.get(cls, 0)
        row["Total"] = len(split_df)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 60)
    print("                STRATIFIED SPLIT SUMMARY TABLE")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60 + "\n")


def main():
    print("Starting Stage 2: Image Preprocessing & Stratified Splitting...")
    df_dataset = process_and_save_all_images()
    if not df_dataset.empty:
        create_stratified_splits(df_dataset)
        print("Stage 2 completed successfully!")
    else:
        print("[ERROR] Preprocessing failed: No images found.")


if __name__ == "__main__":
    main()
