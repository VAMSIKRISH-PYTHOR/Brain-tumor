"""
Dataset Setup Module for Brain Tumor Detection Framework.

Downloads the Kaggle dataset 'masoudnickparvar/brain-tumor-mri-dataset'
via the Kaggle API into data/raw/, verifies image counts per class,
and prints a formatted summary table. Handles missing kaggle.json/access_token
credentials with clear, user-friendly error messages.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
import pandas as pd

# Add parent directory to path to allow importing src modules directly
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW_DIR, KAGGLE_DATASET, CLASSES, DEFAULT_KAGGLE_TOKEN


def setup_kaggle_credentials():
    """
    Ensures Kaggle API credentials exist.
    Checks environment variables KAGGLE_API_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY,
    or credential files in ~/.kaggle/. If missing, sets up KAGGLE_API_TOKEN using
    the provided token or default token.
    """
    home = Path.home()
    kaggle_dir = home / ".kaggle"
    access_token_file = kaggle_dir / "access_token"
    kaggle_json_file = kaggle_dir / "kaggle.json"

    # Check if env vars or files exist
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("Using KAGGLE_API_TOKEN from environment variables.")
        return True
    
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("Using KAGGLE_USERNAME and KAGGLE_KEY from environment variables.")
        return True

    if access_token_file.exists() or kaggle_json_file.exists():
        print(f"Found existing Kaggle credentials in {kaggle_dir}")
        return True

    # Setup token automatically if not present
    token_to_use = os.environ.get("KAGGLE_API_TOKEN", DEFAULT_KAGGLE_TOKEN)
    if token_to_use:
        print(f"Setting up Kaggle API Token in environment: {token_to_use[:8]}...")
        os.environ["KAGGLE_API_TOKEN"] = token_to_use
        
        # Also persist to ~/.kaggle/access_token for Kaggle CLI compatibility
        try:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            with open(access_token_file, "w") as f:
                f.write(token_to_use)
            print(f"Saved token to {access_token_file}")
            return True
        except Exception as e:
            print(f"Warning: Could not write to {access_token_file}: {e}")
            return True

    print("\n" + "=" * 60)
    print("[ERROR] Kaggle API credentials not found!")
    print("=" * 60)
    print("Please set your Kaggle API token using one of the following methods:")
    print("  1. Set environment variable: $env:KAGGLE_API_TOKEN='KGAT_...'")
    print("  2. Place 'kaggle.json' or 'access_token' inside ~/.kaggle/")
    print("=" * 60 + "\n")
    return False


def download_dataset():
    """
    Downloads and extracts the Kaggle dataset into DATA_RAW_DIR.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    training_dir = DATA_RAW_DIR / "Training"
    testing_dir = DATA_RAW_DIR / "Testing"

    if training_dir.exists() and testing_dir.exists():
        print(f"Dataset already present in {DATA_RAW_DIR}. Skipping download.")
        return True

    if not setup_kaggle_credentials():
        return False

    print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
    
    # Try using kaggle CLI / python subprocess
    zip_path = DATA_RAW_DIR / "brain-tumor-mri-dataset.zip"
    
    try:
        # Try running kaggle CLI
        cmd = [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", KAGGLE_DATASET,
            "-p", str(DATA_RAW_DIR)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Kaggle CLI download output: {result.stdout}\n{result.stderr}")
            # Try alternate zip filename if API fetched dataset directly as zip
    except Exception as e:
        print(f"Kaggle download exception: {e}")

    # Search for downloaded zip files in DATA_RAW_DIR
    zip_files = list(DATA_RAW_DIR.glob("*.zip"))
    if not zip_files:
        # Also check if raw folder directly contains downloaded folders
        if (DATA_RAW_DIR / "Training").exists():
            print("Extracted dataset folders already found.")
            return True
        print(f"[ERROR] Could not find downloaded zip file in {DATA_RAW_DIR}")
        return False

    for zf in zip_files:
        print(f"Unzipping {zf.name} to {DATA_RAW_DIR}...")
        with zipfile.ZipFile(zf, "r") as zip_ref:
            zip_ref.extractall(DATA_RAW_DIR)
        print(f"Extracted successfully. Removing {zf.name}...")
        try:
            os.remove(zf)
        except Exception:
            pass

    return True


def verify_and_summarize_dataset():
    """
    Inspects image counts per class across Training and Testing directories
    and displays a formatted summary table.
    """
    summary_data = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for split in ["Training", "Testing"]:
        split_dir = DATA_RAW_DIR / split
        if not split_dir.exists():
            print(f"[WARNING] Directory not found: {split_dir}")
            continue

        split_total = 0
        split_counts = {}
        for cls_name in CLASSES:
            # Match folder name case-insensitively or exact match
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                # try finding matching folder name in case of capitalization difference
                matches = [d for d in split_dir.iterdir() if d.is_dir() and d.name.lower() == cls_name.lower()]
                cls_dir = matches[0] if matches else cls_dir

            if cls_dir.exists():
                images = [f for f in cls_dir.glob("*") if f.suffix.lower() in valid_extensions]
                count = len(images)
            else:
                count = 0
            
            split_counts[cls_name] = count
            split_total += count

        split_counts["Total"] = split_total
        summary_data.append({"Split": split, **split_counts})

    if not summary_data:
        print("[ERROR] No dataset splits found for verification.")
        return False

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 60)
    print("                DATASET INGESTION SUMMARY TABLE")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("=" * 60 + "\n")
    return True


def main():
    print("Starting Stage 1: Dataset Setup & Verification...")
    success = download_dataset()
    if success:
        verify_and_summarize_dataset()
        print("Stage 1 completed successfully!")
    else:
        print("Stage 1 failed. Please check credentials or dataset download.")


if __name__ == "__main__":
    main()
