"""
Global Configuration Settings for Brain Tumor Detection Framework.
Defines file system paths, hyperparameters, preprocessing constants, and reproducibility seeds.
"""

import os
from pathlib import Path

# Project Base Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SPLITS_DIR = DATA_DIR / "splits"

# Models and Reports Directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
APP_DIR = PROJECT_ROOT / "app"

# Ensure runtime directories exist
for path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SPLITS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, APP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset & Taxonomy Configuration
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Image Dimensions & Normalization
IMAGE_SIZE = (224, 224)  # (Height, Width) for EfficientNetB0 backbone
IMAGE_CHANNELS = 3

# Preprocessing Settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
MEDIAN_BLUR_KERNEL = 3
ENABLE_SEGMENTATION_ROI = False  # Toggleable Otsu ROI segmentation cropping

# Train/Val/Test Split Ratios
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Deep Learning / Feature Extractor Settings
BACKBONE_NAME = "EfficientNetB0"
FEATURE_DIM = 1280
FINE_TUNE_LAYERS = 20
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Genetic Algorithm Feature Selection Settings
GA_POP_SIZE = 30
GA_GENERATIONS = 40
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.05
GA_ALPHA_PENALTY = 0.01  # Fitness penalty per selected feature count ratio
GA_TOURNAMENT_SIZE = 3

# Kaggle API Credentials Setup
# Set your Kaggle API token as an environment variable or place kaggle.json in ~/.kaggle/
DEFAULT_KAGGLE_TOKEN = os.environ.get("KAGGLE_TOKEN", "")
