"""
Master Pipeline Orchestration Script for Brain Tumor MRI Detection & Classification.

Runs the complete end-to-end pipeline:
Stage 1: Dataset Verification & Setup (Kaggle API)
Stage 2: Image Preprocessing (CLAHE, Median Blur, Resizing) & Stratified Splitting
Stage 3: Morphological Region of Interest (ROI) Segmentation
Stage 4: Data Augmentation & DataLoader Pipelines
Stage 5: Deep Feature Extraction (EfficientNetB0 Transfer Learning)
Stage 6: Genetic Algorithm (GA) Feature Selection
Stage 7: Machine Learning Classifier Benchmark (Softmax / SVM / RF / XGBoost)
Stage 8: Comprehensive Model Evaluation & Metrics Generation
Stage 9: Launch Streamlit Inference App
"""

import sys
import subprocess
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable

STAGES = [
    ("Stage 1 & 2: Data Preprocessing & Splitting", "src/preprocessing.py"),
    ("Stage 3: Tumor ROI Segmentation Demo", "src/segmentation.py"),
    ("Stage 4: Data Augmentation & DataLoader Verification", "src/data_loader.py"),
    ("Stage 5: Deep Feature Extraction (EfficientNetB0)", "src/feature_extractor.py"),
    ("Stage 6: Genetic Algorithm Feature Selection", "src/optimization/ga_selector.py"),
    ("Stage 7: Classifier Training & Hyperparameter Tuning", "src/classifier.py"),
    ("Stage 8: Model Evaluation & Metrics Benchmarking", "src/evaluate.py"),
]


def run_stage(title: str, script_path: str):
    print("\n" + "=" * 80)
    print(f"       EXECUTING: {title.upper()}")
    print("=" * 80)

    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        print(f"Error: Script file not found: {full_path}")
        sys.exit(1)

    result = subprocess.run([PYTHON_EXE, str(full_path)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{title}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    print("=" * 80)
    print("      BRAIN TUMOR MRI DETECTION & CLASSIFICATION - FULL PIPELINE")
    print("=" * 80)

    for title, script in STAGES:
        run_stage(title, script)

    print("\n" + "=" * 80)
    print("   ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nTo launch the interactive Streamlit inference web application, run:")
    print("    py -m streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    main()
