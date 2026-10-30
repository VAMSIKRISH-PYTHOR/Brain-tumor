# 🧠 Brain Tumor Detection & Multi-Class Classification Pipeline

An end-to-end medical computer vision and evolutionary machine learning framework for detecting and classifying brain tumors from Magnetic Resonance Imaging (MRI) scans into **4 distinct diagnostic classes**:
- `Glioma`
- `Meningioma`
- `Pituitary`
- `No Tumor`

Dataset Source: [Kaggle Masoud Nickparvar Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,200 total processed images).

---

## 📐 System Architecture & Pipeline Stages

```
   Raw MRI Input (7,200 scans)
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 2: Preprocessing & Stratified Splitting  │
   │ (CLAHE + 3x3 Median Blur + Min-Max Norm)       │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 3: Morphological ROI Tumor Segmentation  │
   │ (Otsu Thresholding + Morphological Erosion)    │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 4: PyTorch DataLoader & Augmentation    │
   │ (±15° Rotation, Zoom, Horizontal Flip)         │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 5: Deep Feature Extractor                │
   │ (EfficientNetB0 Fine-Tuning ──► 1,280-D GAP)   │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 6: Genetic Algorithm Feature Selection   │
   │ (Tournament Selection + Bit-Flip Mutation)    │
   │ 1,280 features  ──►  630 optimal features      │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 7: Multi-Classifier Hyperparameter Tuning│
   │ (Softmax, SVM-RBF, Random Forest, XGBoost)     │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 8: Evaluation & ROC-AUC Metrics          │
   └────────────────────────────────────────────────┘
              │
              ▼
   ┌────────────────────────────────────────────────┐
   │ Stage 9: Interactive Streamlit Web Application │
   └────────────────────────────────────────────────┘
```

---

## 📊 Evaluation & Classifier Benchmark Results

Evaluated on the independent test set (1,080 test MRI scans) across all metrics:

| Classifier | Accuracy | Precision (Macro) | Recall / Sensitivity (Macro) | Specificity (Macro) | F1-Score (Macro) | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF Kernel)** 🏆 | **92.59%** | **92.65%** | **92.59%** | **97.53%** | **92.60%** | **0.9802** |
| **Softmax (Logistic Reg)** | 88.89% | 89.01% | 88.89% | 96.30% | 88.92% | 0.9802 |
| **XGBoost** | 88.52% | 88.61% | 88.52% | 96.17% | 88.54% | 0.9797 |
| **Random Forest** | 88.15% | 88.16% | 88.15% | 96.05% | 88.10% | 0.9776 |

> **Champion Classifier**: **Support Vector Machine (RBF Kernel)** achieved the highest test accuracy (**92.59%**), highest specificity (**97.53%**), and an OvR ROC-AUC score of **0.9802** on 630 GA-selected features (a **50.78% feature dimension reduction**).

---

## 📁 Repository Structure

```
Brain tumor/
├── app/
│   └── streamlit_app.py        # Streamlit interactive Web Application
├── data/
│   ├── raw/                    # Downloaded raw Kaggle dataset
│   ├── processed/              # Preprocessed 224x224 CLAHE+Blur images
│   └── splits/                 # CSV train, val, test split metadata
├── models/
│   ├── efficientnet_b0_finetuned.pth  # Fine-tuned backbone weights
│   ├── ga_selected_features.npy       # 630-feature binary mask
│   ├── best_classifier.joblib         # Champion SVM model
│   └── all_classifiers.joblib         # All 4 tuned classifier heads
├── reports/
│   ├── metrics.json            # Machine-readable performance metrics
│   └── figures/                # Confusion matrices, ROC curves, GA plots
├── src/
│   ├── config.py               # Global hyperparameters & system paths
│   ├── dataset_setup.py        # Kaggle API downloader
│   ├── preprocessing.py        # CLAHE, Median blur, min-max norm
│   ├── segmentation.py         # Otsu thresholding & Lightweight U-Net
│   ├── data_loader.py          # PyTorch Datasets & DataLoaders
│   ├── feature_extractor.py    # EfficientNetB0 fine-tuning & feature export
│   ├── classifier.py           # GridSearchCV hyperparameter tuning
│   ├── evaluate.py             # Evaluation & ROC-AUC generation
│   └── optimization/
│       ├── fitness.py          # GA penalty-aware fitness evaluator
│       └── ga_selector.py      # Genetic Algorithm optimizer
├── run_all.py                  # Master orchestration script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Installation Guide

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed on your system. Install dependencies using:

```bash
py -m pip install -r requirements.txt
```

### 2. Download Dataset
The pipeline automatically downloads the Kaggle dataset via API token:
```bash
py src/dataset_setup.py
```

### 3. Run Entire End-to-End Pipeline
To run the full diagnostic pipeline from data preprocessing to evaluation:
```bash
py run_all.py
```

### 4. Launch Streamlit Web Application
To start the interactive web application for real-time MRI diagnosis:
```bash
py -m streamlit run app/streamlit_app.py
```

---

## 🧪 Figures & Artifacts Generated

1. **Augmentation Preview**: `reports/figures/augmentation_demo.png`
2. **Segmentation Preview**: `reports/figures/segmentation_demo.png`
3. **GA Optimization Curve**: `reports/figures/ga_fitness_curve.png`
4. **Confusion Matrices Grid**: `reports/figures/confusion_matrices.png`
5. **OvR ROC Curves Grid**: `reports/figures/roc_curves.png`
6. **Benchmark Comparison Bar Chart**: `reports/figures/metrics_comparison.png`
