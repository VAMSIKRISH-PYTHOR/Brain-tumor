# Brain Tumor Detection and Classification through Optimised Computational Intelligence using MRI Scans

## 1. Project Title
**Brain Tumor Detection and Classification through Optimised Computational Intelligence using MRI Scans**

---

## 2. Problem Statement
Brain tumors represent one of the most critical and challenging central nervous system malignancies. Accurate, non-invasive classification of magnetic resonance imaging (MRI) brain scans into specific tumor types (Glioma, Meningioma, Pituitary Tumor) or non-tumorous tissue (No Tumor) is vital for early treatment planning. This project provides a robust, reproducible computational intelligence framework combining deep convolutional feature extraction, Genetic Algorithm (GA) feature selection, and Support Vector Machine (SVM) classification.

---

## 3. System Architecture & Pipeline
The application follows a modular 6-stage machine learning pipeline:

```
Input Raw MRI Scan
       │
       ▼
1. Image Preprocessing (Auto Skull-Crop, CLAHE Contrast, Median Blur)
       │
       ▼
2. Deep Feature Extraction (EfficientNetB0 → 1,280 features)
       │
       ▼
3. Evolutionary Feature Selection (Genetic Algorithm → 630 selected features)
       │
       ▼
4. Classifier Head (SVM with RBF Kernel)
       │
       ▼
5. 4-Class Probability Prediction & Confidence Distribution
```

---

## 4. Target Classes
The system classifies brain MRI scans into exactly four mutually exclusive diagnostic categories:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor** (Normal brain tissue)

---

## 5. Image Preprocessing
- **Skull Auto-Cropping**: Isolates intracranial brain tissue by stripping DICOM text headers, footers, and background padding.
- **Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) with a $2\times2$ grid size and clip limit of $2.0$.
- **Noise Reduction**: Median Blur filtering ($3\times3$ kernel) to eliminate high-frequency acquisition artifacts.
- **Normalization**: Min-Max scaling to $[0, 1]$ followed by resizing to $(224 \times 224 \times 3)$.

---

## 6. EfficientNetB0 Deep Feature Extraction
- **Model**: Fine-tuned EfficientNetB0 deep convolutional neural network.
- **Output Vector**: 1,280-dimensional global average pooling feature vector per MRI scan.

---

## 7. Genetic Algorithm (GA) Feature Selection
- **Methodology**: Evolutionary search using binary-coded Genetic Algorithm.
- **Dimensionality Reduction**: Filters raw features from **1,280** down to **630** optimal features (~50.8% feature reduction).
- **Impact**: Removes redundant/noisy deep features, improves classification speed, and prevents overfitting.

---

## 8. SVM Classifier Head
- **Algorithm**: Support Vector Machine with Radial Basis Function (RBF) kernel ($C = 10.0$, $\gamma = \text{'scale'}$).
- **Probability Calibration**: Uses Platt scaling (`predict_proba`) to generate normalized confidence distributions across all four classes.
- **Single Source of Truth**: Displayed prediction strictly corresponds to $\text{argmax}(P_{\text{class}})$.

---

## 9. Test Dataset & Ground-Truth Verification
- **Total Test Dataset Size**: **1,080** MRI scans (`models/X_test.npy` & `models/y_test.npy`).
- **Evaluation Protocol**: Evaluated strictly against verified ground-truth labels. No labels or predictions are hardcoded or inferred.

---

## 10. Confusion Matrix
Evaluated on 1,080 test samples:

| Actual \ Predicted | Glioma | Meningioma | Pituitary Tumor | No Tumor |
|---|:---:|:---:|:---:|:---:|
| **Actual Glioma** | **240** | 26 | 0 | 4 |
| **Actual Meningioma** | 9 | **236** | 23 | 2 |
| **Actual Pituitary Tumor** | 0 | 4 | **266** | 0 |
| **Actual No Tumor** | 0 | 5 | 3 | **262** |

---

## 11. Performance Metrics Summary

| Metric | Overall Performance |
|---|:---:|
| **Overall Accuracy** | **92.96%** |
| **Macro Precision** | **93.08%** |
| **Macro Sensitivity (Recall)** | **92.96%** |
| **Macro Specificity** | **97.65%** |
| **Macro F1-Score** | **92.95%** |

### Per-Class Performance Breakdown

| Class | Precision | Recall (Sensitivity) | Specificity | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Glioma** | 96.39% | 88.89% | 98.89% | 92.49% |
| **Meningioma** | 87.08% | 87.41% | 95.68% | 87.25% |
| **Pituitary Tumor** | 91.10% | 98.52% | 96.79% | 94.66% |
| **No Tumor** | 97.76% | 97.04% | 99.26% | 97.40% |

---

## 12. Project Limitations & Ethical Scope
- **Optional Candidate ROI**: The intensity-based Morphological Candidate ROI is an image-processing candidate visualization tool. It does not constitute a pixel-level U-Net tumor segmentation mask.
- **Sequence Sensitivity**: Classification performance was evaluated on axial brain MRI scans; correlation with multi-planar T1+C / T2 clinical sequences is recommended.

---

## 13. Academic & Clinical Disclaimer
`⚠️ AI model output for academic/research purposes only. Not a medical diagnosis.`
