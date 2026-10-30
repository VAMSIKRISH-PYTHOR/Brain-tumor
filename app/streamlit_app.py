"""
Stage 9: Streamlit Interactive Inference & Model Evaluation Application.

Project Title:
Brain Tumor Detection and Classification through Optimised Computational Intelligence using MRI Scans

Pipeline:
MRI -> Preprocessing -> EfficientNetB0 (1280 features) -> GA Selection (630 features) -> SVM (RBF) Classifier Head -> 4-Class Probabilities
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    CLASSES,
    IMAGE_SIZE,
    DEVICE
)
from src.preprocessing import preprocess_single_image
from src.segmentation import extract_roi_otsu
from src.feature_extractor import PyTorchFeatureExtractor

# Streamlit Page Config
st.set_page_config(
    page_title="Brain Tumor Detection & Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_models():
    """
    Cached loader for PyTorch Feature Extractor, GA Feature Mask, and Champion Classifier.
    """
    # 1. Feature Extractor
    extractor = PyTorchFeatureExtractor(weights_path=MODELS_DIR / "efficientnet_b0_finetuned.pth", device=DEVICE)

    # 2. GA Mask
    mask_path = MODELS_DIR / "ga_selected_features.npy"
    if mask_path.exists():
        ga_mask = np.load(mask_path).astype(bool)
    else:
        ga_mask = None

    # 3. Champion Classifier
    classifier_path = MODELS_DIR / "best_classifier.joblib"
    if classifier_path.exists():
        classifier = joblib.load(classifier_path)
    else:
        classifier = None

    # 4. Dict of all classifiers
    all_models_path = MODELS_DIR / "all_classifiers.joblib"
    if all_models_path.exists():
        all_classifiers = joblib.load(all_models_path)
    else:
        all_classifiers = {}

    return extractor, ga_mask, classifier, all_classifiers


@st.cache_data
def compute_test_metrics(models_dir_path: Path):
    """
    Computes ground-truth performance metrics directly from pre-saved test dataset arrays.
    """
    x_test_path = models_dir_path / "X_test.npy"
    y_test_path = models_dir_path / "y_test.npy"
    ga_path = models_dir_path / "ga_selected_features.npy"
    clf_path = models_dir_path / "best_classifier.joblib"

    if not (x_test_path.exists() and y_test_path.exists() and clf_path.exists()):
        return None

    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    ga_mask = np.load(ga_path).astype(bool) if ga_path.exists() else None
    clf = joblib.load(clf_path)

    X_sel = X_test[:, ga_mask] if (ga_mask is not None and X_test.shape[1] == len(ga_mask)) else X_test
    preds = clf.predict(X_sel)

    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds)

    class_names = ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]
    per_class = []
    sensitivities = []
    specificities = []

    for i, c_name in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_c = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

        sensitivities.append(sens)
        specificities.append(spec)
        per_class.append({
            "Class": c_name,
            "Precision": f"{prec * 100:.2f}%",
            "Recall (Sensitivity)": f"{sens * 100:.2f}%",
            "Specificity": f"{spec * 100:.2f}%",
            "F1-Score": f"{f1_c * 100:.2f}%"
        })

    macro_spec = np.mean(specificities)

    return {
        "total_samples": len(y_test),
        "accuracy": f"{acc * 100:.2f}%",
        "precision": f"{p * 100:.2f}%",
        "recall_sensitivity": f"{r * 100:.2f}%",
        "specificity": f"{macro_spec * 100:.2f}%",
        "f1_score": f"{f1 * 100:.2f}%",
        "per_class_df": pd.DataFrame(per_class),
        "cm_df": pd.DataFrame(cm, index=class_names, columns=class_names)
    }


def auto_crop_brain_skull(img_bgr: np.ndarray) -> np.ndarray:
    """
    Crops out DICOM text margins, headers/footers, and outer black borders
    so the brain skull fills the 224x224 input frame identically to standard dataset images.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        h_img, w_img = gray.shape

        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)

        if (x2 - x1) > 0.20 * w_img and (y2 - y1) > 0.20 * h_img:
            return img_bgr[y1:y2, x1:x2]

    return img_bgr


def main():
    st.title("🧠 Brain Tumor Detection and Classification through Optimised Computational Intelligence using MRI Scans")
    st.markdown(
        """
        **End-to-End AI Classification Pipeline**:
        Input MRI $\\rightarrow$ Preprocessing (CLAHE + Median Blur) $\\rightarrow$ EfficientNetB0 Deep Feature Extractor (1280 dims) $\\rightarrow$
        GA Feature Selection (630 dims) $\\rightarrow$ SVM (RBF) Classifier Head $\\rightarrow$ 4-Class Probability Prediction
        """
    )
    st.markdown("---")

    # Load models
    with st.spinner("Loading AI models and feature extraction pipelines..."):
        extractor, ga_mask, best_classifier, all_classifiers = load_models()

    # Sidebar Options
    st.sidebar.header("⚙️ Pipeline Settings")

    if all_classifiers:
        model_choice = st.sidebar.selectbox(
            "Select Classifier Head:",
            options=list(all_classifiers.keys()),
            index=0 if "SVM (RBF)" not in all_classifiers else list(all_classifiers.keys()).index("SVM (RBF)")
        )
        selected_classifier = all_classifiers[model_choice]
    else:
        selected_classifier = best_classifier
        model_choice = "SVM (RBF)"

    # Tumor ROI detection is always enabled — highlights tumor with red outline in panel 3
    enable_segmentation = True

    # System Overview Section (Task 5)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ System Overview")
    st.sidebar.markdown(
        f"""
        - **Feature Extractor**: EfficientNetB0 (Fine-tuned)
        - **Raw Features**: 1280
        - **Feature Selection**: Genetic Algorithm (GA)
        - **Selected Features**: {np.sum(ga_mask) if ga_mask is not None else 1280}
        - **Classifier**: SVM (RBF)
        - **Target Classes**: Glioma, Meningioma, Pituitary Tumor, No Tumor
        """
    )

    # Disclaimer Section (Task 6)
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ **Disclaimer**: AI model output for academic/research purposes only. Not a medical diagnosis.")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        # Load Raw Image
        pil_img = Image.open(uploaded_file).convert("RGB")
        raw_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Auto Crop Brain Skull to remove DICOM text margins & black padding
        cleaned_brain_bgr = auto_crop_brain_skull(raw_bgr)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("1. Input Raw MRI")
            st.image(pil_img, use_container_width=True, caption=f"Original Size: {pil_img.size[0]}x{pil_img.size[1]}")

        # Preprocessing on cleaned brain scan
        preprocessed_bgr = preprocess_single_image(cleaned_brain_bgr, target_size=IMAGE_SIZE)
        preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("2. Preprocessed MRI")
            st.image(preprocessed_rgb, use_container_width=True, caption="Auto Skull-Cropped, CLAHE + Median Blur")

        # Segmentation & Candidate ROI Validation
        roi_res = extract_roi_otsu(preprocessed_rgb)
        roi_mask, roi_cropped_rgb, bbox = roi_res[:3]
        roi_stats_ret = roi_res[3] if len(roi_res) == 4 else {}

        nonzero_px = int(np.sum(roi_mask > 0))
        total_px = int(roi_mask.size)
        fg_ratio = float((nonzero_px / total_px) * 100)
        num_labels, labels, cc_stats, centroids = cv2.connectedComponentsWithStats(roi_mask)
        largest_cc_area = int(np.max(cc_stats[1:, cv2.CC_STAT_AREA])) if num_labels > 1 else 0

        is_roi_valid = (nonzero_px > 0 and fg_ratio < 40.0 and nonzero_px < total_px and largest_cc_area > 0 and bbox != (0, 0, 0, 0))

        roi_mask_rgb = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
        input_for_extraction = preprocessed_bgr

        # NOTE: panel 3 overlay is rendered AFTER classification so we know the prediction.
        # We store it here and render below after inference.

        st.markdown("---")

        # Feature Extraction & GA Filtering
        with st.spinner("Extracting EfficientNetB0 features & applying GA selection mask..."):
            features_1280 = extractor.extract_features_single(input_for_extraction)

            if ga_mask is not None:
                features_selected = features_1280[ga_mask].reshape(1, -1)
            else:
                features_selected = features_1280.reshape(1, -1)

        # Classification Inference
        if selected_classifier is None:
            st.error("No trained classifier found! Please train classifiers in Stage 7 first.")
            return

        if hasattr(selected_classifier, "predict_proba"):
            probs = selected_classifier.predict_proba(features_selected)[0]
        else:
            pred_idx_raw = selected_classifier.predict(features_selected)[0]
            probs = np.zeros(len(CLASSES))
            probs[pred_idx_raw] = 1.0

        # Model classes mapping inspection & single source of truth
        model_classes = getattr(selected_classifier, "classes_", range(len(CLASSES)))

        # Mathematically strict consistency: pred_idx ALWAYS equals argmax of probability distribution
        pred_idx = int(np.argmax(probs))

        # Map to internal class string
        raw_class_val = model_classes[pred_idx]
        if isinstance(raw_class_val, (int, np.integer)):
            internal_class_name = CLASSES[raw_class_val]
        else:
            internal_class_name = str(raw_class_val).lower()

        # Single Source Display Mapping
        DISPLAY_NAMES = {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "pituitary": "Pituitary Tumor",
            "notumor": "No Tumor"
        }
        display_class_name = DISPLAY_NAMES.get(internal_class_name, internal_class_name.capitalize())
        model_confidence = float(probs[pred_idx] * 100)
        is_predicted_tumor = internal_class_name != "notumor"

        # Generate marked image using exact same single source of truth
        from src.segmentation import create_marked_mri_image
        marked_mri_rgb = create_marked_mri_image(preprocessed_rgb, roi_mask, bbox, internal_class_name, model_confidence)

        # Now render Panel 3 — gated on the actual model prediction
        with col3:
            st.subheader("3. Tumor ROI Detection")
            overlay_img = preprocessed_rgb.copy()
            if is_predicted_tumor and is_roi_valid and bbox != (0, 0, 0, 0):
                x_b, y_b, w_b, h_b = bbox
                contours_found, _ = cv2.findContours(
                    roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(overlay_img, contours_found, -1, (255, 0, 0), 2)
                cv2.rectangle(
                    overlay_img,
                    (x_b, y_b),
                    (x_b + w_b, y_b + h_b),
                    (255, 0, 0), 3
                )
                roi_caption = f"🔴 Tumor ROI highlighted — {nonzero_px} px ({fg_ratio:.1f}% coverage)"
            else:
                roi_caption = "✅ No distinct tumor ROI detected in this scan"
            st.image(overlay_img, use_container_width=True, caption=roi_caption)

        # Display Classification Results & Candidate ROI Analysis (Task 2)
        st.subheader("📍 AI Classification & ROI Analysis")

        m_col1, m_col2 = st.columns([1, 1.1])

        with m_col1:
            caption_text = (
                "Model prediction with tumor ROI contour highlighted"
                if (is_predicted_tumor and is_roi_valid)
                else "Model prediction on preprocessed MRI scan"
            )
            st.image(
                marked_mri_rgb,
                use_container_width=True,
                caption=caption_text
            )
            # Convert marked RGB to bytes for download
            pil_marked = Image.fromarray(marked_mri_rgb)
            buf = io.BytesIO()
            pil_marked.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Classification Image",
                data=buf.getvalue(),
                file_name=f"classification_mri_{internal_class_name}.png",
                mime="image/png"
            )

        with m_col2:
            # Color badge depending on class (Task 2: Model Prediction & Model Confidence)
            if internal_class_name == "notumor":
                st.success(f"### Model Prediction: **{display_class_name}**")
            elif internal_class_name == "glioma":
                st.error(f"### Model Prediction: **{display_class_name}**")
            elif internal_class_name == "meningioma":
                st.warning(f"### Model Prediction: **{display_class_name}**")
            else:  # pituitary
                st.warning(f"### Model Prediction: **{display_class_name}**")

            st.metric(label="Model Confidence", value=f"{model_confidence:.2f}%")

            # ROI status must agree with model prediction: only show candidate
            # region as active when the model actually predicts a tumor class.
            roi_status_msg = (
                "🔴 Candidate region detected"
                if (is_roi_valid and is_predicted_tumor)
                else "✅ No tumor region detected"
            )
            st.markdown(f"**Tumor ROI Status**: {roi_status_msg}")

            st.markdown(f"**Classifier Head**: `{model_choice}`")
            st.markdown(f"**Features Evaluated**: `{features_selected.shape[1]}` / `1280` GA-optimized features")

            st.markdown("#### Probability Distribution Across Classes")

            # Map all graph labels strictly using model classes
            graph_labels = []
            for i in range(len(CLASSES)):
                cls_val = model_classes[i] if i < len(model_classes) else i
                cls_str = CLASSES[cls_val] if isinstance(cls_val, (int, np.integer)) else str(cls_val).lower()
                graph_labels.append(DISPLAY_NAMES.get(cls_str, cls_str.capitalize()))

            df_probs = pd.DataFrame({
                "Class": graph_labels,
                "Probability (%)": probs * 100
            })

            fig, ax = plt.subplots(figsize=(6, 2.5))
            bars = ax.barh(df_probs["Class"], df_probs["Probability (%)"], color=["#d62728" if label == display_class_name else "#1f77b4" for label in graph_labels])
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontweight="bold")
            ax.grid(axis="x", linestyle=":", alpha=0.6)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", ha="left", va="center", fontsize=9, fontweight="bold")

            st.pyplot(fig)

            # Verification Telemetry Expander
            with st.expander("🔍 Pipeline Verification Telemetry"):
                telemetry_data = {
                    "input_raw_shape": list(raw_bgr.shape),
                    "preprocessed_image_shape": list(preprocessed_rgb.shape),
                    "efficientnet_raw_feature_shape": list(features_1280.shape),
                    "ga_selected_feature_shape": list(features_selected.shape),
                    "classifier_input_shape": list(features_selected.shape),
                    "predicted_class_index": pred_idx,
                    "internal_class_name": internal_class_name,
                    "display_class_name": display_class_name,
                    "classifier_classes_array": [int(c) if isinstance(c, (int, np.integer)) else str(c) for c in model_classes],
                    "probability_values": {label: f"{prob * 100:.2f}%" for label, prob in zip(graph_labels, probs)},
                    "morphological_roi_enabled": enable_segmentation,
                    "candidate_roi_validity": is_roi_valid if enable_segmentation else "N/A (Disabled)"
                }
                st.json(telemetry_data)

    else:
        st.info("👆 Please upload a Brain MRI image above to start automated classification analysis.")

    # ---------------------------------------------------------
    # Model Performance & Evaluation Section (Task 3 & Task 4)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Model Performance & Evaluation Metrics (Ground-Truth Test Dataset)")

    with st.expander("📈 View Comprehensive Model Evaluation Metrics & Confusion Matrix (1,080 Test Samples)", expanded=True):
        eval_res = compute_test_metrics(MODELS_DIR)
        if eval_res is not None:
            e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
            e_col1.metric("Overall Accuracy", eval_res["accuracy"])
            e_col2.metric("Macro Precision", eval_res["precision"])
            e_col3.metric("Macro Sensitivity (Recall)", eval_res["recall_sensitivity"])
            e_col4.metric("Macro Specificity", eval_res["specificity"])
            e_col5.metric("Macro F1-Score", eval_res["f1_score"])

            st.markdown("#### Per-Class Classification Performance")
            st.dataframe(eval_res["per_class_df"], use_container_width=True)

            st.markdown("#### Confusion Matrix (Actual Ground-Truth vs Model Prediction)")
            st.dataframe(eval_res["cm_df"], use_container_width=True)

            st.caption(
                f"ℹ️ Performance dynamically calculated from {eval_res['total_samples']} test dataset samples using ground-truth labels (`y_test.npy`)."
            )

            # Exportable Results (Task 4)
            st.markdown("#### 📥 Export Evaluation Results")
            summary_df = pd.DataFrame([{
                "Total Test Samples": eval_res["total_samples"],
                "Accuracy": eval_res["accuracy"],
                "Macro Precision": eval_res["precision"],
                "Macro Sensitivity (Recall)": eval_res["recall_sensitivity"],
                "Macro Specificity": eval_res["specificity"],
                "Macro F1-Score": eval_res["f1_score"]
            }])

            csv_metrics = summary_df.to_csv(index=False)
            csv_per_class = eval_res["per_class_df"].to_csv(index=False)
            csv_cm = eval_res["cm_df"].to_csv()

            json_report = json.dumps({
                "project_title": "Brain Tumor Detection and Classification through Optimised Computational Intelligence using MRI Scans",
                "total_test_samples": eval_res["total_samples"],
                "macro_metrics": {
                    "accuracy": eval_res["accuracy"],
                    "precision": eval_res["precision"],
                    "recall_sensitivity": eval_res["recall_sensitivity"],
                    "specificity": eval_res["specificity"],
                    "f1_score": eval_res["f1_score"]
                },
                "per_class_metrics": eval_res["per_class_df"].to_dict(orient="records"),
                "confusion_matrix": eval_res["cm_df"].to_dict()
            }, indent=2)

            exp_col1, exp_col2, exp_col3 = st.columns(3)
            exp_col1.download_button(
                label="📥 Export Metrics Summary (CSV)",
                data=csv_metrics,
                file_name="model_metrics_summary.csv",
                mime="text/csv"
            )
            exp_col2.download_button(
                label="📥 Export Per-Class & Confusion Matrix (CSV)",
                data=f"--- PER-CLASS PERFORMANCE ---\n{csv_per_class}\n\n--- CONFUSION MATRIX ---\n{csv_cm}",
                file_name="per_class_and_confusion_matrix.csv",
                mime="text/csv"
            )
            exp_col3.download_button(
                label="📥 Export Full Report (JSON)",
                data=json_report,
                file_name="evaluation_report.json",
                mime="application/json"
            )
        else:
            st.warning("Ground-truth test evaluation dataset not found in `models/` directory.")

    st.markdown("---")
    st.caption("⚠️ **Disclaimer**: AI model output for academic/research purposes only. Not a medical diagnosis.")


if __name__ == "__main__":
    main()
