"""
Stage 8: Comprehensive Model Evaluation & Metrics Module.

Evaluates all trained classifiers (Softmax, SVM, Random Forest, XGBoost) on the test dataset.
Computes:
- Accuracy, Precision, Recall / Sensitivity, Specificity, F1-score (Macro, Weighted, Per-Class)
- One-vs-Rest ROC-AUC curves and per-class AUC scores
- Confusion Matrices for all classifiers
- Saves reports to reports/metrics.json and plots to reports/figures/
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    CLASSES,
    RANDOM_SEED
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_per_class_specificity(cm):
    """
    Calculates per-class specificity from a multi-class confusion matrix.
    Specificity = TN / (TN + FP)
    """
    num_classes = cm.shape[0]
    specificities = []
    total_samples = cm.sum()

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total_samples - (tp + fp + fn)

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)

    return np.array(specificities)


def evaluate_classifier(model, X_test, y_test, class_names):
    """
    Evaluates a single classifier on the test set and returns comprehensive metrics.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        # Softmax normalization for decision function if multi-class
        exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
        y_prob = exp_df / np.sum(exp_df, axis=1, keepdims=True)
    else:
        y_prob = None

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Precision, Recall (Sensitivity), F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(class_names)), zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

    # Per-class Specificity
    specificity = calculate_per_class_specificity(cm)

    # Macro Averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    macro_specificity = np.mean(specificity)

    # Multi-class One-vs-Rest ROC-AUC
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    roc_auc_per_class = {}
    macro_roc_auc = 0.0

    if y_prob is not None:
        auc_list = []
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            class_auc = auc(fpr, tpr)
            roc_auc_per_class[class_names[i]] = float(class_auc)
            auc_list.append(class_auc)
        macro_roc_auc = float(np.mean(auc_list))

    metrics = {
        "accuracy": float(acc),
        "macro_metrics": {
            "precision": float(macro_precision),
            "recall_sensitivity": float(macro_recall),
            "specificity": float(macro_specificity),
            "f1_score": float(macro_f1),
            "roc_auc": float(macro_roc_auc)
        },
        "per_class": {}
    }

    for i, cname in enumerate(class_names):
        metrics["per_class"][cname] = {
            "precision": float(precision[i]),
            "recall_sensitivity": float(recall[i]),
            "specificity": float(specificity[i]),
            "f1_score": float(f1[i]),
            "roc_auc": float(roc_auc_per_class.get(cname, 0.0)),
            "support": int(support[i])
        }

    return metrics, cm, y_prob


def plot_confusion_matrices(all_cms, model_names, class_names, save_path):
    """
    Plots a 2x2 grid of confusion matrices for all evaluated classifiers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(zip(model_names, all_cms)):
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices grid plot to: {save_path}")


def plot_roc_curves(all_probs, y_test, model_names, class_names, save_path):
    """
    Plots a 2x2 grid of One-vs-Rest ROC Curves for each classifier across 4 classes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (name, y_prob) in enumerate(zip(model_names, all_probs)):
        ax = axes[idx]
        if y_prob is None:
            ax.text(0.5, 0.5, "ROC Unavailable", ha="center", va="center")
            continue

        macro_auc_sum = 0.0
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            class_auc = auc(fpr, tpr)
            macro_auc_sum += class_auc
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{cname} (AUC = {class_auc:.4f})")

        mean_auc = macro_auc_sum / len(class_names)
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
        ax.set_title(f"ROC Curves ({name}) | Macro AUC: {mean_auc:.4f}", fontsize=13, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves grid plot to: {save_path}")


def plot_metrics_benchmark(all_metrics, model_names, save_path):
    """
    Plots a grouped bar chart comparing Accuracy, Precision, Recall, Specificity, F1, and ROC-AUC across classifiers.
    """
    metric_keys = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]

    data = []
    for name in model_names:
        m = all_metrics[name]
        row = [m["accuracy"]]
        for k in metric_keys[1:]:
            row.append(m["macro_metrics"][k])
        data.append(row)

    df = pd.DataFrame(data, columns=metric_labels, index=model_names)

    ax = df.plot(kind="bar", figsize=(14, 7), colormap="viridis", width=0.8)
    plt.title("Classifier Benchmark Comparison Across Key Evaluation Metrics", fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.ylim(0.8, 1.02)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=11)
    plt.xticks(rotation=0, fontsize=12, fontweight="bold")

    # Add numeric labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark comparison bar chart to: {save_path}")


def main():
    print("=" * 70)
    print("STAGE 8: COMPREHENSIVE MODEL EVALUATION & METRICS BENCHMARKING")
    print("=" * 70)

    # 1. Load test data and GA feature mask
    X_test_raw = np.load(MODELS_DIR / "X_test.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")

    mask_path = MODELS_DIR / "ga_selected_features.npy"
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        print(f"Applying GA Feature Mask: Retaining {np.sum(mask)} / {len(mask)} features.")
    else:
        print("[WARNING] GA feature mask not found. Using full 1280 features.")
        mask = np.ones(X_test_raw.shape[1], dtype=bool)

    X_test = X_test_raw[:, mask]

    # 2. Load trained models
    models_path = MODELS_DIR / "all_classifiers.joblib"
    if not models_path.exists():
        raise FileNotFoundError(f"Trained models dict not found at {models_path}. Run Stage 7 first!")

    models_dict = joblib.load(models_path)
    model_names = list(models_dict.keys())
    print(f"Loaded {len(model_names)} models for evaluation: {model_names}")

    # 3. Evaluate each model
    all_metrics = {}
    all_cms = []
    all_probs = []

    table_rows = []

    for name, model in models_dict.items():
        metrics, cm, y_prob = evaluate_classifier(model, X_test, y_test, CLASSES)
        all_metrics[name] = metrics
        all_cms.append(cm)
        all_probs.append(y_prob)

        macro = metrics["macro_metrics"]
        table_rows.append({
            "Classifier": name,
            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Precision": f"{macro['precision']*100:.2f}%",
            "Recall (Sens)": f"{macro['recall_sensitivity']*100:.2f}%",
            "Specificity": f"{macro['specificity']*100:.2f}%",
            "F1-Score": f"{macro['f1_score']*100:.2f}%",
            "ROC-AUC": f"{macro['roc_auc']:.4f}"
        })

    # 4. Print Summary Benchmark Table
    df_summary = pd.DataFrame(table_rows)
    print("\n" + "=" * 80)
    print("                     FINAL EVALUATION BENCHMARK SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")

    # 5. Save metrics to JSON
    json_path = REPORTS_DIR / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved evaluation metrics to: {json_path}")

    # 6. Generate and save visualization plots
    plot_confusion_matrices(all_cms, model_names, CLASSES, FIGURES_DIR / "confusion_matrices.png")
    plot_roc_curves(all_probs, y_test, model_names, CLASSES, FIGURES_DIR / "roc_curves.png")
    plot_metrics_benchmark(all_metrics, model_names, FIGURES_DIR / "metrics_comparison.png")

    print("\nStage 8 evaluation completed successfully!")


if __name__ == "__main__":
    main()
