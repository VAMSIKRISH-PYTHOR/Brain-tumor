"""
Classifier Training, Hyperparameter Tuning, and Benchmark Comparison Module.

Trains and evaluates:
1. Softmax (Multinomial Logistic Regression / MLP) Classifier
2. Support Vector Machine (SVM) with RBF Kernel
3. Random Forest Classifier
4. XGBoost Classifier

Uses 5-Fold Cross-Validation and GridSearchCV hyperparameter tuning on the GA-selected feature subset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODELS_DIR,
    CLASSES,
    RANDOM_SEED
)


def load_ga_features():
    """
    Loads raw feature arrays and applies the GA-selected binary feature mask.
    """
    X_train_raw = np.load(MODELS_DIR / "X_train.npy")
    y_train = np.load(MODELS_DIR / "y_train.npy")
    X_val_raw = np.load(MODELS_DIR / "X_val.npy")
    y_val = np.load(MODELS_DIR / "y_val.npy")
    X_test_raw = np.load(MODELS_DIR / "X_test.npy")
    y_test = np.load(MODELS_DIR / "y_test.npy")

    mask_path = MODELS_DIR / "ga_selected_features.npy"
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        print(f"Applying GA Feature Mask: Retaining {np.sum(mask)} / {len(mask)} features.")
    else:
        print("[WARNING] GA feature mask not found. Using full 1280 features.")
        mask = np.ones(X_train_raw.shape[1], dtype=bool)

    X_train = X_train_raw[:, mask]
    X_val = X_val_raw[:, mask]
    X_test = X_test_raw[:, mask]

    return X_train, y_train, X_val, y_val, X_test, y_test, mask


def train_and_tune_classifiers(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Trains and tunes Softmax, SVM, Random Forest, and XGBoost classifiers.
    """
    # Combine Train + Val for 5-fold cross-validation tuning
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # 1. Softmax (Logistic Regression) Head
    print("\n--- [1/4] Tuning Softmax Classifier ---", flush=True)
    softmax_param_grid = {"C": [0.1, 1.0, 10.0]}
    softmax_base = LogisticRegression(solver="lbfgs", max_iter=500, random_state=RANDOM_SEED)
    grid_softmax = GridSearchCV(softmax_base, softmax_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_softmax.fit(X_train_val, y_train_val)
    best_softmax = grid_softmax.best_estimator_
    print(f"Softmax Best Params: {grid_softmax.best_params_} | CV Acc: {grid_softmax.best_score_*100:.2f}%", flush=True)

    # 2. SVM (RBF Kernel)
    print("\n--- [2/4] Tuning SVM (RBF Kernel) Classifier ---", flush=True)
    svm_param_grid = {
        "C": [1.0, 10.0],
        "gamma": ["scale"]
    }
    svm_base = SVC(kernel="rbf", random_state=RANDOM_SEED)
    grid_svm = GridSearchCV(svm_base, svm_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_svm.fit(X_train_val, y_train_val)
    best_svm_raw = grid_svm.best_estimator_

    # Calibrate probabilities
    from sklearn.calibration import CalibratedClassifierCV
    best_svm = CalibratedClassifierCV(estimator=best_svm_raw, ensemble=False)
    best_svm.fit(X_train_val, y_train_val)
    print(f"SVM Best Params: {grid_svm.best_params_} | CV Acc: {grid_svm.best_score_*100:.2f}%", flush=True)

    # 3. Random Forest Classifier
    print("\n--- [3/4] Tuning Random Forest Classifier ---", flush=True)
    rf_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [20, None]
    }
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1)
    grid_rf = GridSearchCV(rf_base, rf_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_rf.fit(X_train_val, y_train_val)
    best_rf = grid_rf.best_estimator_
    print(f"Random Forest Best Params: {grid_rf.best_params_} | CV Acc: {grid_rf.best_score_*100:.2f}%", flush=True)

    # 4. XGBoost Classifier
    print("\n--- [4/4] Tuning XGBoost Classifier ---", flush=True)
    xgb_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.1]
    }
    xgb_base = xgb.XGBClassifier(tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_SEED, n_jobs=-1)
    grid_xgb = GridSearchCV(xgb_base, xgb_param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_xgb.fit(X_train_val, y_train_val)
    best_xgb = grid_xgb.best_estimator_
    print(f"XGBoost Best Params: {grid_xgb.best_params_} | CV Acc: {grid_xgb.best_score_*100:.2f}%", flush=True)

    # Model Benchmark Comparison
    models_dict = {
        "Softmax": best_softmax,
        "SVM (RBF)": best_svm,
        "Random Forest": best_rf,
        "XGBoost": best_xgb
    }

    comparison_results = []
    best_overall_model = None
    best_overall_score = 0.0
    best_overall_name = ""

    for name, model in models_dict.items():
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)

        test_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)

        comparison_results.append({
            "Classifier": name,
            "Validation Accuracy": f"{val_acc*100:.2f}%",
            "Test Accuracy": f"{test_acc*100:.2f}%",
            "val_acc_raw": val_acc,
            "test_acc_raw": test_acc
        })

        if test_acc > best_overall_score:
            best_overall_score = test_acc
            best_overall_model = model
            best_overall_name = name

    df_comp = pd.DataFrame(comparison_results).drop(columns=["val_acc_raw", "test_acc_raw"])
    print("\n" + "=" * 60, flush=True)
    print("             CLASSIFIER BENCHMARK COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(df_comp.to_string(index=False), flush=True)
    print("=" * 60 + "\n", flush=True)

    print(f"Selected Champion Model: {best_overall_name} with Test Accuracy: {best_overall_score*100:.2f}%", flush=True)

    # Save champion model to models/best_classifier.joblib
    champion_path = MODELS_DIR / "best_classifier.joblib"
    joblib.dump(best_overall_model, champion_path)
    print(f"Saved best classifier model to {champion_path}", flush=True)

    # Also save dictionary of all trained models for evaluation benchmarking
    all_models_path = MODELS_DIR / "all_classifiers.joblib"
    joblib.dump(models_dict, all_models_path)

    return best_overall_name, best_overall_model


def main():
    print("Starting Stage 7: Classifier Training & Hyperparameter Tuning...")
    X_train, y_train, X_val, y_val, X_test, y_test, mask = load_ga_features()
    train_and_tune_classifiers(X_train, y_train, X_val, y_val, X_test, y_test)
    print("Stage 7 completed successfully!")


if __name__ == "__main__":
    main()
