"""
Fitness Evaluation Module for Feature Selection.

Evaluates binary feature masks using validation accuracy of a ultra-fast
proxy classifier (RidgeClassifier) minus a parsimony penalty for feature dimensionality.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score


def evaluate_fitness(
    mask: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 0.01
) -> tuple[float, float, int]:
    """
    Computes fitness score for a candidate binary feature mask.

    Args:
        mask: Binary array of shape (D,) where 1 indicates feature selected, 0 dropped.
        X_train: Training features matrix (N_train, D).
        y_train: Training labels array (N_train,).
        X_val: Validation features matrix (N_val, D).
        y_val: Validation labels array (N_val,).
        alpha: Penalty weight per selected feature ratio.

    Returns:
        fitness: Combined fitness score.
        val_acc: Validation accuracy of proxy classifier.
        num_selected: Count of active features in mask.
    """
    mask_bool = mask.astype(bool)
    num_selected = int(np.sum(mask_bool))
    total_features = len(mask)

    # Edge case: If no features selected, return zero fitness
    if num_selected == 0:
        return 0.0, 0.0, 0

    # Select feature columns
    X_tr_sub = X_train[:, mask_bool]
    X_v_sub = X_val[:, mask_bool]

    # Train ultra-fast RidgeClassifier proxy (analytical closed-form solution)
    try:
        clf = RidgeClassifier(alpha=1.0, random_state=42)
        clf.fit(X_tr_sub, y_train)
        val_preds = clf.predict(X_v_sub)
        val_acc = accuracy_score(y_val, val_preds)
    except Exception:
        val_acc = 0.0

    # Parsimony penalty proportional to fraction of retained features
    penalty = alpha * (num_selected / total_features)
    fitness = val_acc - penalty

    return float(fitness), float(val_acc), num_selected
