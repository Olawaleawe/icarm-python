"""Performance metrics — mirrors R's icarm_metrics()."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from icarm._utils import detect_task


def icarm_metrics(
    y_true,
    y_pred,
    y_prob: np.ndarray | None = None,
    positive: str | None = None,
    task: str | None = None,
) -> dict[str, float]:
    """
    Compute task-appropriate performance metrics.

    Binary classification: accuracy, balanced_acc, f1, precision,
    recall, specificity, auc.
    Multiclass: accuracy, balanced_acc, f1, precision, recall
    (all macro-averaged).
    Regression: mae, rmse, r2.

    Parameters
    ----------
    y_true : array-like
        True labels or values.
    y_pred : array-like
        Predicted labels or values.
    y_prob : array-like, optional
        Predicted probabilities (binary positive class).
    positive : str, optional
        Positive class label (binary only).
    task : str, optional
        Override task detection.

    Returns
    -------
    dict[str, float]

    Examples
    --------
    >>> icarm_metrics(y_test, y_pred, y_prob=proba, positive="Yes")
    {'accuracy': 0.82, 'balanced_acc': 0.79, ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    detected = task or detect_task(y_true)

    if detected == "regression":
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        result = {"mae": mae, "rmse": rmse, "r2": r2}

    elif detected == "binary":
        pos_label = positive or sorted(np.unique(y_true))[-1]
        acc  = float(skm.accuracy_score(y_true, y_pred))
        bacc = float(skm.balanced_accuracy_score(y_true, y_pred))
        f1   = float(skm.f1_score(y_true, y_pred,
                                   pos_label=pos_label,
                                   zero_division=0))
        prec = float(skm.precision_score(y_true, y_pred,
                                          pos_label=pos_label,
                                          zero_division=0))
        rec  = float(skm.recall_score(y_true, y_pred,
                                       pos_label=pos_label,
                                       zero_division=0))
        # Specificity = TNR
        cm   = skm.confusion_matrix(y_true, y_pred,
                                     labels=[pos_label,
                                             *[c for c in
                                               np.unique(y_true)
                                               if c != pos_label]])
        tn   = cm[1:, 1:].sum() if cm.shape[0] > 1 else 0
        fp   = cm[1:, 0].sum()  if cm.shape[0] > 1 else 0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        result = {
            "accuracy": acc, "balanced_acc": bacc,
            "f1": f1, "precision": prec,
            "recall": rec, "specificity": spec,
        }
        if y_prob is not None:
            try:
                auc = float(skm.roc_auc_score(
                    y_true == pos_label, y_prob))
                result["auc"] = auc
            except Exception:
                pass

    else:  # multiclass
        acc  = float(skm.accuracy_score(y_true, y_pred))
        bacc = float(skm.balanced_accuracy_score(y_true, y_pred))
        f1   = float(skm.f1_score(y_true, y_pred,
                                   average="macro", zero_division=0))
        prec = float(skm.precision_score(y_true, y_pred,
                                          average="macro",
                                          zero_division=0))
        rec  = float(skm.recall_score(y_true, y_pred,
                                       average="macro",
                                       zero_division=0))
        result = {
            "accuracy": acc, "balanced_acc": bacc,
            "f1": f1, "precision": prec, "recall": rec,
        }

    # Print formatted output (mirrors R print behaviour)
    print("\n  ".join(["icarm_metrics:"] +
          [f"{k:<18} {v:.4f}" for k, v in result.items()]))
    return result
