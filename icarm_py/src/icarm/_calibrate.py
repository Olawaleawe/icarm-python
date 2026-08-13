"""Calibration and threshold sweep — mirrors icarm_calibrate / icarm_thresholds."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icarm._utils import PALETTE, icarm_theme
if TYPE_CHECKING:
    from icarm._fit import IcarmModel

@dataclass
class IcarmCalibration:
    brier: float
    ece: float
    bins: pd.DataFrame
    model: "IcarmModel"

    def plot(self, title: str | None = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 6))
        df = self.bins.dropna()
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.scatter(df["mean_prob"], df["mean_obs"],
                   s=df["n"] / df["n"].max() * 200,
                   color=PALETTE["primary"], alpha=0.8,
                   label="Calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction")
        ax.set_title(title or "Calibration (Reliability Diagram)",
                     fontsize=12, fontweight="bold",
                     color=PALETTE["primary"])
        ax.legend(fontsize=9)
        icarm_theme(ax)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        ece_rating = "GOOD" if self.ece < 0.05 else (
            "MODERATE" if self.ece < 0.10 else "POOR")
        return (f"icarm_calibration\n"
                f"  Brier score : {self.brier:.4f}\n"
                f"  ECE         : {self.ece:.4f}  [{ece_rating}]")


def icarm_calibrate(
    model: "IcarmModel",
    X: pd.DataFrame,
    y,
    positive: str | None = None,
    n_bins: int = 10,
) -> IcarmCalibration:
    """
    Compute calibration diagnostics (Brier score and ECE).

    Parameters
    ----------
    model : IcarmModel
    X : pd.DataFrame
    y : array-like
    positive : str, optional
    n_bins : int
        Number of bins. Default 10.

    Returns
    -------
    IcarmCalibration
    """
    model._check_fitted()
    if not hasattr(model._fitted_model, "predict_proba"):
        raise ValueError("Calibration requires a probabilistic model.")
    pos = positive or model.positive
    pos_i = model._classes.index(pos)
    proba = model._fitted_model.predict_proba(
        X[model._feature_names].values)[:, pos_i]
    y_bin = (np.asarray(y) == pos).astype(float)

    brier = float(np.mean((proba - y_bin) ** 2))
    breaks = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 0:
            rows.append({"mean_prob": proba[mask].mean(),
                         "mean_obs" : y_bin[mask].mean(),
                         "n"        : int(mask.sum())})
    df = pd.DataFrame(rows)
    if len(df) > 0:
        ece = float(((df["mean_prob"] - df["mean_obs"]).abs()
                     * df["n"] / len(y_bin)).sum())
    else:
        ece = 0.0
    return IcarmCalibration(brier=brier, ece=ece,
                             bins=df, model=model)


def icarm_thresholds(
    model: "IcarmModel",
    X: pd.DataFrame,
    y,
    positive: str | None = None,
    thresholds=None,
) -> pd.DataFrame:
    """
    Threshold sweep — metrics at each decision threshold.

    Parameters
    ----------
    model : IcarmModel
    X : pd.DataFrame
    y : array-like
    positive : str, optional
    thresholds : array-like, optional
        Threshold grid. Default np.arange(0.10, 0.91, 0.05).

    Returns
    -------
    pd.DataFrame with columns: threshold, accuracy, f1, precision,
    recall, specificity.
    """
    from sklearn import metrics as skm
    model._check_fitted()
    pos = positive or model.positive
    pos_i = model._classes.index(pos)
    proba = model._fitted_model.predict_proba(
        X[model._feature_names].values)[:, pos_i]
    y_arr = np.asarray(y)
    grid  = (np.arange(0.10, 0.91, 0.05)
             if thresholds is None else np.asarray(thresholds))
    rows  = []
    for thr in grid:
        yhat = np.where(proba >= thr, pos,
                        [c for c in model._classes if c != pos][0])
        cm   = skm.confusion_matrix(y_arr, yhat,
                                     labels=model._classes)
        tp   = cm[model._classes.index(pos),
                   model._classes.index(pos)]
        fp   = cm[:, model._classes.index(pos)].sum() - tp
        fn   = cm[model._classes.index(pos), :].sum() - tp
        tn   = cm.sum() - tp - fp - fn
        rows.append({
            "threshold"  : round(thr, 2),
            "accuracy"   : round((tp + tn) / len(y_arr), 4),
            "f1"         : round(2 * tp / (2 * tp + fp + fn + 1e-9), 4),
            "precision"  : round(tp / (tp + fp + 1e-9), 4),
            "recall"     : round(tp / (tp + fn + 1e-9), 4),
            "specificity": round(tn / (tn + fp + 1e-9), 4),
        })
    return pd.DataFrame(rows)
