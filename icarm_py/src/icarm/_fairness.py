"""Fairness auditing — mirrors R's icarm_fairness()."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icarm._utils import PALETTE, detect_task, icarm_theme
if TYPE_CHECKING:
    from icarm._fit import IcarmModel

@dataclass
class IcarmFairness:
    """Result of icarm_fairness(). Per-group metrics."""
    metrics: pd.DataFrame
    task: str
    protected: str
    model: "IcarmModel"

    def summary(self) -> dict:
        """Return equity summary (mirrors R's icarm_equity_summary)."""
        return icarm_equity_summary(self)

    def plot(self, metric: str = "tpr_gap",
             ref_line: float | None = None,
             title: str | None = None) -> plt.Figure:
        """Horizontal lollipop chart per group."""
        if metric not in self.metrics.columns:
            raise ValueError(
                f"metric '{metric}' not available. "
                f"Options: {list(self.metrics.columns)}")
        df = self.metrics.sort_values(metric)
        vals = df[metric]
        colors = [PALETTE["fair"] if v >= (ref_line or 0)
                  else PALETTE["unfair"] for v in vals]
        fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.55)))
        ax.hlines(df[self.protected].astype(str), 0, vals,
                  color=PALETTE["neutral"], linewidth=1.5)
        ax.scatter(vals, df[self.protected].astype(str),
                   c=colors, s=80, zorder=3)
        if ref_line is not None:
            ax.axvline(ref_line, color="red", linewidth=1,
                       linestyle="--", alpha=0.7,
                       label=f"Reference = {ref_line}")
            ax.legend(fontsize=9)
        ax.set_xlabel(metric)
        ax.set_ylabel(self.protected)
        ax.set_title(title or f"Fairness: {metric} by {self.protected}",
                     fontsize=12, fontweight="bold",
                     color=PALETTE["primary"])
        icarm_theme(ax)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        lines = [
            "icarm_fairness",
            f"  Protected: {self.protected}",
            f"  Task     : {self.task}",
            f"  Groups   : {len(self.metrics)}",
        ]
        print("\n".join(lines))
        print(self.metrics.to_string(index=False))
        return ""


def icarm_fairness(
    model: "IcarmModel",
    X: pd.DataFrame,
    y,
    protected: str,
) -> IcarmFairness:
    """
    Compute per-group fairness metrics across a protected attribute.

    Binary: accuracy, TPR, FPR, PPV, DPR (demographic parity ratio),
    EO gap (equalized odds gap), with 80% rule pass/fail.
    Regression: MAE and RMSE gaps relative to the lowest-error group.
    Multiclass: per-group balanced accuracy.

    Parameters
    ----------
    model : IcarmModel
    X : pd.DataFrame
    y : array-like
        True labels or values.
    protected : str
        Column name of the protected attribute in X.

    Returns
    -------
    IcarmFairness

    Examples
    --------
    >>> f = icarm_fairness(model, X_test, y_test, protected="gender")
    >>> f.plot(metric="tpr_gap")
    >>> f.summary()
    """
    model._check_fitted()
    y_arr = np.asarray(y)
    task  = model._task
    grp   = X[protected].astype(str)
    X_feat = X[model._feature_names]  ## select only trained features
    y_hat = model.predict(X_feat)

    rows = []
    if task == "binary":
        pos = model.positive
        proba = None
        if hasattr(model._fitted_model, "predict_proba"):
            pos_i = model._classes.index(pos)
            proba = model._fitted_model.predict_proba(X_feat[model._feature_names].values)[:, pos_i]

        # Reference = highest positive rate group
        pos_rates = {g: (y_hat[grp == g] == pos).mean()
                     for g in grp.unique()}
        max_rate  = max(pos_rates.values()) + 1e-9

        for g in grp.unique():
            mask = grp == g
            yt, yp = y_arr[mask], y_hat[mask]
            tp  = ((yp == pos) & (yt == pos)).sum()
            fp  = ((yp == pos) & (yt != pos)).sum()
            tn  = ((yp != pos) & (yt != pos)).sum()
            fn  = ((yp != pos) & (yt == pos)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            acc = (tp + tn) / len(yt)
            dpr = pos_rates[g] / max_rate
            rows.append({protected: g, "n": mask.sum(),
                         "acc": round(acc, 4),
                         "tpr": round(tpr, 4),
                         "fpr": round(fpr, 4),
                         "ppv": round(ppv, 4),
                         "dp_ratio": round(dpr, 4)})

    elif task == "regression":
        for g in grp.unique():
            mask = grp == g
            yt, yp = y_arr[mask].astype(float), y_hat[mask].astype(float)
            rows.append({protected: g, "n": int(mask.sum()),
                         "mae": round(float(np.abs(yt - yp).mean()), 4),
                         "rmse": round(float(np.sqrt(((yt - yp) ** 2).mean())), 4)})

        df = pd.DataFrame(rows)
        ref_mae = df["mae"].min()
        ref_rmse = df["rmse"].min()
        df["mae_gap"]  = (df["mae"]  - ref_mae).round(4)
        df["rmse_gap"] = (df["rmse"] - ref_rmse).round(4)
        return IcarmFairness(metrics=df, task=task,
                             protected=protected, model=model)

    else:  # multiclass
        from sklearn.metrics import balanced_accuracy_score
        ref_bacc = 0.0
        for g in grp.unique():
            mask = grp == g
            bacc = balanced_accuracy_score(y_arr[mask], y_hat[mask])
            ref_bacc = max(ref_bacc, bacc)
            rows.append({protected: g, "n": int(mask.sum()),
                         "balanced_acc": round(bacc, 4)})
        df = pd.DataFrame(rows)
        df["bacc_gap"] = (ref_bacc - df["balanced_acc"]).round(4)
        return IcarmFairness(metrics=df, task=task,
                             protected=protected, model=model)

    return IcarmFairness(metrics=pd.DataFrame(rows), task=task,
                         protected=protected, model=model)


def icarm_equity_summary(fairness: IcarmFairness) -> dict:
    """
    Equity summary (mirrors R's icarm_equity_summary).

    Returns pass/fail flags for the 80% rule (demographic parity)
    and equalized odds.

    Parameters
    ----------
    fairness : IcarmFairness

    Returns
    -------
    dict
    """
    df   = fairness.metrics
    task = fairness.task
    out  = {"n_groups": len(df)}

    if task == "binary" and "dp_ratio" in df.columns:
        out["min_dp_ratio"]           = float(df["dp_ratio"].min())
        out["disparate_impact_pass"]  = bool(df["dp_ratio"].min() >= 0.8)
        if "tpr" in df.columns:
            out["max_tpr_gap"] = float(df["tpr"].max() - df["tpr"].min())
            out["equal_opp_pass"] = bool(out["max_tpr_gap"] <= 0.1)
    elif task == "regression":
        out["max_mae_gap"]  = float(df["mae_gap"].max())
        out["max_rmse_gap"] = float(df["rmse_gap"].max())
    else:
        out["max_bacc_gap"] = float(df["bacc_gap"].max()) \
            if "bacc_gap" in df.columns else None

    print("icarm_equity_summary:")
    for k, v in out.items():
        print(f"  {k:<30} {v}")
    return out
