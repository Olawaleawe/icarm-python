"""Partial Dependence Profiles — mirrors R's icarm_pdp()."""
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
class IcarmPDP:
    """Result of icarm_pdp()."""
    pdp: pd.DataFrame   # feature_value, mean_pred, p10, p90
    feature: str
    is_numeric: bool
    model: "IcarmModel"

    def plot(self, title: str | None = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        df = self.pdp
        if self.is_numeric:
            x = df["feature_value"].astype(float)
            ax.fill_between(x, df["p10"], df["p90"],
                            alpha=0.2,
                            color=PALETTE["secondary"],
                            label="10th-90th pct")
            ax.plot(x, df["mean_pred"],
                    color=PALETTE["primary"],
                    linewidth=2, label="PDP mean")
            ax.legend(fontsize=9)
        else:
            ax.bar(df["feature_value"].astype(str),
                   df["mean_pred"],
                   color=PALETTE["secondary"], alpha=0.8)
            ax.errorbar(df["feature_value"].astype(str),
                        df["mean_pred"],
                        yerr=[df["mean_pred"] - df["p10"],
                              df["p90"] - df["mean_pred"]],
                        fmt="none", color=PALETTE["primary"],
                        capsize=4)
        ax.set_xlabel(self.feature)
        ax.set_ylabel("Predicted value")
        ax.set_title(title or f"PDP: {self.feature}",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["primary"])
        icarm_theme(ax)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        return (f"icarm_pdp\n  Feature: {self.feature}"
                f" ({'numeric' if self.is_numeric else 'categorical'})"
                f"\n  Grid points: {len(self.pdp)}")


def icarm_pdp(
    model: "IcarmModel",
    X: pd.DataFrame,
    feature: str,
    n_intervals: int = 20,
) -> IcarmPDP:
    """
    Compute a Partial Dependence Profile for one feature.

    PDP(v) = (1/n) * sum_i f(v, x_{-j}^i)

    augmented with 10th-90th percentile band over individual
    predictions.

    Parameters
    ----------
    model : IcarmModel
    X : pd.DataFrame
    feature : str
        Feature name to profile.
    n_intervals : int
        Grid size for numeric features. Default 20.

    Returns
    -------
    IcarmPDP

    Examples
    --------
    >>> pdp = icarm_pdp(model, X_test, feature="age")
    >>> pdp.plot()
    """
    model._check_fitted()
    names = model._feature_names
    X_arr = X[names].copy()
    vals  = X_arr[feature]
    is_num = pd.api.types.is_numeric_dtype(vals)

    grid = (
        np.percentile(vals.dropna(),
                      np.linspace(0, 100, n_intervals + 1))
        if is_num
        else vals.dropna().unique()
    )
    grid = np.unique(grid)

    def _pred_mean_pct(v):
        X_mod = X_arr.copy()
        X_mod[feature] = v
        p = model._fitted_model.predict(X_mod[names].values)
        if model._task == "binary" and hasattr(
                model._fitted_model, "predict_proba"):
            pos_i = model._classes.index(model.positive)
            p = model._fitted_model.predict_proba(
                X_mod[names].values)[:, pos_i]
        p = p.astype(float)
        return float(p.mean()), float(np.percentile(p, 10)), float(np.percentile(p, 90))

    rows = [{"feature_value": v, "mean_pred": m,
             "p10": lo, "p90": hi}
            for v in grid for m, lo, hi in [_pred_mean_pct(v)]]
    return IcarmPDP(
        pdp=pd.DataFrame(rows),
        feature=feature,
        is_numeric=is_num,
        model=model,
    )
