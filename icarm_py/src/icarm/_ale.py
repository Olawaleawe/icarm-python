"""Accumulated Local Effects — mirrors R's icarm_ale()."""
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
class IcarmALE:
    """Result of icarm_ale(). Centred ALE profile."""
    ale: pd.DataFrame   # feature_value, ale_effect, n
    feature: str
    is_numeric: bool
    model: "IcarmModel"

    def plot(self, title: str | None = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        df = self.ale
        if self.is_numeric:
            x   = df["feature_value"].astype(float)
            ale = df["ale_effect"]
            ax.fill_between(x,
                            ale.clip(upper=0),
                            0,
                            alpha=0.3,
                            color=PALETTE["unfair"])
            ax.fill_between(x, 0, ale.clip(lower=0),
                            alpha=0.3,
                            color=PALETTE["secondary"])
            ax.plot(x, ale, color=PALETTE["primary"],
                    linewidth=2)
            ax.axhline(0, color="grey", linewidth=0.8,
                       linestyle="--")
        else:
            colors = [PALETTE["secondary"] if v >= 0
                      else PALETTE["unfair"]
                      for v in df["ale_effect"]]
            ax.bar(df["feature_value"].astype(str),
                   df["ale_effect"], color=colors, alpha=0.8)
            ax.axhline(0, color="grey", linewidth=0.8)

        ax.set_xlabel(self.feature)
        ax.set_ylabel("ALE effect (centred)")
        ax.set_title(title or f"ALE: {self.feature}",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["primary"])
        icarm_theme(ax)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        rng = (self.ale["ale_effect"].min(),
               self.ale["ale_effect"].max())
        return (f"icarm_ale\n  Feature : {self.feature}"
                f"\n  Intervals: {len(self.ale)}"
                f"\n  ALE range: [{rng[0]:.4f}, {rng[1]:.4f}]")


def icarm_ale(
    model: "IcarmModel",
    X: pd.DataFrame,
    feature: str,
    n_intervals: int = 20,
) -> IcarmALE:
    """
    Compute an Accumulated Local Effects (ALE) profile.

    ALE is unbiased for correlated predictors, unlike PDPs.
    Uses quantile-based bins and accumulates local differences.

    Parameters
    ----------
    model : IcarmModel
    X : pd.DataFrame
    feature : str
    n_intervals : int
        Number of quantile bins. Default 20.

    Returns
    -------
    IcarmALE

    Examples
    --------
    >>> ale = icarm_ale(model, X_test, feature="age")
    >>> ale.plot()
    """
    model._check_fitted()
    names  = model._feature_names
    X_arr  = X[names].copy()
    vals   = X_arr[feature]
    is_num = pd.api.types.is_numeric_dtype(vals)

    def _pred(df_sub: pd.DataFrame) -> np.ndarray:
        p = model._fitted_model.predict(df_sub[names].values)
        if model._task == "binary" and hasattr(
                model._fitted_model, "predict_proba"):
            pos_i = model._classes.index(model.positive)
            p = model._fitted_model.predict_proba(
                df_sub[names].values)[:, pos_i]
        return p.astype(float)

    rows = []
    if is_num:
        breaks = np.unique(np.percentile(
            vals.dropna(),
            np.linspace(0, 100, n_intervals + 1)))
        acc = 0.0
        for k in range(len(breaks) - 1):
            lo, hi = breaks[k], breaks[k + 1]
            idx = (vals >= lo) & (vals <= hi)
            if idx.sum() < 2:
                continue
            sub = X_arr[idx].copy()
            sub_lo = sub.copy(); sub_lo[feature] = lo
            sub_hi = sub.copy(); sub_hi[feature] = hi
            delta = float((_pred(sub_hi) - _pred(sub_lo)).mean())
            acc  += delta
            rows.append({"feature_value": (lo + hi) / 2,
                         "delta": delta,
                         "n": int(idx.sum())})
        df = pd.DataFrame(rows)
        if len(df):
            cumsum = df["delta"].cumsum()
            df["ale_effect"] = cumsum - cumsum.mean()
    else:
        levels = vals.dropna().unique()
        ref    = levels[0]
        deltas = {}
        for lv in levels[1:]:
            idx = vals.isin([ref, lv])
            if idx.sum() < 2:
                continue
            sub = X_arr[idx].copy()
            sub_ref = sub.copy(); sub_ref[feature] = ref
            sub_lv  = sub.copy(); sub_lv[feature]  = lv
            delta = float((_pred(sub_lv) - _pred(sub_ref)).mean())
            deltas[lv] = delta
            rows.append({"feature_value": lv, "delta": delta,
                         "n": int(idx.sum())})
        df = pd.DataFrame(rows)
        if len(df):
            df["ale_effect"] = df["delta"] - df["delta"].mean()

    return IcarmALE(ale=df, feature=feature,
                    is_numeric=is_num, model=model)
