"""Data drift detection — mirrors R's icarm_drift()."""
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
class IcarmDrift:
    drift: pd.DataFrame
    n_train: int
    n_new: int
    model: "IcarmModel"

    def plot(self, title: str | None = None) -> plt.Figure:
        df = self.drift.copy()
        color_map = {"none": PALETTE["fair"],
                     "moderate": PALETTE["accent"],
                     "high": PALETTE["unfair"]}
        colors = df["psi_flag"].map(color_map).tolist()
        df_s   = df.sort_values("statistic")
        fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.45)))
        ax.hlines(df_s["feature"], 0, df_s["statistic"],
                  color=PALETTE["neutral"], linewidth=1.5)
        ax.scatter(df_s["statistic"], df_s["feature"],
                   c=df_s["psi_flag"].map(color_map), s=70, zorder=3)
        ax.set_xlabel("Drift statistic")
        ax.set_title(title or "Data Drift Summary",
                     fontsize=12, fontweight="bold",
                     color=PALETTE["primary"])
        icarm_theme(ax)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        h = (self.drift["psi_flag"] == "high").sum()
        m = (self.drift["psi_flag"] == "moderate").sum()
        n = len(self.drift) - h - m
        return (f"icarm_drift\n"
                f"  Reference: {self.n_train} obs | New: {self.n_new} obs\n"
                f"  Features checked: {len(self.drift)}\n"
                f"  High drift: {h} | Moderate: {m} | None: {n}")


def icarm_drift(
    model: "IcarmModel",
    X_train: pd.DataFrame,
    X_new: pd.DataFrame,
    n_bins: int = 10,
) -> IcarmDrift:
    """
    Detect covariate drift using PSI (numeric) and chi-square (categorical).

    PSI < 0.1 = no drift; 0.1-0.2 = moderate; > 0.2 = high.

    Parameters
    ----------
    model : IcarmModel
    X_train : pd.DataFrame  Reference (training) data.
    X_new   : pd.DataFrame  New (deployment) data.
    n_bins  : int           Bins for PSI. Default 10.

    Returns
    -------
    IcarmDrift

    Examples
    --------
    >>> drift = icarm_drift(model, X_train, X_test)
    >>> drift.plot()
    """
    from scipy.stats import chi2_contingency
    features = model._feature_names
    rows = []
    for feat in features:
        tr = X_train[feat].dropna()
        nw = X_new[feat].dropna()
        if pd.api.types.is_numeric_dtype(tr):
            breaks = np.percentile(tr,
                np.linspace(0, 100, n_bins + 1))
            breaks = np.unique(breaks)
            if len(breaks) < 3:
                continue
            breaks[0] = -np.inf; breaks[-1] = np.inf
            p_tr = np.bincount(
                np.searchsorted(breaks[1:-1], tr.values),
                minlength=len(breaks) - 1) / len(tr)
            p_nw = np.bincount(
                np.searchsorted(breaks[1:-1], nw.values),
                minlength=len(breaks) - 1) / len(nw)
            p_tr = np.clip(p_tr, 1e-4, None)
            p_nw = np.clip(p_nw, 1e-4, None)
            psi  = float(np.sum((p_nw - p_tr) * np.log(p_nw / p_tr)))
            flag = ("high" if psi > 0.2 else
                    "moderate" if psi > 0.1 else "none")
            rows.append({"feature": feat, "method": "PSI",
                         "statistic": round(psi, 4),
                         "psi_flag": flag})
        else:
            lvs  = sorted(set(tr.astype(str)) | set(nw.astype(str)))
            f_tr = pd.Categorical(tr.astype(str),
                                   categories=lvs).value_counts()
            f_nw = pd.Categorical(nw.astype(str),
                                   categories=lvs).value_counts()
            ct   = np.vstack([f_tr.values, f_nw.values])
            if ct.sum() == 0 or (ct.sum(axis=0) == 0).any():
                continue
            try:
                chi2, p, *_ = chi2_contingency(ct)
                flag = ("high" if p < 0.01 else
                        "moderate" if p < 0.05 else "none")
                rows.append({"feature": feat, "method": "chi-square",
                             "statistic": round(float(chi2), 4),
                             "psi_flag": flag})
            except Exception:
                pass
    return IcarmDrift(drift=pd.DataFrame(rows),
                      n_train=len(X_train),
                      n_new=len(X_new),
                      model=model)
