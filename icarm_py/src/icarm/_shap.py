"""SHAP attribution — mirrors R's icarm_shap()."""

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
class IcarmShap:
    """Result of icarm_shap(). Marginal SHAP approximation."""
    shap_values : pd.DataFrame    # obs x features
    mean_abs    : pd.Series       # mean |SHAP| per feature
    baseline    : float
    model       : "IcarmModel"

    def plot(self, title: str | None = None,
             n_features: int = 10) -> plt.Figure:
        """Beeswarm plot: each dot = one observation."""
        top_feats = (self.mean_abs
                     .sort_values(ascending=False)
                     .head(n_features)
                     .index.tolist())
        df_top = self.shap_values[top_feats]

        fig, ax = plt.subplots(figsize=(9, max(4, len(top_feats) * 0.5)))
        cmap = plt.get_cmap("coolwarm")

        for y_pos, feat in enumerate(reversed(top_feats)):
            vals = df_top[feat].values
            # Scale feature values for colour
            feat_scaled = (vals - vals.min()) / (
                (vals.max() - vals.min()) + 1e-9)
            colors = cmap(feat_scaled)
            jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(vals, y_pos + jitter, c=colors,
                       alpha=0.6, s=18, zorder=2)

        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(list(reversed(top_feats)), fontsize=9)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value")
        ax.set_title(title or "SHAP Attributions",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["primary"])
        icarm_theme(ax)
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Feature value (scaled)", fontsize=8)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        n_obs = len(self.shap_values)
        lines = [
            "icarm_shap",
            f"  Observations : {n_obs}",
            f"  Baseline     : {self.baseline:.4f}",
            "",
            "  Mean |SHAP| (top 5):",
        ]
        for feat, val in self.mean_abs.sort_values(
                ascending=False).head(5).items():
            bar = "|" * min(20, int(val / self.mean_abs.max() * 20))
            lines.append(f"    {feat:<30} {bar}")
        return "\n".join(lines)


def icarm_shap(
    model: "IcarmModel",
    X: pd.DataFrame,
    n_samples: int = 50,
    random_state: int | None = 2025,
) -> IcarmShap:
    """
    Compute model-agnostic approximate SHAP values.

    Uses the marginal interventional estimator (Equation 1 in the paper):
        phi_j(i) = (1/M) * sum_k [ f(x_j^i, x_{-j}^k)
                                  - f(x_j^k, x_{-j}^i) ]

    where M = n_samples background observations.

    Parameters
    ----------
    model : IcarmModel
        A fitted model.
    X : pd.DataFrame
        Data to explain (can be test set or a sample).
    n_samples : int
        Background sample size M. Default 50.
    random_state : int, optional
        Seed for background sampling. Default 2025.

    Returns
    -------
    IcarmShap

    Examples
    --------
    >>> shap = icarm_shap(model, X_test, n_samples=50)
    >>> shap.plot()
    """
    model._check_fitted()
    rng     = np.random.default_rng(random_state)
    n, p    = X.shape
    names   = model._feature_names
    X_arr   = X[names].values.astype(float)

    # Background sample
    bg_idx  = rng.choice(n, size=min(n_samples, n), replace=False)
    bg      = X_arr[bg_idx]

    # Predict function (returns scalar prediction per row)
    def _pred(Xm: np.ndarray) -> np.ndarray:
        df_m = pd.DataFrame(Xm, columns=names)
        if model._task == "regression":
            return model._fitted_model.predict(Xm).ravel()
        if model._task == "binary":
            proba = model._fitted_model.predict_proba(Xm)
            pos_i = model._classes.index(model.positive)
            return proba[:, pos_i]
        # Multiclass — return max probability
        return model._fitted_model.predict_proba(Xm).max(axis=1)

    baseline = float(_pred(bg).mean())

    # Compute SHAP for every observation
    shap_mat = np.zeros((n, p))
    for i in range(n):
        x_i = X_arr[i]
        for j in range(p):
            # Marginal interventional estimator
            Xhi  = bg.copy(); Xhi[:, j]  = x_i[j]
            Xlo  = bg.copy(); Xlo[:, j]  = bg[:, j]
            shap_mat[i, j] = float(
                (_pred(Xhi) - _pred(Xlo)).mean()
            )

    sv_df     = pd.DataFrame(shap_mat, columns=names)
    mean_abs  = sv_df.abs().mean().sort_values(ascending=False)

    return IcarmShap(
        shap_values=sv_df,
        mean_abs=mean_abs,
        baseline=baseline,
        model=model,
    )
