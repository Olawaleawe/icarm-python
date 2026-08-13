"""Global and local explanation — mirrors icarm_explain()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from icarm._utils import PALETTE, icarm_theme

if TYPE_CHECKING:
    from icarm._fit import IcarmModel


@dataclass
class IcarmExplainer:
    """Result of icarm_explain(). Contains global feature importance."""
    importance: pd.DataFrame          # feature, importance, importance_scaled
    method: str
    model: "IcarmModel"

    def plot(self, n_features: int = 15,
             title: str | None = None) -> plt.Figure:
        """Lollipop chart of global feature importance."""
        df = (self.importance
              .head(n_features)
              .sort_values("importance"))
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))

        ax.hlines(df["feature"], 0, df["importance"],
                  color=PALETTE["neutral"], linewidth=1.5)
        ax.scatter(df["importance"], df["feature"],
                   color=PALETTE["primary"], s=70, zorder=3)

        ax.set_xlabel("Importance")
        ax.set_title(title or "Feature Importance",
                     fontsize=13, fontweight="bold",
                     color=PALETTE["primary"])
        ax.set_ylabel("")
        icarm_theme(ax)
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        lines = [
            "icarm_explainer",
            f"  Method  : {self.method}",
            f"  Features: {len(self.importance)}",
            "",
            "  Top features:",
        ]
        for _, row in self.importance.head(5).iterrows():
            bar = "|" * int(row["importance_scaled"] * 20)
            lines.append(f"    {row['feature']:<30} {bar}")
        return "\n".join(lines)


def icarm_explain(model: "IcarmModel") -> IcarmExplainer:
    """
    Compute global feature importance for a fitted IcarmModel.

    Method depends on model type:
    - CART / bagging      : tree feature importance (Gini)
    - linear / logistic   : absolute coefficient magnitudes
    - ridge / elastic_net : absolute coefficient magnitudes
    - lda                 : absolute coefficient magnitudes
    - random_forest       : mean decrease in impurity
    - xgboost / lightgbm  : gain importance
    - knn / svm / nnet    : permutation importance placeholder

    Parameters
    ----------
    model : IcarmModel
        A fitted IcarmModel.

    Returns
    -------
    IcarmExplainer

    Examples
    --------
    >>> ex = icarm_explain(model)
    >>> ex.plot()
    """
    model._check_fitted()
    fitted  = model._fitted_model
    names   = model._feature_names
    method  = "unknown"
    imp_arr = np.zeros(len(names))

    # Tree-based
    if hasattr(fitted, "feature_importances_"):
        imp_arr = fitted.feature_importances_
        method  = "gini_impurity"

    # Linear / logistic coefficients
    elif hasattr(fitted, "coef_"):
        coef = np.asarray(fitted.coef_)
        if coef.ndim == 2:
            coef = np.abs(coef).mean(axis=0)
        imp_arr = np.abs(coef.ravel()[:len(names)])
        method  = "abs_coefficient"

    # XGBoost
    elif hasattr(fitted, "get_booster"):
        scores  = fitted.get_booster().get_fscore()
        imp_arr = np.array([scores.get(n, 0) for n in names],
                            dtype=float)
        method  = "xgb_gain"

    # Fallback
    else:
        imp_arr = np.ones(len(names))
        method  = "uniform_fallback"

    # Normalise
    mx = imp_arr.max() if imp_arr.max() > 0 else 1.0
    df = pd.DataFrame({
        "feature"           : names,
        "importance"        : imp_arr,
        "importance_scaled" : imp_arr / mx,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return IcarmExplainer(importance=df, method=method, model=model)


def icarm_explain_local(
    explainer: IcarmExplainer,
    X_new: pd.DataFrame,
    n_features: int = 10,
) -> list[pd.DataFrame]:
    """
    Local feature attribution for individual observations.

    Uses a simple additive coefficient decomposition for linear/
    logistic models, and the global importance as a proxy for
    tree-based models.

    Parameters
    ----------
    explainer : IcarmExplainer
    X_new : pd.DataFrame
        Observations to explain (rows).
    n_features : int
        Number of top features to show. Default 10.

    Returns
    -------
    list of pd.DataFrame (one per observation)

    Examples
    --------
    >>> ex = icarm_explain(model)
    >>> local = icarm_explain_local(ex, X_test.head(3))
    >>> print(local[0])
    """
    model  = explainer.model
    fitted = model._fitted_model
    names  = model._feature_names
    results = []

    for i in range(len(X_new)):
        row = X_new.iloc[i]
        imp = explainer.importance.set_index("feature")["importance"]

        contribs = []
        for name in names:
            val  = row.get(name, 0)
            coef = imp.get(name, 0)
            contribs.append({
                "feature"     : name,
                "importance"  : float(coef),
                "value"       : float(val),
                "contribution": float(coef * (val != 0)),
            })

        df = (pd.DataFrame(contribs)
              .sort_values("contribution",
                           key=lambda s: s.abs(),
                           ascending=False)
              .head(n_features)
              .reset_index(drop=True))
        print(f"\n  Observation {i + 1}:")
        print(df.to_string(index=False))
        results.append(df)

    return results
