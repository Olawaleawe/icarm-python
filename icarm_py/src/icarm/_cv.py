"""Cross-validation and learning curves — mirrors icarm_cv / icarm_learning_curve."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from icarm._utils import PALETTE, icarm_theme
if TYPE_CHECKING:
    from icarm._fit import IcarmModel

@dataclass
class IcarmCV:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    task: str
    folds: int

    def plot(self, title: str | None = None) -> plt.Figure:
        metrics = [c for c in self.fold_metrics.columns if c != "fold"]
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4),
                                  squeeze=False)
        for i, met in enumerate(metrics):
            ax = axes[0, i]
            vals = self.fold_metrics[met]
            mean = self.summary.loc[self.summary["metric"] == met,
                                    "mean"].values[0]
            ax.bar(self.fold_metrics["fold"], vals,
                   color=PALETTE["secondary"], alpha=0.75, width=0.6)
            ax.axhline(mean, color=PALETTE["primary"],
                       linewidth=1.5, linestyle="--",
                       label=f"mean={mean:.3f}")
            ax.set_title(met, fontsize=10, fontweight="bold")
            ax.set_xlabel("Fold")
            ax.legend(fontsize=8)
            icarm_theme(ax)
        fig.suptitle(title or f"{self.folds}-Fold CV",
                     fontsize=12, fontweight="bold",
                     color=PALETTE["primary"])
        fig.tight_layout()
        plt.show()
        return fig

    def __repr__(self) -> str:
        lines = [f"icarm_cv  [{self.folds} folds | {self.task}]"]
        for _, row in self.summary.iterrows():
            lines.append(f"  {row['metric']:<20} "
                         f"{row['mean']:.4f}  (SD {row['sd']:.4f})")
        return "\n".join(lines)


def icarm_cv(
    model: "IcarmModel",
    X: pd.DataFrame,
    y,
    folds: int = 5,
    random_state: int | None = 2025,
) -> IcarmCV:
    """
    Stratified k-fold cross-validation.

    Parameters
    ----------
    model : IcarmModel (unfitted or fitted — re-fitted per fold)
    X : pd.DataFrame
    y : array-like
    folds : int
        Number of folds. Default 5.
    random_state : int, optional

    Returns
    -------
    IcarmCV

    Examples
    --------
    >>> cv = icarm_cv(model, X, y, folds=5)
    >>> print(cv)
    >>> cv.plot()
    """
    from icarm._metrics import icarm_metrics
    from icarm._fit import IcarmModel as IM
    y_arr = np.asarray(y)
    task  = model._task or model.task
    if task == "auto":
        from icarm._utils import detect_task
        task = detect_task(y_arr)

    if task in ("binary", "multiclass"):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                                    random_state=random_state)
    else:
        splitter = KFold(n_splits=folds, shuffle=True,
                          random_state=random_state)

    fold_rows = []
    for k, (tr_i, va_i) in enumerate(splitter.split(X, y_arr), 1):
        X_tr, X_va = X.iloc[tr_i], X.iloc[va_i]
        y_tr, y_va = y_arr[tr_i], y_arr[va_i]
        m_k = IM(model=model.model, task=task,
                 positive=model.positive,
                 random_state=random_state)
        try:
            m_k.fit(X_tr, y_tr)
        except Exception:
            continue
        y_hat = m_k.predict(X_va)
        y_p   = None
        if task == "binary" and m_k.positive:
            try:
                y_p = m_k.predict(X_va, type="prob")[m_k.positive].values
            except Exception:
                pass
        # Suppress print inside loop
        import io, sys
        old = sys.stdout; sys.stdout = io.StringIO()
        met = icarm_metrics(y_va, y_hat, y_prob=y_p,
                             positive=m_k.positive, task=task)
        sys.stdout = old
        row = {"fold": k}; row.update(met)
        fold_rows.append(row)

    df_folds = pd.DataFrame(fold_rows)
    metric_cols = [c for c in df_folds.columns if c != "fold"]
    summary = pd.DataFrame([{
        "metric": m,
        "mean": df_folds[m].mean(),
        "sd"  : df_folds[m].std(),
    } for m in metric_cols])

    return IcarmCV(fold_metrics=df_folds, summary=summary,
                   task=task, folds=folds)


@dataclass
class IcarmLearningCurve:
    data: pd.DataFrame   # size, train_error, val_error, train_sd, val_sd
    model: "IcarmModel"
    metric: str = "error"

    def plot(self, title: str | None = None) -> plt.Figure:
        df = self.data
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["size"], df["train_error"],
                color=PALETTE["primary"], linewidth=2, label="Train")
        ax.fill_between(df["size"],
                        df["train_error"] - df["train_sd"],
                        df["train_error"] + df["train_sd"],
                        alpha=0.15, color=PALETTE["primary"])
        ax.plot(df["size"], df["val_error"],
                color=PALETTE["secondary"], linewidth=2,
                label="Validation")
        ax.fill_between(df["size"],
                        df["val_error"] - df["val_sd"],
                        df["val_error"] + df["val_sd"],
                        alpha=0.15, color=PALETTE["secondary"])
        ax.set_xlabel("Training set size")
        ax.set_ylabel(self.metric)
        ax.set_title(title or f"Learning Curve -- {self.model.model}",
                     fontsize=12, fontweight="bold",
                     color=PALETTE["primary"])
        ax.legend(fontsize=9)
        icarm_theme(ax)
        fig.tight_layout()
        plt.show()
        return fig


def icarm_learning_curve(
    model: "IcarmModel",
    X: pd.DataFrame,
    y,
    cv: int = 5,
    sizes=None,
    random_state: int | None = 2025,
) -> IcarmLearningCurve:
    """
    Compute learning curves via repeated cross-validation.

    Parameters
    ----------
    model : IcarmModel
    X : pd.DataFrame
    y : array-like
    cv : int
        Number of CV folds. Default 5.
    sizes : array-like, optional
        Training fractions. Default linspace(0.1, 0.9, 9).
    random_state : int, optional

    Returns
    -------
    IcarmLearningCurve
    """
    from icarm._fit import IcarmModel as IM
    from sklearn.model_selection import learning_curve
    from icarm._utils import detect_task

    if sizes is None:
        sizes = np.linspace(0.1, 0.9, 9)
    task = model._task or detect_task(np.asarray(y))
    sklearn_m = IM(model=model.model, task=task,
                   positive=model.positive,
                   random_state=random_state)
    sklearn_m.fit(X, y)
    sk_est = sklearn_m._fitted_model

    from sklearn.model_selection import learning_curve as lc_fn
    train_s, test_s, _ = lc_fn(
        sk_est, X, y,
        train_sizes=sizes,
        cv=cv,
        scoring=("neg_mean_absolute_error"
                 if task == "regression"
                 else "accuracy"),
        return_times=True,
        n_jobs=1,
    )[:3]

    sign = -1 if task == "regression" else 1
    df = pd.DataFrame({
        "size"      : train_s,
        "train_error": (sign * train_s).mean(axis=1) * sign
                       if task == "regression"
                       else 1 - test_s.mean(axis=1),
        "val_error"  : 1 - test_s.mean(axis=1),
        "train_sd"   : train_s.std(axis=1),
        "val_sd"     : test_s.std(axis=1),
    })

    return IcarmLearningCurve(data=df, model=model)
