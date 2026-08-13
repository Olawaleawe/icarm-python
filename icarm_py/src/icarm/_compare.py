"""Multi-model comparison — mirrors R's icarm_compare()."""
from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icarm._utils import PALETTE, icarm_theme
if TYPE_CHECKING:
    from icarm._fit import IcarmModel


def icarm_compare(
    models: dict[str, "IcarmModel"],
    X_test: pd.DataFrame,
    y_test,
    protected: str | None = None,
    positive: str | None = None,
) -> pd.DataFrame:
    """
    Compare multiple fitted IcarmModel objects on the same test set.

    Parameters
    ----------
    models : dict[str, IcarmModel]
        Named dictionary of fitted models.
    X_test : pd.DataFrame
    y_test : array-like
    protected : str, optional
        Protected attribute column for fairness comparison.
    positive : str, optional
        Positive class (binary only).

    Returns
    -------
    pd.DataFrame  Metric comparison table.

    Examples
    --------
    >>> cmp = icarm_compare({"Logistic": m1, "CART": m2},
    ...                      X_test, y_test, protected="gender")
    >>> cmp.plot()
    """
    import io, sys
    rows = []
    for name, m in models.items():
        m._check_fitted()
        y_hat = m.predict(X_test)
        y_p   = None
        if m._task == "binary" and hasattr(
                m._fitted_model, "predict_proba"):
            pos_i = m._classes.index(m.positive)
            y_p = m._fitted_model.predict_proba(
                X_test[m._feature_names].values)[:, pos_i]
        old = sys.stdout; sys.stdout = io.StringIO()
        from icarm._metrics import icarm_metrics
        met = icarm_metrics(y_test, y_hat, y_prob=y_p,
                             positive=m.positive or positive,
                             task=m._task)
        sys.stdout = old
        row = {"model": name}; row.update(met)

        # Fairness
        if protected and protected in X_test.columns:
            from icarm._fairness import icarm_fairness, icarm_equity_summary
            old = sys.stdout; sys.stdout = io.StringIO()
            f   = icarm_fairness(m, X_test, y_test, protected=protected)
            eq  = icarm_equity_summary(f)
            sys.stdout = old
            row.update({f"fairness_{k}": v for k, v in eq.items()
                        if k != "n_groups"})
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\nicarm_compare:")
    print(df.to_string(index=False))
    return df
