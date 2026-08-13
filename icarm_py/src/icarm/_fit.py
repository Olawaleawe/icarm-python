"""
Core model fitting — IcarmModel class and icarm_fit() / icarm_split().
Mirrors the R icarm_fit() entry point with automatic task detection
and 18 model types spanning interpretable and extended families.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from icarm._utils import data_hash, detect_task

# ── Supported models ──────────────────────────────────────────────────
_INTERPRETABLE = {
    "cart", "logistic", "logistic_l1", "linear", "ridge",
    "elastic_net", "gam", "multinomial", "naive_bayes", "lda", "knn",
}
_EXTENDED = {
    "random_forest", "xgboost", "svm",
    "lightgbm", "bagging", "nnet_reg",
}
_ALL_MODELS = _INTERPRETABLE | _EXTENDED | {"auto", "custom"}

_VALID_BY_TASK: dict[str, set[str]] = {
    "binary": {
        "auto", "cart", "logistic", "logistic_l1",
        "ridge", "elastic_net", "gam",
        "naive_bayes", "lda", "knn",
        "random_forest", "xgboost", "svm",
        "lightgbm", "bagging", "nnet_reg", "custom",
    },
    "multiclass": {
        "auto", "cart", "multinomial",
        "naive_bayes", "lda", "knn",
        "random_forest", "xgboost",
        "lightgbm", "bagging", "nnet_reg", "custom",
    },
    "regression": {
        "auto", "cart", "linear", "gam",
        "ridge", "elastic_net", "knn",
        "random_forest", "xgboost", "svm",
        "lightgbm", "bagging", "nnet_reg", "custom",
    },
}


# ── IcarmModel ────────────────────────────────────────────────────────
class IcarmModel:
    """
    Unified entry point for ICARM-compliant machine learning.

    Automatically detects the prediction task from the target variable
    type (regression / binary / multiclass) and fits the requested
    model with full provenance metadata.

    Parameters
    ----------
    model : str
        Model type. One of: 'auto', 'cart', 'logistic', 'logistic_l1',
        'linear', 'ridge', 'elastic_net', 'gam', 'multinomial',
        'naive_bayes', 'lda', 'knn', 'random_forest', 'xgboost',
        'svm', 'lightgbm', 'bagging', 'nnet_reg', 'custom'.
        Default 'auto' selects 'cart'.
    task : str
        'auto' (default), 'binary', 'multiclass', or 'regression'.
    positive : str or None
        Positive class label for binary classification.
    random_state : int or None
        Random seed for reproducibility. Default 2025.

    Examples
    --------
    >>> from icarm import IcarmModel
    >>> import pandas as pd
    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer(as_frame=True)
    >>> X, y = data.data, data.target.map({0: "malignant", 1: "benign"})
    >>> model = IcarmModel(model="logistic", positive="benign")
    >>> model.fit(X, y)
    >>> model.metrics(X, y)
    """

    def __init__(
        self,
        model: str = "auto",
        task: str = "auto",
        positive: str | None = None,
        random_state: int | None = 2025,
    ) -> None:
        self.model        = model
        self.task         = task
        self.positive     = positive
        self.random_state = random_state

        # Set after fit()
        self._fitted_model: Any = None
        self._task: str | None  = None
        self._classes: list     = []
        self._feature_names: list[str] = []
        self._n_train: int      = 0
        self._data_hash: str    = ""
        self._trained_at: str   = ""
        self._formula: str      = ""

    # ── fit ────────────────────────────────────────────────────────────
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "IcarmModel":
        """
        Fit the model on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series or array-like
            Target variable.
        feature_names : list[str], optional
            Override column names from X.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Task detection
        self._task = (
            detect_task(y) if self.task == "auto" else self.task
        )

        # Model selection
        model_name = self.model
        if model_name == "auto":
            model_name = "cart"

        if model_name not in _VALID_BY_TASK.get(self._task, set()):
            raise ValueError(
                f"model='{model_name}' not available for "
                f"task='{self._task}'. "
                f"Valid: {sorted(_VALID_BY_TASK[self._task])}"
            )

        if model_name in _EXTENDED:
            warnings.warn(
                f"'{model_name}' is an extended (non-interpretable) model. "
                "Use icarm_shap() or icarm_ale() for post-hoc explanation.",
                UserWarning,
                stacklevel=2,
            )

        # Feature names
        if feature_names is not None:
            self._feature_names = list(feature_names)
        elif isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = [f"X{i}" for i in range(X.shape[1])]

        # Convert
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y)

        # Classes
        if self._task in ("binary", "multiclass"):
            self._classes = sorted(np.unique(y_arr).tolist())
            if self.positive is None and self._task == "binary":
                self.positive = self._classes[-1]

        # Fit
        self._fitted_model = _build_sklearn(
            model_name, self._task, self.random_state
        )
        self._fitted_model.fit(X_arr, y_arr)

        # Provenance
        self._n_train    = len(y_arr)
        self._data_hash  = data_hash(X)
        self._trained_at = datetime.now(timezone.utc).isoformat()
        self.model       = model_name
        self._formula    = (
            f"{getattr(y, 'name', 'y')} ~ "
            + " + ".join(self._feature_names)
        )
        return self

    # ── predict ────────────────────────────────────────────────────────
    def predict(
        self,
        X: pd.DataFrame,
        type: Literal["class", "prob"] = "class",
    ) -> np.ndarray | pd.DataFrame:
        """
        Generate predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        type : {'class', 'prob'}
            Return predicted classes or probabilities.
        """
        self._check_fitted()
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        if type == "prob":
            if not hasattr(self._fitted_model, "predict_proba"):
                raise ValueError(
                    f"model='{self.model}' does not support "
                    "probability prediction."
                )
            proba = self._fitted_model.predict_proba(X_arr)
            return pd.DataFrame(proba, columns=self._classes)
        return self._fitted_model.predict(X_arr)

    # ── convenience wrappers ───────────────────────────────────────────
    def metrics(self, X: pd.DataFrame, y) -> dict[str, float]:
        """Compute task-appropriate performance metrics."""
        from icarm._metrics import icarm_metrics
        yhat = self.predict(X)
        y_prob = None
        if self._task == "binary":
            y_prob = self.predict(X, type="prob")[self.positive].values
        return icarm_metrics(y, yhat, y_prob=y_prob,
                              positive=self.positive, task=self._task)

    def explain(self) -> "IcarmExplainer":
        """Compute global feature importance."""
        from icarm._explain import icarm_explain
        return icarm_explain(self)

    def shap(self, X: pd.DataFrame, n_samples: int = 50,
             **kw) -> "IcarmShap":
        """Compute approximate SHAP values."""
        from icarm._shap import icarm_shap
        return icarm_shap(self, X, n_samples=n_samples, **kw)

    def pdp(self, X: pd.DataFrame, feature: str,
            n_intervals: int = 20) -> "IcarmPDP":
        """Compute Partial Dependence Profile."""
        from icarm._pdp import icarm_pdp
        return icarm_pdp(self, X, feature=feature,
                         n_intervals=n_intervals)

    def ale(self, X: pd.DataFrame, feature: str,
            n_intervals: int = 20) -> "IcarmALE":
        """Compute Accumulated Local Effects profile."""
        from icarm._ale import icarm_ale
        return icarm_ale(self, X, feature=feature,
                         n_intervals=n_intervals)

    def fairness(self, X: pd.DataFrame, y,
                 protected: str) -> "IcarmFairness":
        """Audit group-level fairness."""
        from icarm._fairness import icarm_fairness
        return icarm_fairness(self, X, y, protected=protected)

    def learning_curve(self, X: pd.DataFrame, y, cv: int = 5,
                       sizes=None) -> "IcarmLearningCurve":
        """Compute learning curve."""
        from icarm._cv import icarm_learning_curve
        return icarm_learning_curve(self, X, y, cv=cv, sizes=sizes)

    def drift(self, X_train: pd.DataFrame,
              X_new: pd.DataFrame) -> "IcarmDrift":
        """Detect data drift between reference and new data."""
        from icarm._drift import icarm_drift
        return icarm_drift(self, X_train, X_new)

    def scorecard(self, analyst: str = "",
                  project: str = "") -> None:
        """Print the ICARM accountability scorecard."""
        from icarm._audit import icarm_scorecard
        icarm_scorecard(self, analyst=analyst, project=project)

    def audit(self, analyst: str = "",
              notes: str = "") -> str:
        """Return a JSON audit trail string."""
        from icarm._audit import icarm_audit
        return icarm_audit(self, analyst=analyst, notes=notes)

    # ── repr ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        fitted = self._fitted_model is not None
        icarm_flag = "YES" if self.model in _INTERPRETABLE else "NO"
        lines = [
            "=" * 60,
            "  icarm_model",
            f"  Task    : {self._task or 'not fitted'}",
            f"  Model   : {self.model}",
            f"  Formula : {self._formula or 'not fitted'}",
            f"  N train : {self._n_train}",
            f"  Features: {len(self._feature_names)}",
            f"  ICARM-compliant: {icarm_flag}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _check_fitted(self) -> None:
        if self._fitted_model is None:
            raise RuntimeError("Call .fit(X, y) before predicting.")


# ── sklearn model builder ─────────────────────────────────────────────
def _build_sklearn(model_name: str, task: str,
                   random_state: int | None) -> Any:
    """Return the appropriate scikit-learn estimator."""
    rs = random_state

    # ── Interpretable ──────────────────────────────────────────────
    if model_name == "cart":
        from sklearn.tree import (DecisionTreeClassifier,
                                   DecisionTreeRegressor)
        if task == "regression":
            return DecisionTreeRegressor(random_state=rs)
        return DecisionTreeClassifier(random_state=rs)

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=rs, max_iter=1000)

    if model_name == "logistic_l1":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(penalty="l1", solver="saga",
                                   random_state=rs, max_iter=1000)

    if model_name == "linear":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    if model_name == "ridge":
        from sklearn.linear_model import (Ridge, RidgeClassifier)
        if task == "regression":
            return Ridge()
        return RidgeClassifier(random_state=rs)

    if model_name == "elastic_net":
        from sklearn.linear_model import (ElasticNet,
                                           LogisticRegression)
        if task == "regression":
            return ElasticNet()
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                   l1_ratio=0.5, random_state=rs,
                                   max_iter=2000)

    if model_name == "gam":
        try:
            from pygam import LinearGAM, LogisticGAM
        except ImportError:
            raise ImportError(
                "Install pygam: pip install pygam"
            )
        if task == "regression":
            return LinearGAM()
        return LogisticGAM()

    if model_name == "multinomial":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(multi_class="multinomial",
                                   solver="lbfgs", random_state=rs,
                                   max_iter=1000)

    if model_name == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()

    if model_name == "lda":
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis)
        return LinearDiscriminantAnalysis()

    if model_name == "knn":
        from sklearn.neighbors import (KNeighborsClassifier,
                                        KNeighborsRegressor)
        if task == "regression":
            return KNeighborsRegressor(n_neighbors=5)
        return KNeighborsClassifier(n_neighbors=5)

    # ── Extended ───────────────────────────────────────────────────
    if model_name == "random_forest":
        from sklearn.ensemble import (RandomForestClassifier,
                                       RandomForestRegressor)
        if task == "regression":
            return RandomForestRegressor(n_estimators=100, random_state=rs)
        return RandomForestClassifier(n_estimators=100, random_state=rs)

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            raise ImportError("pip install xgboost")
        if task == "regression":
            return XGBRegressor(random_state=rs,
                                 eval_metric="rmse",
                                 verbosity=0)
        return XGBClassifier(random_state=rs,
                              use_label_encoder=False,
                              eval_metric="logloss",
                              verbosity=0)

    if model_name == "svm":
        from sklearn.svm import SVC, SVR
        if task == "regression":
            return SVR()
        return SVC(probability=True, random_state=rs)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError:
            raise ImportError("pip install lightgbm")
        if task == "regression":
            return LGBMRegressor(random_state=rs, verbosity=-1)
        return LGBMClassifier(random_state=rs, verbosity=-1)

    if model_name == "bagging":
        from sklearn.ensemble import (BaggingClassifier,
                                       BaggingRegressor)
        if task == "regression":
            return BaggingRegressor(n_estimators=50, random_state=rs)
        return BaggingClassifier(n_estimators=50, random_state=rs)

    if model_name == "nnet_reg":
        from sklearn.neural_network import (MLPClassifier,
                                             MLPRegressor)
        if task == "regression":
            return MLPRegressor(hidden_layer_sizes=(64, 32),
                                 random_state=rs, max_iter=500)
        return MLPClassifier(hidden_layer_sizes=(64, 32),
                              random_state=rs, max_iter=500)

    raise ValueError(f"Unknown model: '{model_name}'")


# ── Functional API ────────────────────────────────────────────────────
def icarm_fit(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model: str = "auto",
    task: str = "auto",
    positive: str | None = None,
    random_state: int | None = 2025,
) -> IcarmModel:
    """
    Fit an ICARM model (functional interface, mirrors R's icarm_fit).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or array-like
        Target variable.
    model : str
        Model type (default 'auto' -> CART).
    task : str
        'auto', 'binary', 'multiclass', or 'regression'.
    positive : str, optional
        Positive class label for binary classification.
    random_state : int, optional
        Random seed. Default 2025.

    Returns
    -------
    IcarmModel
        A fitted model object.

    Examples
    --------
    >>> m = icarm_fit(X_train, y_train, model="logistic",
    ...               positive="Yes")
    >>> m.metrics(X_test, y_test)
    """
    m = IcarmModel(model=model, task=task,
                   positive=positive, random_state=random_state)
    m.fit(X, y)
    return m


def icarm_split(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    prop: float = 0.75,
    stratify: bool = True,
    random_state: int | None = 2025,
) -> tuple[tuple[pd.DataFrame, np.ndarray],
           tuple[pd.DataFrame, np.ndarray]]:
    """
    Reproducible train/test split (mirrors R's icarm_split).

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series or array-like
    prop : float
        Proportion for training. Default 0.75.
    stratify : bool
        Stratify split on y for classification. Default True.
    random_state : int, optional
        Random seed. Default 2025.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)

    Examples
    --------
    >>> (X_tr, y_tr), (X_te, y_te) = icarm_split(X, y)
    """
    strat = y if stratify else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        train_size=prop,
        stratify=strat,
        random_state=random_state,
    )
    return (X_tr.reset_index(drop=True),
            np.asarray(y_tr)), \
           (X_te.reset_index(drop=True),
            np.asarray(y_te))
