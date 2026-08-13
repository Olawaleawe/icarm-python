"""
icarm: Interpretable, Contextual-Accountable and Responsible
       Machine Learning for Python.

Mirrors the R icarm package (CRAN) with a consistent API for
task-agnostic modelling, explanation, fairness auditing,
drift detection, cross-validation, and accountability reporting.

Quick start
-----------
>>> from icarm import IcarmModel
>>> model = IcarmModel(model="logistic", positive="Yes")
>>> model.fit(X_train, y_train)
>>> model.metrics(X_test, y_test)
>>> model.explain().plot()
>>> model.fairness(X_test, y_test, protected="gender").summary()
>>> model.scorecard(analyst="O. O. Awe")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("icarm")
except PackageNotFoundError:
    __version__ = "dev"

__author__ = "Olushina Olawale Awe"
__email__  = "olawaleawe@gmail.com"

# -- Core object -------------------------------------------------------
from icarm._fit import IcarmModel

# -- Functional API (mirrors R's icarm_*() naming) ---------------------
from icarm._fit      import icarm_fit, icarm_split
from icarm._metrics  import icarm_metrics
from icarm._explain  import icarm_explain, icarm_explain_local
from icarm._shap     import icarm_shap
from icarm._pdp      import icarm_pdp
from icarm._ale      import icarm_ale
from icarm._fairness import icarm_fairness, icarm_equity_summary
from icarm._calibrate import icarm_calibrate, icarm_thresholds
from icarm._drift    import icarm_drift
from icarm._cv       import icarm_cv
from icarm._compare  import icarm_compare
from icarm._audit    import icarm_audit, icarm_scorecard
from icarm._datasets import load_medical, load_financial, load_racism_survey

__all__ = [
    # Class
    "IcarmModel",
    # Functional
    "icarm_fit", "icarm_split", "icarm_metrics",
    "icarm_explain", "icarm_explain_local",
    "icarm_shap", "icarm_pdp", "icarm_ale",
    "icarm_fairness", "icarm_equity_summary",
    "icarm_calibrate", "icarm_thresholds",
    "icarm_drift", "icarm_cv", "icarm_compare",
    "icarm_audit", "icarm_scorecard",
    "load_medical", "load_financial", "load_racism_survey",
]
