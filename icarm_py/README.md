# icarm

**Interpretable, Contextual-Accountable and Responsible Machine Learning for Python**

[![PyPI version](https://badge.fury.io/py/icarm.svg)](https://badge.fury.io/py/icarm)
[![CI](https://github.com/Olawaleawe/icarm-python/actions/workflows/ci.yml/badge.svg)](https://github.com/Olawaleawe/icarm-python/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Python port of the [icarm R package](https://CRAN.R-project.org/package=icarm).
Provides a unified pipeline for task-agnostic modelling, explanation,
fairness auditing, data drift detection, cross-validation, and
accountability reporting.

## Installation

```bash
pip install icarm

# With extended model support (XGBoost, LightGBM, GAM, SHAP)
pip install "icarm[extended]"
```

## Quick start

```python
from icarm import IcarmModel, icarm_split, load_medical

# Load built-in dataset
df = load_medical()
X  = df.drop(columns=["readmitted"])
y  = df["readmitted"]

# Split
(X_tr, y_tr), (X_te, y_te) = icarm_split(X, y, prop=0.75)

# Fit (task auto-detected: binary)
model = IcarmModel(model="logistic", positive="Yes")
model.fit(X_tr, y_tr)
print(model)

# Performance
model.metrics(X_te, y_te)

# Global explanation
ex = model.explain()
ex.plot()

# SHAP values
shap = model.shap(X_te, n_samples=50)
shap.plot()

# ALE profile
ale = model.ale(X_te, feature="num_prior_visits")
ale.plot()

# Fairness audit
f = model.fairness(X_te, y_te, protected="gender")
f.plot(metric="dp_ratio", ref_line=0.8)
f.summary()

# Accountability
model.scorecard(analyst="O. O. Awe",
                project="Hospital Readmission Study")
trail = model.audit(analyst="O. O. Awe")
```

## Functional API (mirrors R)

```python
from icarm import (
    icarm_fit, icarm_split, icarm_metrics,
    icarm_explain, icarm_shap, icarm_ale,
    icarm_fairness, icarm_equity_summary,
    icarm_calibrate, icarm_thresholds,
    icarm_drift, icarm_cv, icarm_compare,
    icarm_audit, icarm_scorecard,
)

m = icarm_fit(X_tr, y_tr, model="logistic", positive="Yes")
icarm_metrics(y_te, m.predict(X_te),
              y_prob=m.predict(X_te, type="prob")["Yes"].values,
              positive="Yes")
```

## Supported models (18 total)

| Family | Model code |
|---|---|
| Interpretable | `cart`, `logistic`, `logistic_l1`, `linear`, `ridge`, `elastic_net`, `gam`, `multinomial`, `naive_bayes`, `lda`, `knn` |
| Extended | `random_forest`, `xgboost`, `svm`, `lightgbm`, `bagging`, `nnet_reg` |
| Custom | `custom` |

## R package

The companion R package is available on CRAN:

```r
install.packages("icarm")
```

## Author

**Olushina Olawale Awe**
Alexander von Humboldt Visiting Professor, PH Ludwigsburg, Germany
ORCID: 0000-0002-0442-4519

## Citation

```bibtex
@software{Awe_icarm_python_2026,
  author  = {Awe, Olushina Olawale},
  title   = {{icarm}: Interpretable, Contextual-Accountable and
             Responsible Machine Learning for Python},
  year    = {2026},
  url     = {https://github.com/Olawaleawe/icarm-python},
  version = {0.1.0}
}
```

## License

MIT
