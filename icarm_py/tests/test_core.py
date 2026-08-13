"""Core unit tests for the icarm Python package."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris

from icarm import (
    IcarmModel, icarm_fit, icarm_split,
    icarm_metrics, icarm_explain, icarm_fairness,
    icarm_equity_summary, icarm_calibrate, icarm_thresholds,
    icarm_drift, icarm_audit, icarm_scorecard,
    load_medical, load_financial, load_racism_survey,
)


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def binary_data():
    bc = load_breast_cancer(as_frame=True)
    X  = bc.data
    y  = pd.Series(
        np.where(bc.target == 1, "benign", "malignant"),
        name="diagnosis")
    return X, y

@pytest.fixture
def regression_data():
    from sklearn.datasets import load_diabetes
    ds = load_diabetes(as_frame=True)
    return ds.data, ds.target

@pytest.fixture
def multiclass_data():
    iris = load_iris(as_frame=True)
    return iris.data, pd.Series(
        load_iris().target_names[iris.target], name="species")

@pytest.fixture
def fitted_binary(binary_data):
    X, y = binary_data
    (X_tr, y_tr), _ = icarm_split(X, y)
    return icarm_fit(X_tr, y_tr, model="cart", positive="benign")


# ── Task detection ────────────────────────────────────────────────────
def test_task_auto_binary(binary_data):
    X, y = binary_data
    m = IcarmModel(model="cart", positive="benign")
    m.fit(X, y)
    assert m._task == "binary"

def test_task_auto_regression(regression_data):
    X, y = regression_data
    m = IcarmModel(model="linear")
    m.fit(X, y)
    assert m._task == "regression"

def test_task_auto_multiclass(multiclass_data):
    X, y = multiclass_data
    m = IcarmModel(model="cart")
    m.fit(X, y)
    assert m._task == "multiclass"


# ── icarm_split ───────────────────────────────────────────────────────
def test_split_sizes(binary_data):
    X, y = binary_data
    (X_tr, y_tr), (X_te, y_te) = icarm_split(X, y, prop=0.75)
    assert len(X_tr) + len(X_te) == len(X)
    assert abs(len(X_tr) / len(X) - 0.75) < 0.02

def test_split_reproducible(binary_data):
    X, y = binary_data
    (X1, _), _ = icarm_split(X, y, random_state=42)
    (X2, _), _ = icarm_split(X, y, random_state=42)
    pd.testing.assert_frame_equal(X1, X2)


# ── icarm_fit / predict ───────────────────────────────────────────────
@pytest.mark.parametrize("model_name", [
    "cart", "logistic", "logistic_l1", "naive_bayes", "lda", "knn",
])
def test_binary_models(binary_data, model_name):
    X, y = binary_data
    (X_tr, y_tr), (X_te, y_te) = icarm_split(X, y)
    m = icarm_fit(X_tr, y_tr, model=model_name, positive="benign")
    y_hat = m.predict(X_te)
    assert len(y_hat) == len(X_te)
    proba = m.predict(X_te, type="prob")
    assert proba.shape == (len(X_te), 2)

@pytest.mark.parametrize("model_name", [
    "cart", "linear", "knn", "ridge", "elastic_net",
])
def test_regression_models(regression_data, model_name):
    X, y = regression_data
    (X_tr, y_tr), (X_te, y_te) = icarm_split(X, y, stratify=False)
    m = icarm_fit(X_tr, y_tr, model=model_name)
    y_hat = m.predict(X_te)
    assert len(y_hat) == len(X_te)
    assert np.isfinite(y_hat).all()

def test_predict_before_fit_raises():
    m = IcarmModel(model="logistic")
    X = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(RuntimeError):
        m.predict(X)

def test_invalid_model_raises(binary_data):
    X, y = binary_data
    with pytest.raises(ValueError):
        icarm_fit(X, y, model="bad_model")


# ── icarm_metrics ─────────────────────────────────────────────────────
def test_regression_metrics(regression_data):
    X, y = regression_data
    (X_tr, y_tr), (X_te, y_te) = icarm_split(X, y, stratify=False)
    m   = icarm_fit(X_tr, y_tr, model="linear")
    met = icarm_metrics(y_te, m.predict(X_te), task="regression")
    assert "mae" in met and "rmse" in met and "r2" in met
    assert met["mae"] >= 0
    assert met["rmse"] >= met["mae"]

def test_binary_metrics(fitted_binary, binary_data):
    X, y = binary_data
    _, (X_te, y_te) = icarm_split(X, y)
    y_hat = fitted_binary.predict(X_te)
    met   = icarm_metrics(y_te, y_hat, positive="benign",
                           task="binary")
    assert "accuracy" in met
    assert 0 <= met["accuracy"] <= 1


# ── icarm_explain ─────────────────────────────────────────────────────
def test_explain_returns_dataframe(fitted_binary):
    ex = icarm_explain(fitted_binary)
    assert isinstance(ex.importance, pd.DataFrame)
    assert "feature" in ex.importance.columns
    assert len(ex.importance) > 0
    assert (ex.importance["importance_scaled"] <= 1.0).all()


# ── icarm_fairness ────────────────────────────────────────────────────
def test_fairness_binary(fitted_binary, binary_data):
    X, y = binary_data
    _, (X_te, y_te) = icarm_split(X, y)
    X_te = X_te.copy()
    X_te["gender"] = np.random.choice(["M", "F"], len(X_te))
    f  = icarm_fairness(fitted_binary, X_te, y_te,
                         protected="gender")
    eq = icarm_equity_summary(f)
    assert "n_groups" in eq
    assert eq["n_groups"] == 2

def test_fairness_regression(regression_data):
    X, y = regression_data
    (X_tr, y_tr), (X_te, y_te) = icarm_split(X, y, stratify=False)
    m = icarm_fit(X_tr, y_tr, model="linear")
    X_te = X_te.copy()
    X_te["region"] = np.random.choice(["A", "B", "C"], len(X_te))
    f  = icarm_fairness(m, X_te, y_te, protected="region")
    assert "mae_gap" in f.metrics.columns


# ── icarm_calibrate / icarm_thresholds ───────────────────────────────
def test_calibrate(fitted_binary, binary_data):
    X, y = binary_data
    _, (X_te, y_te) = icarm_split(X, y)
    cal = icarm_calibrate(fitted_binary, X_te, y_te,
                           positive="benign")
    assert 0 <= cal.brier <= 1
    assert 0 <= cal.ece <= 1

def test_thresholds(fitted_binary, binary_data):
    X, y = binary_data
    _, (X_te, y_te) = icarm_split(X, y)
    thr = icarm_thresholds(fitted_binary, X_te, y_te,
                            positive="benign")
    assert isinstance(thr, pd.DataFrame)
    assert "threshold" in thr.columns
    assert len(thr) > 0


# ── icarm_drift ───────────────────────────────────────────────────────
def test_drift(fitted_binary, binary_data):
    X, y = binary_data
    (X_tr, _), (X_te, _) = icarm_split(X, y)
    drift = icarm_drift(fitted_binary, X_tr, X_te)
    assert isinstance(drift.drift, pd.DataFrame)
    assert "psi_flag" in drift.drift.columns
    assert set(drift.drift["psi_flag"]).issubset(
        {"none", "moderate", "high"})


# ── Audit and scorecard ───────────────────────────────────────────────
def test_audit_json(fitted_binary):
    import json
    trail = icarm_audit(fitted_binary, analyst="Test")
    record = json.loads(trail)
    assert "timestamp" in record
    assert "data_hash" in record
    assert record["model"] == "cart"
    assert isinstance(record["icarm_compliant"], bool)

def test_scorecard_runs(fitted_binary, capsys):
    icarm_scorecard(fitted_binary,
                     analyst="Test", project="Test Project")
    captured = capsys.readouterr()
    assert "ICARM ACCOUNTABILITY SCORECARD" in captured.out


# ── Built-in datasets ─────────────────────────────────────────────────
def test_load_medical():
    df = load_medical()
    assert len(df) == 500
    assert "readmitted" in df.columns

def test_load_financial():
    df = load_financial()
    assert len(df) == 1000
    assert "default" in df.columns

def test_load_racism_survey():
    df = load_racism_survey()
    assert len(df) == 150
    assert "racism_impact" in df.columns
