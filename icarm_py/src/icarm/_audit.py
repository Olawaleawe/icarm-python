"""Accountability — JSON audit trail and scorecard. Mirrors icarm_audit / icarm_scorecard."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import TYPE_CHECKING
import json
import icarm
from icarm._utils import to_json
if TYPE_CHECKING:
    from icarm._fit import IcarmModel


def icarm_audit(
    model: "IcarmModel",
    analyst: str = "",
    notes: str = "",
) -> str:
    """
    Generate a JSON audit trail with SHA-256 data provenance.

    Satisfies Article 12 of the EU AI Act on event logging for
    high-risk AI systems.

    Parameters
    ----------
    model : IcarmModel
    analyst : str
    notes : str

    Returns
    -------
    str  JSON string.

    Examples
    --------
    >>> trail = icarm_audit(model, analyst="O. O. Awe")
    >>> print(trail)
    >>> with open("audit.json", "w") as f:
    ...     f.write(trail)
    """
    model._check_fitted()
    icarm_flag = model.model in {
        "cart", "logistic", "logistic_l1", "linear",
        "ridge", "elastic_net", "gam", "multinomial",
        "naive_bayes", "lda", "knn",
    }
    record = {
        "icarm_version"   : icarm.__version__,
        "timestamp"       : datetime.now(timezone.utc).isoformat(),
        "analyst"         : analyst,
        "task"            : model._task,
        "model"           : model.model,
        "formula"         : model._formula,
        "n_train"         : model._n_train,
        "n_features"      : len(model._feature_names),
        "feature_names"   : model._feature_names,
        "data_hash"       : model._data_hash,
        "trained_at"      : model._trained_at,
        "icarm_compliant" : icarm_flag,
        "notes"           : notes,
    }
    return json.dumps(record, indent=2)


def icarm_scorecard(
    model: "IcarmModel",
    analyst: str = "",
    project: str = "",
) -> None:
    """
    Print an ICARM accountability scorecard.

    Satisfies Article 11 of the EU AI Act on technical documentation
    for high-risk AI systems.

    Parameters
    ----------
    model : IcarmModel
    analyst : str
    project : str

    Examples
    --------
    >>> icarm_scorecard(model, analyst="O. O. Awe",
    ...                  project="Hospital Readmission Study")
    """
    model._check_fitted()
    sep  = "=" * 60
    sep2 = "-" * 60
    flag = "YES" if model.model in {
        "cart", "logistic", "logistic_l1", "linear",
        "ridge", "elastic_net", "gam", "multinomial",
        "naive_bayes", "lda", "knn",
    } else "NO"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(sep)
    print("  ICARM ACCOUNTABILITY SCORECARD")
    print(sep)
    if project:
        print(f"  Project  : {project}")
    if analyst:
        print(f"  Analyst  : {analyst}")
    print(f"  Generated: {ts}")
    print(sep2)
    print(f"  [Model]")
    print(f"    Task    : {model._task}")
    print(f"    Model   : {model.model}")
    print(f"    Formula : {model._formula}")
    print(f"    N train : {model._n_train}")
    print(f"    ICARM-compliant: {flag}")
    print(sep2)
    print(f"  [Provenance]")
    print(f"    Data hash   : {model._data_hash[:20]}...")
    print(f"    Trained at  : {model._trained_at}")
    print(sep)
