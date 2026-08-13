"""Internal utilities shared across icarm modules."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


# ── Task detection ────────────────────────────────────────────────────
def detect_task(y: pd.Series | np.ndarray) -> str:
    """
    Detect prediction task from target variable.

    Rules (mirrors R icarm):
    - numeric/float  -> "regression"
    - 2 unique values -> "binary"
    - 3+ unique values -> "multiclass"
    """
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating) or (
        np.issubdtype(y.dtype, np.integer)
        and len(np.unique(y)) > 10
    ):
        return "regression"
    n_unique = len(np.unique(y))
    if n_unique == 2:
        return "binary"
    return "multiclass"


# ── SHA-256 data hash ─────────────────────────────────────────────────
def data_hash(X: pd.DataFrame | np.ndarray) -> str:
    """Compute a SHA-256 hex digest of a dataset for provenance."""
    if isinstance(X, pd.DataFrame):
        raw = X.to_csv(index=False).encode()
    else:
        raw = np.ascontiguousarray(X).tobytes()
    return hashlib.sha256(raw).hexdigest()


# ── Prediction helpers ────────────────────────────────────────────────
def predict_proba_binary(model: Any, X: pd.DataFrame,
                         positive: str | None,
                         classes: list[str]) -> np.ndarray:
    """Return probability of the positive class."""
    proba = model.predict_proba(X)
    if positive is not None and positive in classes:
        pos_idx = list(classes).index(positive)
    else:
        pos_idx = 1
    return proba[:, pos_idx]


# ── Safe JSON serialisation ───────────────────────────────────────────
def to_json(obj: Any) -> str:
    """Serialise obj to a JSON string, converting numpy scalars."""
    def _convert(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        return o
    return json.dumps(obj, default=_convert, indent=2)


# ── Colour palette (matches R icarm) ─────────────────────────────────
PALETTE = {
    "primary"   : "#1A3A6B",
    "secondary" : "#22C9A8",
    "accent"    : "#F0A500",
    "fair"      : "#22C9A8",
    "unfair"    : "#E05B4B",
    "neutral"   : "#94A3B8",
}


# ── Plot theme helper ─────────────────────────────────────────────────
def icarm_theme(ax: Any) -> None:
    """Apply the icarm plot theme to a matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_facecolor("#F8FAFC")
    ax.figure.set_facecolor("white")
