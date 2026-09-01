"""Leakage-aware splitting and a dependable tabular baseline."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def temporal_split(frame: pd.DataFrame, *, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an already time-ordered frame without shuffling."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("at least two rows are required")
    split_index = max(1, min(len(frame) - 1, int(len(frame) * (1.0 - test_fraction))))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def build_baseline(*, random_state: int = 42) -> Pipeline:
    """Build a missing-value-safe gradient-boosting classification baseline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=200,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

