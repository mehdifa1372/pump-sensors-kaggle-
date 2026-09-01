"""Causal feature engineering for timestamped sensor readings."""

from __future__ import annotations

import pandas as pd


def make_rolling_features(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "timestamp",
    sensor_columns: list[str] | None = None,
    window: int = 12,
) -> pd.DataFrame:
    """Return a sorted copy with trailing mean, standard deviation, and difference features."""
    if timestamp_column not in frame:
        raise ValueError(f"missing timestamp column: {timestamp_column}")
    if window < 2:
        raise ValueError("window must be at least 2")

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="raise", utc=True)
    result = result.sort_values(timestamp_column, kind="stable").reset_index(drop=True)

    if sensor_columns is None:
        sensor_columns = [
            column
            for column in result.select_dtypes(include="number").columns
            if column != timestamp_column
        ]
    missing = sorted(set(sensor_columns) - set(result.columns))
    if missing:
        raise ValueError(f"missing sensor columns: {', '.join(missing)}")
    non_numeric = [column for column in sensor_columns if not pd.api.types.is_numeric_dtype(result[column])]
    if non_numeric:
        raise ValueError(f"sensor columns must be numeric: {', '.join(non_numeric)}")

    for column in sensor_columns:
        trailing = result[column].rolling(window=window, min_periods=1)
        result[f"{column}__mean_{window}"] = trailing.mean()
        result[f"{column}__std_{window}"] = trailing.std(ddof=0)
        result[f"{column}__diff_1"] = result[column].diff()
    return result

