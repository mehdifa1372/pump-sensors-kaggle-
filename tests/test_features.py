import pandas as pd
import pytest

from pump_monitoring import make_rolling_features


def test_features_sort_time_and_use_trailing_window_only():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-03", "2026-01-01", "2026-01-02"],
            "pressure": [30.0, 10.0, 20.0],
        }
    )
    result = make_rolling_features(frame, sensor_columns=["pressure"], window=2)
    assert result["pressure"].tolist() == [10.0, 20.0, 30.0]
    assert result["pressure__mean_2"].tolist() == [10.0, 15.0, 25.0]
    assert pd.isna(result.loc[0, "pressure__diff_1"])
    assert result.loc[1, "pressure__diff_1"] == 10.0


def test_features_reject_missing_timestamp():
    with pytest.raises(ValueError, match="missing timestamp"):
        make_rolling_features(pd.DataFrame({"sensor": [1.0, 2.0]}))

