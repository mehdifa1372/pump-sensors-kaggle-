import pandas as pd
import pytest

from pump_monitoring import build_baseline, temporal_split


def test_temporal_split_preserves_order():
    frame = pd.DataFrame({"value": range(10)})
    train, test = temporal_split(frame, test_fraction=0.2)
    assert train["value"].tolist() == list(range(8))
    assert test["value"].tolist() == [8, 9]


def test_temporal_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        temporal_split(pd.DataFrame({"value": [1, 2]}), test_fraction=1.0)


def test_baseline_is_reproducible():
    first = build_baseline(random_state=7)
    second = build_baseline(random_state=7)
    assert first.get_params()["classifier__random_state"] == second.get_params()["classifier__random_state"]

