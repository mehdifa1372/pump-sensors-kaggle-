# Pump Sensor Predictive Maintenance

A reproducible starter project for detecting abnormal pump behavior from multivariate time-series sensor readings. The repository focuses on leakage-safe temporal splitting, causal rolling features, and an interpretable baseline workflow before more complex sequence models are considered.

## Project goals

- Convert timestamped sensor readings into model-ready features.
- Generate trailing rolling statistics without using future observations.
- Preserve chronological order during train/test splitting.
- Establish a scikit-learn baseline with explicit missing-value handling.
- Provide tests and CI before adding dataset-specific experimentation.

## Repository structure

```text
.
├── src/pump_monitoring/
│   ├── features.py       # validation and causal rolling features
│   └── modeling.py       # temporal split and baseline estimator
├── tests/
├── pyproject.toml
└── .github/workflows/quality.yml
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

## Example

```python
import pandas as pd
from pump_monitoring import build_baseline, make_rolling_features, temporal_split

frame = pd.read_csv("data/sensor.csv")
features = make_rolling_features(frame, timestamp_column="timestamp", window=12)
train, test = temporal_split(features, test_fraction=0.2)

sensor_columns = [column for column in features if column != "timestamp"]
model = build_baseline(random_state=42)
model.fit(train[sensor_columns], train["machine_status"])
predictions = model.predict(test[sensor_columns])
```

Adapt the target and feature columns to the selected dataset. The example is intentionally schematic because no dataset is committed or assumed.

## Evaluation design

Random row splitting is usually inappropriate for sensor streams because nearby observations are correlated and future information can leak into training. Use chronological splits and consider evaluation across separate failure episodes.

Report at least:

- Class counts and a naive baseline.
- Precision, recall, F1, PR-AUC, and false-alarm rate.
- Detection lead time before a failure event.
- Results by failure episode rather than only by row.
- The cost assumptions behind the selected decision threshold.

No performance result is claimed until it is reproduced from a documented dataset and split.

## Roadmap

- Add a dataset adapter with schema documentation.
- Build episode-aware cross-validation.
- Compare robust statistical, tree-based, isolation, and sequence baselines.
- Add calibration and threshold-cost analysis.
- Add an explainability report and a small monitoring dashboard.

## Responsible use

This is an educational baseline, not a certified industrial monitoring system. Real maintenance decisions require domain review, calibrated alerts, hardware context, drift monitoring, and fail-safe operational procedures.

## Author

Mehdi Faraz — computer vision, machine learning, data science, and applied AI.

