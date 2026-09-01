# Contributing

Protect temporal ordering and document every assumption about timestamps, failure labels, sensor sampling, maintenance periods, and missing data. Tests should demonstrate that feature generation never accesses future rows.

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

Do not commit proprietary sensor data, credentials, model artifacts, or reports containing operationally sensitive information.

