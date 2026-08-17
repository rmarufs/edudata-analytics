# Automated Tests

This folder contains regression tests for the main non-UI parts of the project.
They are meant to catch common breakages after code changes without manually
clicking through the whole Streamlit app.

Run from the repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/edudata_pycache python3 -m unittest discover -s tests
```

The tests cover:

- CSV/Excel loading helpers and type conversion rules.
- Missing-value cleaning helpers.
- Logistic regression preparation, training, metrics, and single prediction.
- CatBoost task detection and feature-importance figure outputs.
- The local `gmdh` package.
- Syntax compilation for all tracked Python files.
- Basic README consistency checks.

These tests do not replace full browser/UI testing. They cover the project logic
that sits behind the Streamlit interface.
