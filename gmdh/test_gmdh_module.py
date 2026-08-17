"""
Standalone smoke test for the local GMDH/MGUA package.

Run from this directory:
    python3 test_gmdh_module.py
"""

from pathlib import Path
import sys

import numpy as np


PACKAGE_PARENT = Path(__file__).resolve().parent.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from gmdh import GMDHRegressor, compute_all, mae, mse, r2, rmse  # noqa: E402


def make_dataset(n_samples: int = 220, random_state: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, 5))
    noise = rng.normal(scale=0.03, size=n_samples)

    y = (
        1.25
        + 2.0 * X[:, 0]
        - 1.4 * X[:, 1]
        + 0.9 * X[:, 0] * X[:, 1]
        + 0.55 * X[:, 2] ** 2
        - 0.35 * X[:, 3] * X[:, 4]
        + noise
    )
    return X, y


def assert_close(name: str, actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{name}: expected {expected:.12g}, got {actual:.12g}")


def main() -> None:
    X, y = make_dataset()

    model = GMDHRegressor(
        max_layers=5,
        top_n=8,
        val_split=0.35,
        max_pairs_per_layer=100,
        random_state=42,
    )
    model.fit(X, y)

    predictions = model.predict(X)
    metrics = model.get_metrics(X, y)
    importances = model.get_feature_importance(X, y, random_state=42)
    formula = model.get_formula(["x0", "x1", "x2", "x3", "x4"])
    structure = model.get_structure()

    restored = GMDHRegressor.from_json(model.to_json())
    restored_predictions = restored.predict(X)

    if predictions.shape != y.shape:
        raise AssertionError(f"predict() returned shape {predictions.shape}, expected {y.shape}")
    if not model.is_fitted_:
        raise AssertionError("fit() did not mark the model as fitted")
    if len(model.layers_) == 0:
        raise AssertionError("fit() did not build any GMDH layers")
    if len(structure) != len(model.layers_):
        raise AssertionError("get_structure() does not match the number of fitted layers")
    if not formula or "x0" not in formula:
        raise AssertionError("get_formula() did not return a readable formula over feature names")
    if metrics["r2"] < 0.90:
        raise AssertionError(f"Model quality is unexpectedly low: R2={metrics['r2']:.4f}")
    if metrics["rmse"] > 1.0:
        raise AssertionError(f"Model error is unexpectedly high: RMSE={metrics['rmse']:.4f}")
    if set(importances) != set(range(X.shape[1])):
        raise AssertionError("Feature importance keys do not cover all input columns")
    if max(importances, key=importances.get) not in {0, 1}:
        raise AssertionError(f"Unexpected most important feature: {importances}")

    max_prediction_diff = float(np.max(np.abs(predictions - restored_predictions)))
    assert_close("JSON round-trip prediction difference", max_prediction_diff, 0.0)

    direct_metrics = compute_all(y, predictions)
    assert_close("mse()", mse(y, predictions), direct_metrics["mse"])
    assert_close("rmse()", rmse(y, predictions), direct_metrics["rmse"])
    assert_close("mae()", mae(y, predictions), direct_metrics["mae"])
    assert_close("r2()", r2(y, predictions), direct_metrics["r2"])

    print("GMDH module smoke test passed")
    print(f"Layers built: {len(model.layers_)}")
    print(
        "Metrics: "
        f"MSE={metrics['mse']:.6f}, "
        f"RMSE={metrics['rmse']:.6f}, "
        f"MAE={metrics['mae']:.6f}, "
        f"R2={metrics['r2']:.6f}"
    )
    print(f"Top feature by permutation importance: x{max(importances, key=importances.get)}")
    print(f"JSON round-trip max prediction difference: {max_prediction_diff:.3g}")


if __name__ == "__main__":
    main()
