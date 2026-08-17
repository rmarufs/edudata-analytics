import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        # все y_true одинаковые — R^2 классически не определён
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


METRICS = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {name: fn(y_true, y_pred) for name, fn in METRICS.items()}