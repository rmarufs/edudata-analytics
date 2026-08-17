import numpy as np


class Neuron:
    """
    Один нейрон МГУА (Ивахненко): квадратичный полином от пары признаков.
    y = a0 + a1*x1 + a2*x2 + a3*x1*x2 + a4*x1^2 + a5*x2^2
    """

    def __init__(self, i: int, j: int):
        self.i = i                # индекс первого признака во входной матрице слоя
        self.j = j                # индекс второго признака
        self.coef_ = None         # [a0, a1, a2, a3, a4, a5] после fit
        self.val_score_ = None    # MSE на валидации — по нему Layer отбирает нейронов

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        x1 = X[:, self.i]
        x2 = X[:, self.j]
        return np.column_stack([
            np.ones_like(x1),
            x1,
            x2,
            x1 * x2,
            x1 ** 2,
            x2 ** 2,
        ])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Neuron":
        A = self._design_matrix(X_train)
        # lstsq устойчивее прямого решения через (A^T A)^-1 A^T y —
        # не падает, если x1 и x2 сильно коррелируют
        coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        A = self._design_matrix(X)
        return A @ self.coef_

    def evaluate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        pred = self.predict(X_val)
        mse = float(np.mean((y_val - pred) ** 2))
        self.val_score_ = mse
        return mse

    def to_dict(self) -> dict:
        return {
            "i": self.i,
            "j": self.j,
            "coef": self.coef_.tolist() if self.coef_ is not None else None,
            "val_score": self.val_score_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Neuron":
        neuron = cls(data["i"], data["j"])
        neuron.coef_ = np.array(data["coef"]) if data["coef"] is not None else None
        neuron.val_score_ = data.get("val_score")
        return neuron