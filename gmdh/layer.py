import itertools
import numpy as np

from .neuron import Neuron


class Layer:
    """
    Один слой МГУА: строит нейроны на всех (или ограниченном подмножестве)
    парах входных признаков, обучает на train, оценивает на val,
    оставляет top_n лучших по MSE.
    """

    def __init__(self, top_n: int, max_pairs: int | None = None, random_state: int | None = None):
        self.top_n = top_n
        self.max_pairs = max_pairs      # защита от комбинаторного взрыва пар
        self.random_state = random_state
        self.neurons_: list[Neuron] = []
        self.best_score_: float | None = None

    def _candidate_pairs(self, n_features: int) -> list[tuple[int, int]]:
        pairs = list(itertools.combinations(range(n_features), 2))
        if self.max_pairs is not None and len(pairs) > self.max_pairs:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(pairs), size=self.max_pairs, replace=False)
            pairs = [pairs[k] for k in idx]
        return pairs

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "Layer":
        n_features = X_train.shape[1]
        if n_features < 2:
            raise ValueError(f"Слою нужно минимум 2 входных признака, получено {n_features}")

        candidates = []
        for i, j in self._candidate_pairs(n_features):
            neuron = Neuron(i, j)
            neuron.fit(X_train, y_train)
            neuron.evaluate(X_val, y_val)
            candidates.append(neuron)

        candidates.sort(key=lambda n: n.val_score_)  # меньше MSE — лучше
        self.neurons_ = candidates[: self.top_n]
        self.best_score_ = self.neurons_[0].val_score_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Прогоняет X через отобранные нейроны слоя -> вход для следующего слоя."""
        return np.column_stack([n.predict(X) for n in self.neurons_])

    def to_dict(self) -> dict:
        return {
            "top_n": self.top_n,
            "best_score": self.best_score_,
            "neurons": [n.to_dict() for n in self.neurons_],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Layer":
        layer = cls(top_n=data["top_n"])
        layer.neurons_ = [Neuron.from_dict(nd) for nd in data["neurons"]]
        layer.best_score_ = data.get("best_score")
        return layer