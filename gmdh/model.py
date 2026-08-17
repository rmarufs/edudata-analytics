import json
import numpy as np

from .layer import Layer
from .criteria import compute_all, mse


class GMDHRegressor:
    """
    МГУА-регрессор (GMDH, Ивахненко). Растит слои квадратичных нейронов
    на парах признаков, отбирая лучшие по MSE на валидации, пока качество растёт.
    """

    def __init__(
        self,
        max_layers: int = 8,
        top_n: int = 10,
        val_split: float = 0.5,
        max_pairs_per_layer: int | None = 300,
        random_state: int | None = None,
    ):
        self.max_layers = max_layers
        self.top_n = top_n
        self.val_split = val_split
        self.max_pairs_per_layer = max_pairs_per_layer
        self.random_state = random_state

        self.layers_: list[Layer] = []
        self.winner_index_: int | None = None
        self.n_features_in_: int | None = None
        self.is_fitted_ = False

    # ---------- обучение ----------

    def _split(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(X.shape[0])
        n_val = int(X.shape[0] * self.val_split)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMDHRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_in_ = X.shape[1]

        X_train, y_train, X_val, y_val = self._split(X, y)

        self.layers_ = []
        best_score = np.inf
        cur_train, cur_val = X_train, X_val

        for _ in range(self.max_layers):
            if cur_train.shape[1] < 2:
                break

            layer = Layer(top_n=self.top_n, max_pairs=self.max_pairs_per_layer,
                          random_state=self.random_state)
            layer.fit(cur_train, y_train, cur_val, y_val)

            if layer.best_score_ >= best_score:
                break  # новый слой не улучшил результат — стоп, предыдущий последний

            best_score = layer.best_score_
            self.layers_.append(layer)
            cur_train = layer.transform(cur_train)
            cur_val = layer.transform(cur_val)

        if not self.layers_:
            raise RuntimeError(
                "Не удалось построить ни одного слоя — проверь данные "
                "(нужно минимум 2 признака и достаточно строк)"
            )

        last_layer = self.layers_[-1]
        self.winner_index_ = int(np.argmin([n.val_score_ for n in last_layer.neurons_]))
        self.is_fitted_ = True
        return self

    # ---------- инференс ----------

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Модель не обучена — сначала вызови fit()")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        cur = X
        for layer in self.layers_[:-1]:
            cur = layer.transform(cur)
        winner = self.layers_[-1].neurons_[self.winner_index_]
        return winner.predict(cur)

    # ---------- метрики и анализ ----------

    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        y = np.asarray(y, dtype=float)
        return compute_all(y, self.predict(X))

    def get_feature_importance(self, X: np.ndarray, y: np.ndarray,
                                random_state: int | None = None) -> dict:
        """
        Permutation importance по исходным признакам: перемешиваем колонку,
        смотрим насколько вырос MSE. Работает поверх predict(), поэтому
        не важно, сколько слоёв внутри и как признак туда попал.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(random_state)

        baseline = mse(y, self.predict(X))
        importances = {}
        for col in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, col] = rng.permutation(X_perm[:, col])
            importances[col] = mse(y, self.predict(X_perm)) - baseline
        return importances

    def get_structure(self) -> list:
        return [layer.to_dict() for layer in self.layers_]

    def get_formula(self, feature_names: list[str] | None = None) -> str:
        """Разворачивает победившую цепочку нейронов подстановкой (без упрощения)."""
        self._check_fitted()
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        def build(layer_idx: int, neuron_idx: int) -> str:
            neuron = self.layers_[layer_idx].neurons_[neuron_idx]
            a0, a1, a2, a3, a4, a5 = neuron.coef_
            if layer_idx == 0:
                t1, t2 = feature_names[neuron.i], feature_names[neuron.j]
            else:
                t1 = build(layer_idx - 1, neuron.i)
                t2 = build(layer_idx - 1, neuron.j)
            return (f"({a0:.4f} + {a1:.4f}*{t1} + {a2:.4f}*{t2} "
                    f"+ {a3:.4f}*{t1}*{t2} + {a4:.4f}*{t1}^2 + {a5:.4f}*{t2}^2)")

        return build(len(self.layers_) - 1, self.winner_index_)

    # ---------- сериализация (JSON) ----------

    def to_dict(self) -> dict:
        return {
            "max_layers": self.max_layers,
            "top_n": self.top_n,
            "val_split": self.val_split,
            "max_pairs_per_layer": self.max_pairs_per_layer,
            "n_features_in": self.n_features_in_,
            "winner_index": self.winner_index_,
            "layers": self.get_structure(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "GMDHRegressor":
        model = cls(
            max_layers=data["max_layers"],
            top_n=data["top_n"],
            val_split=data["val_split"],
            max_pairs_per_layer=data.get("max_pairs_per_layer"),
        )
        model.n_features_in_ = data["n_features_in"]
        model.winner_index_ = data["winner_index"]
        model.layers_ = [Layer.from_dict(ld) for ld in data["layers"]]
        model.is_fitted_ = True
        return model

    @classmethod
    def from_json(cls, payload: str) -> "GMDHRegressor":
        return cls.from_dict(json.loads(payload))