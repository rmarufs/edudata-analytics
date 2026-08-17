import unittest

import numpy as np

from gmdh import GMDHRegressor, compute_all


class GMDHTests(unittest.TestCase):
    def test_gmdh_regressor_fit_predict_metrics_and_json_round_trip(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(180, 5))
        y = (
            1.0
            + 1.8 * X[:, 0]
            - 1.2 * X[:, 1]
            + 0.7 * X[:, 0] * X[:, 1]
            + 0.4 * X[:, 2] ** 2
        )

        model = GMDHRegressor(max_layers=4, top_n=8, val_split=0.35, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)
        metrics = model.get_metrics(X, y)
        formula = model.get_formula(["x0", "x1", "x2", "x3", "x4"])
        restored = GMDHRegressor.from_json(model.to_json())

        self.assertEqual(predictions.shape, y.shape)
        self.assertGreaterEqual(len(model.layers_), 1)
        self.assertGreater(metrics["r2"], 0.9)
        self.assertIn("x0", formula)
        np.testing.assert_allclose(predictions, restored.predict(X))
        self.assertEqual(set(compute_all(y, predictions)), {"mse", "rmse", "mae", "r2"})


if __name__ == "__main__":
    unittest.main()
