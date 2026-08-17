import unittest

import numpy as np
import pandas as pd

from Utils.catboost_modeling import detect_task, plot_feature_importance_signed
from Utils.modeling_utils import (
    compute_feature_importance,
    evaluate_model,
    prepare_features_and_target,
    predict_with_explanation,
    train_logistic_regression,
)


class ModelingTests(unittest.TestCase):
    def test_catboost_task_detection_and_importance_figures(self):
        binary_df = pd.DataFrame({"target": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
        multiclass_df = pd.DataFrame({"target": [0, 1, 2, 0], "x": [1, 2, 3, 4]})
        regression_df = pd.DataFrame({"target": [0.1, 1.5, 2.3, 4.8], "x": [1, 2, 3, 4]})

        self.assertEqual(detect_task(binary_df, "target"), "binary")
        self.assertEqual(detect_task(multiclass_df, "target"), "multiclass")
        self.assertEqual(detect_task(regression_df, "target"), "regression")

        importance = pd.DataFrame(
            {
                "feature": ["positive", "negative", "weak"],
                "importance": [0.7, -0.4, 0.1],
            }
        )
        figures = plot_feature_importance_signed(importance, top_n=3)

        self.assertEqual(set(figures.keys()), {"pos", "neg"})
        self.assertGreater(len(figures["pos"].data), 0)
        self.assertGreater(len(figures["neg"].data), 0)

    def test_logistic_regression_pipeline_end_to_end(self):
        rng = np.random.default_rng(42)
        n = 80
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        group = np.where(x1 > 0, "high", "low")
        target = ((x1 + 0.5 * x2) > 0).astype(int)
        df = pd.DataFrame({"x1": x1, "x2": x2, "group": group, "target": target})

        X, y, label_encoder, _, _ = prepare_features_and_target(df, "target")
        model, meta = train_logistic_regression(
            X,
            y,
            C=1.0,
            penalty="l2",
            max_iter=500,
            label_encoder=label_encoder,
        )
        metrics, roc_data, pr_data = evaluate_model(model, X, y, meta, threshold=0.5)
        importance = compute_feature_importance(model, meta)
        prediction = predict_with_explanation(model, meta, X.iloc[[0]], threshold=0.5)

        self.assertGreaterEqual(metrics["Accuracy"], 0.8)
        self.assertEqual(len(roc_data), 2)
        self.assertEqual(len(pr_data), 2)
        self.assertFalse(importance.empty)
        self.assertIn(prediction["pred_class_int"], {0, 1})
        self.assertIn("explanation", prediction)


if __name__ == "__main__":
    unittest.main()
