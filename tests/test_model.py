import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model import ROOT, FEATURES, build_model, load_students, metrics, run, train_and_evaluate

class RegressionTests(unittest.TestCase):
    def test_imputer_only_learns_from_training_rows(self):
        train = pd.DataFrame({"study_hours": [1.0, 3.0, np.nan]})
        model = build_model().fit(train, [40, 50, 45])
        model.predict(pd.DataFrame({"study_hours": [100.0, np.nan]}))
        self.assertEqual(model.named_steps["simpleimputer"].statistics_[0], 2.0)

    def test_train_test_sets_do_not_overlap(self):
        data = load_students(ROOT / "data/students.csv")
        model, scores, train, predictions = train_and_evaluate(data)
        self.assertEqual(len(train), 96)
        self.assertEqual(len(predictions), 24)
        self.assertFalse(set(train.student_id) & set(predictions.student_id))
        self.assertAlmostEqual(scores["training_median_hours"], train.study_hours.median())
        self.assertTrue(np.isfinite(predictions.predicted_score).all())

    def test_exact_linear_relationship(self):
        data = pd.DataFrame({"student_id": [f"S{i}" for i in range(30)],
                             "study_hours": np.linspace(0, 10, 30)})
        data["exam_score"] = 30 + 4 * data.study_hours
        _, scores, _, _ = train_and_evaluate(data)
        self.assertLess(scores["linear_regression"]["mae"], 1e-10)
        self.assertGreater(scores["mean_baseline"]["mae"], 1)

    def test_metrics_have_the_expected_units(self):
        scores = metrics([10, 20], [12, 18])
        self.assertEqual(scores["mae"], 2)
        self.assertEqual(scores["rmse"], 2)

    def test_pipeline_and_extrapolation_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            run(output_dir=directory, hours=4)
            self.assertTrue((Path(directory) / "regression.png").is_file())
            with self.assertRaisesRegex(ValueError, "training range"):
                run(output_dir=directory, hours=100)

if __name__ == "__main__":
    unittest.main()
