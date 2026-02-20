import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_runner import build_standardized_record, infer_shared_feature_columns


class BenchmarkRunnerTests(unittest.TestCase):
    def test_infer_shared_feature_columns_excludes_label(self):
        cic_rows = [
            {"f1": 1.0, "f2": 2.0, "Label": "BENIGN"},
            {"f1": 2.0, "f2": 3.0, "Label": "ATTACK"},
        ]
        cse_rows = [
            {"f2": 1.0, "f1": 2.0, "Label": "BENIGN"},
            {"f2": 5.0, "f1": 6.0, "Label": "ATTACK"},
        ]
        cols = infer_shared_feature_columns(cic_rows, cse_rows, label_column="Label")
        self.assertEqual(cols, ["f1", "f2"])

    def test_build_standardized_record_contains_required_metrics(self):
        metrics = {
            "accuracy": 0.9,
            "precision": 0.91,
            "recall": 0.89,
            "f1_score": 0.9,
            "auc_roc": 0.95,
            "fpr": 0.05,
        }
        rec = build_standardized_record(
            dataset="CSE-CIC-IDS2018",
            mode="zero_shot",
            threshold_method="mean_plus_k_std",
            threshold=0.12,
            seed=42,
            metrics=metrics,
            model_path="models/hybrid/model.keras",
            code_hash="abc",
            config_snapshot={"sequence_length": 10},
        )
        self.assertEqual(rec["dataset"], "CSE-CIC-IDS2018")
        self.assertEqual(rec["f1"], 0.9)
        self.assertEqual(rec["roc_auc"], 0.95)
        self.assertEqual(rec["threshold"], 0.12)
        self.assertEqual(rec["seed"], 42)


if __name__ == "__main__":
    unittest.main()
