import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hybrid_pipeline as hp


class _StubHistory:
    def __init__(self):
        self.history = {"loss": [0.1], "val_loss": [0.1]}


class _StubModel:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, inputs, outputs, **kwargs):
        self.fit_calls += 1
        return _StubHistory()

    def predict(self, inputs, verbose=0):
        return [[[0.0 for _ in row] for row in sequence] for sequence in inputs]


def _stub_model_builder(_n_features, _sequence_length, _config):
    return _StubModel(), []


class HybridPipelineTests(unittest.TestCase):
    def test_align_feature_columns_orders_and_fills_missing(self):
        rows = [
            {
                "b_feature": 1.0,
                "a_feature": 3.0,
            },
            {
                "b_feature": 2.0,
                "a_feature": 4.0,
            },
        ]

        aligned = hp.align_feature_columns(rows, ["a_feature", "c_feature", "b_feature"])

        self.assertEqual(list(aligned[0].keys()), ["a_feature", "c_feature", "b_feature"])
        self.assertEqual(aligned[0]["c_feature"], 0.0)
        self.assertEqual(aligned[1]["c_feature"], 0.0)

    def test_train_zero_shot_and_few_shot_flow(self):
        cic2017 = [
            {"f1": 0.0, "f2": 0.0, "Label": "BENIGN"},
            {"f1": 0.1, "f2": 0.2, "Label": "BENIGN"},
            {"f1": 0.2, "f2": 0.1, "Label": "BENIGN"},
            {"f1": 0.1, "f2": 0.2, "Label": "BENIGN"},
            {"f1": 0.0, "f2": 0.0, "Label": "BENIGN"},
            {"f1": 0.2, "f2": 0.1, "Label": "BENIGN"},
        ]
        cic2018 = [
            {"f2": 0.1, "f1": 0.1, "Label": "BENIGN"},
            {"f2": 0.2, "f1": 0.2, "Label": "BENIGN"},
            {"f2": 5.0, "f1": 5.0, "Label": "ATTACK"},
            {"f2": 6.0, "f1": 6.0, "Label": "ATTACK"},
            {"f2": 0.3, "f1": 0.1, "Label": "BENIGN"},
            {"f2": 7.0, "f1": 8.0, "Label": "ATTACK"},
        ]

        config = hp.HybridExperimentConfig(sequence_length=2, epochs=1, batch_size=2)
        state = hp.train_cic2017_normal(
            cic2017,
            config=config,
            model_builder=_stub_model_builder,
        )
        self.assertGreaterEqual(state.threshold, 0.0)
        self.assertEqual(state.model.fit_calls, 1)

        zero_shot = hp.evaluate_zero_shot_cic2018(state, cic2018)
        self.assertIn("metrics", zero_shot)
        self.assertIn("f1_score", zero_shot["metrics"])
        self.assertIn("fpr", zero_shot["metrics"])

        adapted_state = hp.adapt_few_shot_benign_1pct(
            state,
            cic2018,
            benign_ratio=0.5,
            adaptation_epochs=1,
        )
        self.assertEqual(adapted_state.model.fit_calls, 2)

        post_adapt = hp.evaluate_post_adaptation(adapted_state, cic2018)
        self.assertIn("metrics", post_adapt)
        self.assertIn("f1_score", post_adapt["metrics"])

    def test_generalization_gap_uses_f1_difference(self):
        gap = hp.compute_generalization_gap(
            in_distribution_metrics={"f1_score": 0.91},
            out_distribution_metrics={"f1_score": 0.76},
        )
        self.assertAlmostEqual(gap, 0.15)


if __name__ == "__main__":
    unittest.main()
