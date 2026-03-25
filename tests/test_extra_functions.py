import json
import os
import tempfile

import pytest
import torch
import numpy as np

from hierroute.extra_functions import set_seed


# ================================================================== #
#  set_seed                                                            #
# ================================================================== #

class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(99)
        b = torch.randn(5)
        assert not torch.equal(a, b)

    def test_numpy_reproducibility(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ================================================================== #
#  Visualize                                                           #
# ================================================================== #

class TestVisualize:
    @pytest.fixture
    def mock_model_dir(self):
        """Create a temp dir with minimal training_info.json and eval_results.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_info = {
                "hyperparameters": {
                    "learning_rate": 1e-3,
                    "max_epochs": 5,
                    "gradient_clip_val": 1,
                    "patience": 15,
                    "delta": 0.0005,
                },
                "results": {
                    "best_epoch": 3,
                    "best_val_acc": 0.85,
                    "total_epochs_run": 5,
                    "early_stopped": False,
                },
                "training_metrics": {
                    "train_loss": [0.5, 0.4, 0.3, 0.25, 0.2],
                    "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9],
                    "train_f1": [0.55, 0.65, 0.75, 0.8, 0.85],
                    "valid_loss": [0.6, 0.5, 0.35, 0.3, 0.32],
                    "valid_acc": [0.55, 0.65, 0.8, 0.85, 0.83],
                    "valid_f1": [0.5, 0.6, 0.75, 0.82, 0.8],
                },
            }
            eval_results = {
                "level_results": {
                    "1": {
                        "overall": {
                            "accuracy": 0.85,
                            "f1": 0.83,
                            "precision": 0.84,
                            "recall": 0.83,
                            "n_samples": 100,
                        },
                        "per_class": {
                            "ClassA": {
                                "node_id": 1,
                                "accuracy": 0.9,
                                "f1": 0.88,
                                "precision": 0.87,
                                "recall": 0.9,
                                "n_samples": 50,
                            },
                            "ClassB": {
                                "node_id": 2,
                                "accuracy": 0.8,
                                "f1": 0.78,
                                "precision": 0.81,
                                "recall": 0.76,
                                "n_samples": 50,
                            },
                        },
                    }
                },
                "mismatch_results": {
                    "structural_errors": 0,
                    "error_pct": 0.0,
                    "passed": True,
                },
            }
            with open(os.path.join(tmpdir, "training_info.json"), "w") as f:
                json.dump(training_info, f)
            with open(os.path.join(tmpdir, "eval_results.json"), "w") as f:
                json.dump(eval_results, f)

            # Minimal predictions.npz
            np.savez(
                os.path.join(tmpdir, "predictions.npz"),
                predictions=np.array([1, 2, 1, 2]),
                targets=np.array([1, 2, 2, 1]),
                level_1_preds=np.array([1, 2, 1, 2]),
                level_1_trues=np.array([1, 2, 2, 1]),
            )
            yield tmpdir

    def test_visualize_init(self, mock_model_dir):
        import matplotlib
        matplotlib.use("Agg")
        from hierroute.extra_functions import Visualize

        viz = Visualize(mock_model_dir)
        assert viz.training_info is not None
        assert viz.eval_results is not None
        assert viz.pred_data is not None

    def test_plot_train_runs(self, mock_model_dir):
        import matplotlib
        matplotlib.use("Agg")
        from hierroute.extra_functions import Visualize

        viz = Visualize(mock_model_dir)
        viz.plot_train()  # should not raise
