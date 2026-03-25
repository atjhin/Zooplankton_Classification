import os
import tempfile

import pytest
import torch
import numpy as np
from torchvision import transforms

from hierroute.model import HierRouteNet
from hierroute.trainer import Trainer
from hierroute.data_setup import ImageDataset, HierImageDataset


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

@pytest.fixture
def simple_hier_adjacency():
    return {
        "root": ["ClassA", "ClassB", "ClassC"],
        "ClassA": [],
        "ClassB": [],
        "ClassC": [],
    }


@pytest.fixture
def tiny_loaders(base_image_dataset, simple_hier_adjacency):
    """Tiny train/val/test loaders from synthetic images."""
    hier_ds = HierImageDataset(
        base_dataset=base_image_dataset,
        adjacency_graph=simple_hier_adjacency,
        levels=1,
        image_transforms=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        leaves_only=True,
    )
    train_idx, val_idx, test_idx = hier_ds.split_train_test_val()
    train_dl, val_dl, test_dl = hier_ds.create_dataloaders(
        batch_size=8,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
    return train_dl, val_dl, test_dl, hier_ds


@pytest.fixture
def tiny_model(tiny_loaders):
    hier_ds = tiny_loaders[3]
    model = HierRouteNet(
        hier_ds.hierarchy,
        hier_ds.label_to_ids,
        expert_type="linear",
    )
    return model


# ================================================================== #
#  Trainer static methods                                              #
# ================================================================== #

class TestTrainerStatics:
    def test_compute_metrics(self, hierarchy_and_labels):
        h, _ = hierarchy_and_labels
        leaf_index = torch.tensor(h.get_leaf_index(), dtype=torch.float32)
        n_nodes = len(h)

        # Perfect predictions: both predict and target point to same leaf
        leaf_probs = torch.zeros(2, n_nodes)
        label_node = torch.zeros(2, n_nodes)
        leaf_ids = [nid for nid in h.nodes if h.is_leaf(nid)]
        for i in range(2):
            leaf_probs[i, leaf_ids[i]] = 1.0
            label_node[i, leaf_ids[i]] = 1.0

        acc, f1 = Trainer.compute_metrics(leaf_probs, label_node, leaf_index)
        assert acc == 1.0
        assert f1 == 1.0

    def test_clip_gradients(self):
        model = torch.nn.Linear(10, 2)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        # Artificially inflate gradients
        for p in model.parameters():
            p.grad.data *= 1000

        Trainer.clip_gradients(model, max_norm=1.0)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert total_norm.item() <= 1.0 + 1e-3


# ================================================================== #
#  Trainer.fit                                                         #
# ================================================================== #

class TestTrainerFit:
    def test_fit_one_epoch(self, tiny_loaders, tiny_model):
        train_dl, val_dl, _, _ = tiny_loaders
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                learning_rate=1e-3,
                max_epochs=1,
                device="cpu",
                model_dir=os.path.join(tmpdir, "run"),
            )
            trainer.fit(tiny_model, train_dl, val_dl)
            assert len(trainer.train_loss) == 1
            assert len(trainer.valid_loss) == 1

    def test_model_saving(self, tiny_loaders, tiny_model):
        train_dl, val_dl, _, _ = tiny_loaders
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "run")
            trainer = Trainer(
                learning_rate=1e-3,
                max_epochs=2,
                device="cpu",
                model_dir=model_dir,
            )
            trainer.fit(tiny_model, train_dl, val_dl)
            assert os.path.exists(os.path.join(model_dir, "best_model.pt"))
            assert os.path.exists(os.path.join(model_dir, "training_info.json"))

    def test_early_stopping(self, tiny_loaders, tiny_model):
        train_dl, val_dl, _, _ = tiny_loaders
        trainer = Trainer(
            learning_rate=1e-3,
            max_epochs=100,
            device="cpu",
        )
        # patience=1 means stop after 1 epoch with no improvement
        trainer.fit(tiny_model, train_dl, val_dl, patience=1, delta=99.0)
        # With delta=99.0 no epoch can improve by that much, so should stop at epoch 2
        assert len(trainer.train_loss) <= 3


# ================================================================== #
#  Trainer.predict                                                     #
# ================================================================== #

class TestTrainerPredict:
    def test_predict_returns_results(self, tiny_loaders, tiny_model):
        _, _, test_dl, _ = tiny_loaders
        trainer = Trainer(
            learning_rate=1e-3,
            max_epochs=1,
            device="cpu",
        )
        results = trainer.predict(tiny_model, test_dl)
        assert "predictions" in results
        assert "targets" in results
        assert "pred_paths" in results
        assert "level_results" in results
        assert "mismatch_results" in results

    def test_predict_save(self, tiny_loaders, tiny_model):
        _, _, test_dl, _ = tiny_loaders
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                learning_rate=1e-3,
                max_epochs=1,
                device="cpu",
                model_dir=tmpdir,
            )
            trainer.predict(tiny_model, test_dl, save=True)
            assert os.path.exists(os.path.join(tmpdir, "predictions.npz"))
            assert os.path.exists(os.path.join(tmpdir, "eval_results.json"))


# ================================================================== #
#  Trainer.count_mismatches                                            #
# ================================================================== #

class TestCountMismatches:
    def test_clean_paths(self, tiny_loaders, tiny_model):
        _, _, test_dl, _ = tiny_loaders
        trainer = Trainer(learning_rate=1e-3, max_epochs=1, device="cpu")
        tiny_model.eval()
        all_paths = []
        with torch.no_grad():
            for batch in test_dl:
                image = batch["image"]
                _, paths = tiny_model.predict(image)
                all_paths.extend(paths)
        result = trainer.count_mismatches(tiny_model, all_paths)
        assert result["passed"] is True
        assert result["structural_errors"] == 0
