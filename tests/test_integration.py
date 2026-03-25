"""
End-to-end integration test: synthetic images → dataset → model → train → predict → visualize.
"""
import os
import tempfile

import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.model import HierRouteNet
from hierroute.trainer import Trainer
from hierroute.extra_functions import set_seed


@pytest.fixture
def integration_setup():
    """Full pipeline setup with synthetic data."""
    set_seed(42)
    tmpdir = tempfile.mkdtemp()
    classes = ["ClassA", "ClassB", "ClassC"]
    n_per_class = 30

    # Create synthetic images
    for cls in classes:
        cls_dir = os.path.join(tmpdir, cls)
        os.makedirs(cls_dir)
        for i in range(n_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L"
            )
            img.save(os.path.join(cls_dir, f"img_{i:03d}.tif"))

    adjacency_graph = {
        "root": ["ClassA", "ClassB", "ClassC"],
        "ClassA": [],
        "ClassB": [],
        "ClassC": [],
    }

    yield tmpdir, classes, n_per_class, adjacency_graph

    import shutil
    shutil.rmtree(tmpdir)


def test_full_pipeline(integration_setup):
    tmpdir, classes, n_per_class, adjacency_graph = integration_setup

    # 1. Build datasets
    base_ds = ImageDataset(
        data_directory=tmpdir,
        class_names=classes,
        max_class_size=n_per_class,
        image_resolution=64,
        image_transforms=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        format_file=".tif",
    )

    hier_ds = HierImageDataset(
        base_dataset=base_ds,
        adjacency_graph=adjacency_graph,
        levels=1,
        image_transforms=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        leaves_only=True,
    )

    assert len(hier_ds) == len(classes) * n_per_class

    # 2. Split and create loaders
    train_idx, val_idx, test_idx = hier_ds.split_train_test_val()
    train_dl, val_dl, test_dl = hier_ds.create_dataloaders(
        batch_size=8,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )

    # 3. Instantiate model
    model = HierRouteNet(
        hier_ds.hierarchy,
        hier_ds.label_to_ids,
        expert_type="cnn",
    )

    # 4. Train for 2 epochs
    model_dir = os.path.join(tmpdir, "model_output")
    trainer = Trainer(
        learning_rate=1e-3,
        max_epochs=2,
        device="cpu",
        model_dir=model_dir,
    )
    trainer.fit(model, train_dl, val_dl, patience=10)

    assert len(trainer.train_loss) == 2
    assert os.path.exists(os.path.join(model_dir, "best_model.pt"))
    assert os.path.exists(os.path.join(model_dir, "training_info.json"))

    # 5. Predict on test set
    results = trainer.predict(model, test_dl, save=True)

    assert results["predictions"].shape[0] == len(test_idx)
    assert results["targets"].shape[0] == len(test_idx)
    assert results["mismatch_results"]["passed"] is True
    assert os.path.exists(os.path.join(model_dir, "predictions.npz"))
    assert os.path.exists(os.path.join(model_dir, "eval_results.json"))

    # 6. Visualize loads without error
    import matplotlib
    matplotlib.use("Agg")
    from hierroute.extra_functions import Visualize

    viz = Visualize(model_dir)
    assert viz.training_info is not None
    assert viz.eval_results is not None
    viz.plot_train()
