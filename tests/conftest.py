import sys
import os
import tempfile
import shutil

import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Ensure hierroute is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hierroute.hierarchy import Hierarchy
from hierroute.constants import hier_adjacency_graph


# ------------------------------------------------------------------ #
#  Hierarchy fixtures                                                  #
# ------------------------------------------------------------------ #

@pytest.fixture
def mnr_adjacency_graph():
    return hier_adjacency_graph


@pytest.fixture
def hierarchy_and_labels(mnr_adjacency_graph):
    """Build a Hierarchy + label_to_id from the MNR adjacency graph."""
    label_to_id = {name: i for i, name in enumerate(mnr_adjacency_graph.keys())}

    hierarchy = Hierarchy()
    all_children = set()
    for _, kids in mnr_adjacency_graph.items():
        all_children.update(kids)
    root = [n for n in mnr_adjacency_graph if n not in all_children][0]

    hierarchy.add_node(label_to_id[root], root)

    def dfs(parent):
        for child in mnr_adjacency_graph[parent]:
            hierarchy.add_node(label_to_id[child], child, label_to_id[parent])
            dfs(child)

    dfs(root)
    return hierarchy, label_to_id


# ------------------------------------------------------------------ #
#  Small hierarchy for quick tests                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def small_adjacency_graph():
    """A minimal 3-level tree: root → A, B; A → A1, A2; B → (leaf)."""
    return {
        "root": ["A", "B"],
        "A": ["A1", "A2"],
        "B": [],
        "A1": [],
        "A2": [],
    }


@pytest.fixture
def small_hierarchy(small_adjacency_graph):
    label_to_id = {name: i for i, name in enumerate(small_adjacency_graph.keys())}
    hierarchy = Hierarchy()

    all_children = set()
    for _, kids in small_adjacency_graph.items():
        all_children.update(kids)
    root = [n for n in small_adjacency_graph if n not in all_children][0]

    hierarchy.add_node(label_to_id[root], root)

    def dfs(parent):
        for child in small_adjacency_graph[parent]:
            hierarchy.add_node(label_to_id[child], child, label_to_id[parent])
            dfs(child)

    dfs(root)
    return hierarchy, label_to_id


# ------------------------------------------------------------------ #
#  Tensor fixtures                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def dummy_batch():
    return torch.randn(4, 3, 64, 64)


# ------------------------------------------------------------------ #
#  Temporary image dataset on disk                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def tmp_image_dir():
    """Create a temp directory with synthetic 64×64 .tif images in 3 class folders."""
    tmpdir = tempfile.mkdtemp()
    classes = ["ClassA", "ClassB", "ClassC"]
    n_images_per_class = 20

    for cls in classes:
        cls_dir = os.path.join(tmpdir, cls)
        os.makedirs(cls_dir)
        for i in range(n_images_per_class):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64), dtype=np.uint8), mode="L"
            )
            img.save(os.path.join(cls_dir, f"img_{i:03d}.tif"))

    yield tmpdir, classes, n_images_per_class
    shutil.rmtree(tmpdir)


@pytest.fixture
def base_image_dataset(tmp_image_dir):
    """ImageDataset built from synthetic images."""
    from hierroute.data_setup import ImageDataset

    tmpdir, classes, _ = tmp_image_dir
    ds = ImageDataset(
        data_directory=tmpdir,
        class_names=classes,
        max_class_size=20,
        image_resolution=64,
        image_transforms=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        format_file=".tif",
    )
    return ds
