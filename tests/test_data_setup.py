import os
import pytest
import torch
from torchvision import transforms

from hierroute.data_setup import ImageDataset, HierImageDataset


# ================================================================== #
#  ImageDataset                                                        #
# ================================================================== #

class TestImageDataset:
    def test_dataset_length(self, tmp_image_dir):
        tmpdir, classes, n_per_class = tmp_image_dir
        ds = ImageDataset(
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
        assert len(ds) == len(classes) * n_per_class

    def test_split_proportions(self, base_image_dataset):
        ds = base_image_dataset
        train, val, test = ds.split_train_test_val(verbose=False)
        total = len(ds)
        assert abs(len(train) / total - 0.7) < 0.1
        assert abs(len(val) / total - 0.1) < 0.1
        assert abs(len(test) / total - 0.2) < 0.1

    def test_split_no_overlap(self, base_image_dataset):
        ds = base_image_dataset
        train, val, test = ds.split_train_test_val(verbose=False)
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_split_covers_all(self, base_image_dataset):
        ds = base_image_dataset
        train, val, test = ds.split_train_test_val(verbose=False)
        assert set(train) | set(val) | set(test) == set(range(len(ds)))

    def test_create_dataloaders(self, base_image_dataset):
        ds = base_image_dataset
        train, val, test = ds.split_train_test_val(verbose=False)
        train_dl, val_dl, test_dl = ds.create_dataloaders(
            batch_size=4,
            train_indices=train,
            val_indices=val,
            test_indices=test,
        )
        assert len(train_dl) > 0
        assert len(val_dl) > 0
        assert len(test_dl) > 0


# ================================================================== #
#  HierImageDataset                                                    #
# ================================================================== #

@pytest.fixture
def simple_hier_adjacency():
    """Adjacency graph that matches the 3 synthetic classes."""
    return {
        "root": ["ClassA", "ClassB", "ClassC"],
        "ClassA": [],
        "ClassB": [],
        "ClassC": [],
    }


@pytest.fixture
def hier_dataset(base_image_dataset, simple_hier_adjacency):
    ds = HierImageDataset(
        base_dataset=base_image_dataset,
        adjacency_graph=simple_hier_adjacency,
        levels=1,
        image_transforms=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        leaves_only=True,
    )
    return ds


class TestHierImageDataset:
    def test_length(self, hier_dataset):
        assert len(hier_dataset) > 0

    def test_getitem_keys(self, hier_dataset):
        item = hier_dataset[0]
        expected_keys = {"image", "label_node", "path", "targets", "masks"}
        assert set(item.keys()) == expected_keys

    def test_getitem_image_shape(self, hier_dataset):
        item = hier_dataset[0]
        assert item["image"].shape == (3, 64, 64)

    def test_path_starts_at_root(self, hier_dataset):
        item = hier_dataset[0]
        h = hier_dataset.hierarchy
        assert item["path"][0] == h.root

    def test_path_ends_at_label(self, hier_dataset):
        item = hier_dataset[0]
        assert item["path"][-1] == item["label_node"]

    def test_targets_are_valid_child_indices(self, hier_dataset):
        item = hier_dataset[0]
        h = hier_dataset.hierarchy
        for parent_id, child_idx in item["targets"].items():
            children = h.children(parent_id)
            assert 0 <= child_idx < len(children)

    def test_masks_are_binary(self, hier_dataset):
        item = hier_dataset[0]
        for parent_id, mask in item["masks"].items():
            assert all(v in (0, 1) for v in mask)

    def test_collate_fn(self, hier_dataset):
        batch = [hier_dataset[i] for i in range(4)]
        collated = hier_dataset.collate_fn(batch)
        assert collated["image"].shape == (4, 3, 64, 64)
        assert collated["label_node"].shape[0] == 4
        # label_node should be one-hot encoded
        assert collated["label_node"].shape[1] == len(hier_dataset.hierarchy)

    def test_leaves_only_filtering(self, base_image_dataset, simple_hier_adjacency):
        ds_all = HierImageDataset(
            base_dataset=base_image_dataset,
            adjacency_graph=simple_hier_adjacency,
            levels=1,
            image_transforms=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]),
            leaves_only=False,
        )
        ds_leaves = HierImageDataset(
            base_dataset=base_image_dataset,
            adjacency_graph=simple_hier_adjacency,
            levels=1,
            image_transforms=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]),
            leaves_only=True,
        )
        h = ds_leaves.hierarchy
        for i in range(len(ds_leaves)):
            label = ds_leaves.labels[i]
            assert h.is_leaf(label)

    def test_create_dataloaders(self, hier_dataset):
        train, val, test = hier_dataset.split_train_test_val()
        train_dl, val_dl, test_dl = hier_dataset.create_dataloaders(
            batch_size=4,
            train_indices=train,
            val_indices=val,
            test_indices=test,
        )
        batch = next(iter(train_dl))
        assert "image" in batch
        assert "label_node" in batch
