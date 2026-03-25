import os
import tempfile

import pytest
import torch
from PIL import Image
from torchvision import transforms

from hierroute.model import FocalLoss, Expert, HierRouteNet


# ================================================================== #
#  FocalLoss                                                           #
# ================================================================== #

class TestFocalLoss:
    def test_shape(self):
        fl = FocalLoss(gamma=2.0)
        inp = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        tgt = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        out = fl(inp, tgt)
        assert out.shape == inp.shape

    def test_near_zero_on_perfect(self):
        fl = FocalLoss(gamma=2.0)
        inp = torch.tensor([[0.999, 0.001]])
        tgt = torch.tensor([[1.0, 0.0]])
        loss = fl(inp, tgt).sum()
        assert loss.item() < 0.01

    def test_gamma_effect(self):
        inp = torch.tensor([[0.9, 0.1]])
        tgt = torch.tensor([[1.0, 0.0]])
        loss_low = FocalLoss(gamma=0.0)(inp, tgt).sum()
        loss_high = FocalLoss(gamma=5.0)(inp, tgt).sum()
        # Higher gamma down-weights easy examples more
        assert loss_high.item() < loss_low.item()


# ================================================================== #
#  Expert                                                              #
# ================================================================== #

class TestExpert:
    @pytest.mark.parametrize("expert_type,input_shape", [
        ("linear", (4, 1280)),
        ("mlp", (4, 1280)),
        ("cnn", (4, 1280, 2, 2)),
    ])
    def test_output_shape(self, expert_type, input_shape):
        num_children = 3
        expert = Expert(1280, num_children, expert_type=expert_type)
        x = torch.randn(*input_shape)
        weights, logits = expert(x)
        assert weights.shape == (4, num_children)
        assert logits.shape == (4, num_children)

    def test_soft_mode_sums_to_one(self):
        expert = Expert(64, 4, mode="soft", expert_type="linear")
        x = torch.randn(2, 64)
        weights, _ = expert(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_hard_mode_returns_indices(self):
        expert = Expert(64, 4, mode="hard", expert_type="linear")
        x = torch.randn(2, 64)
        indices, logits = expert(x)
        assert indices.shape == (2,)
        assert indices.dtype == torch.int64

    def test_invalid_expert_type(self):
        with pytest.raises(ValueError, match="Unsupported expert_type"):
            Expert(64, 4, expert_type="transformer")

    def test_invalid_mode(self):
        expert = Expert(64, 4, mode="invalid", expert_type="linear")
        with pytest.raises(ValueError, match="mode must be"):
            expert(torch.randn(2, 64))


# ================================================================== #
#  HierRouteNet                                                         #
# ================================================================== #

class TestHierRouteNet:
    @pytest.mark.parametrize("expert_type", ["linear", "mlp", "cnn"])
    def test_forward_output_shapes(self, hierarchy_and_labels, dummy_batch, expert_type):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type=expert_type)
        model.eval()
        with torch.no_grad():
            leaf_probs, node_logits = model(dummy_batch)
        assert leaf_probs.shape == (4, len(h))
        assert isinstance(node_logits, dict)

    @pytest.mark.parametrize("expert_type", ["linear", "mlp", "cnn"])
    def test_forward_probs_sum_to_one(self, hierarchy_and_labels, dummy_batch, expert_type):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type=expert_type)
        model.eval()
        with torch.no_grad():
            leaf_probs, _ = model(dummy_batch)
        # Sum over all nodes — only leaf probs contribute meaningfully,
        # but path-product guarantees total leaf probability = 1
        leaf_index = torch.tensor(h.get_leaf_index(), dtype=torch.float32)
        leaf_sums = (leaf_probs * leaf_index).sum(dim=1)
        assert torch.allclose(leaf_sums, torch.ones(4), atol=1e-4)

    @pytest.mark.parametrize("expert_type", ["linear", "mlp", "cnn"])
    def test_predict_output_shapes(self, hierarchy_and_labels, dummy_batch, expert_type):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type=expert_type)
        model.eval()
        with torch.no_grad():
            leaf_ids, paths = model.predict(dummy_batch)
        assert leaf_ids.shape == (4,)
        assert len(paths) == 4

    def test_predict_paths_are_valid(self, hierarchy_and_labels, dummy_batch):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type="mlp")
        model.eval()
        with torch.no_grad():
            leaf_ids, paths = model.predict(dummy_batch)

        for path in paths:
            # Starts at root
            assert path[0] == h.root
            # Ends at leaf
            assert h.is_leaf(path[-1])
            # Each step is a valid parent→child edge
            for i in range(len(path) - 1):
                assert path[i + 1] in h.children(path[i])

    def test_loss_fn_scalar(self, hierarchy_and_labels, dummy_batch):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type="linear")
        leaf_probs, _ = model(dummy_batch)

        # Create dummy one-hot targets for a leaf node
        targets = torch.zeros_like(leaf_probs)
        leaf_nodes = [nid for nid in h.nodes if h.is_leaf(nid)]
        for i in range(4):
            targets[i, leaf_nodes[i % len(leaf_nodes)]] = 1.0

        loss = model.loss_fn(leaf_probs, targets)
        assert loss.dim() == 0  # scalar

    def test_loss_fn_gradient_flows(self, hierarchy_and_labels, dummy_batch):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type="mlp")
        leaf_probs, _ = model(dummy_batch)

        targets = torch.zeros_like(leaf_probs)
        leaf_nodes = [nid for nid in h.nodes if h.is_leaf(nid)]
        for i in range(4):
            targets[i, leaf_nodes[i % len(leaf_nodes)]] = 1.0

        loss = model.loss_fn(leaf_probs, targets)
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_freeze_backbone(self, hierarchy_and_labels):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, freeze_backbone=True, expert_type="linear")
        for p in model.shared.parameters():
            assert not p.requires_grad

    def test_checkpoint_loading(self, hierarchy_and_labels):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type="linear")

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(model.state_dict(), os.path.join(tmpdir, "best_model.pt"))
            loaded = HierRouteNet(h, label_to_id, checkpoint_dir=tmpdir, expert_type="linear")

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_batch_size_one(self, hierarchy_and_labels):
        h, label_to_id = hierarchy_and_labels
        model = HierRouteNet(h, label_to_id, expert_type="cnn")
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            leaf_probs, _ = model(x)
            leaf_ids, paths = model.predict(x)
        assert leaf_probs.shape == (1, len(h))
        assert leaf_ids.shape == (1,)


# ================================================================== #
#  Predict with real images                                            #
# ================================================================== #

DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "Processed Data"
)

REAL_IMAGES = {
    "Cladocera": os.path.join(DATA_ROOT, "Cladocera", "20240410_Erie_CMS0201_2mm_Rep1_000009_1.tif"),
    "Rotifer": os.path.join(DATA_ROOT, "Rotifer", "20240410_Erie_CMS0201_2mm_Rep2_000019_1.tif"),
    "Bubbles": os.path.join(DATA_ROOT, "Bubbles", "04072021_Huron_10_2mm_Rep2_AD_000003_292.tif"),
}


def _load_real_image(path):
    """Load a .tif image and preprocess it to (1, 3, 64, 64)."""
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).repeat(3, 1, 1)  # L → 3-channel
    return tensor.unsqueeze(0)  # add batch dim


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in REAL_IMAGES.values()),
    reason="Real image files not found on this machine",
)
class TestPredictRealImages:
    @pytest.mark.parametrize("expert_type", ["linear", "mlp", "cnn"])
    def test_predict_real_images(self, hierarchy_and_labels, expert_type):
        h, label_to_id = hierarchy_and_labels

        model = HierRouteNet(h, label_to_id, expert_type=expert_type)
        model.eval()

        for class_name, img_path in REAL_IMAGES.items():
            x = _load_real_image(img_path)
            with torch.no_grad():
                leaf_ids, paths = model.predict(x)

            # Output structure checks
            assert leaf_ids.shape == (1,)
            leaf_id = leaf_ids.item()
            assert h.is_leaf(leaf_id), f"Predicted node {leaf_id} is not a leaf"

            path = paths[0]
            assert path[0] == h.root
            assert path[-1] == leaf_id
            for i in range(len(path) - 1):
                assert path[i + 1] in h.children(path[i]), (
                    f"Invalid edge {path[i]} -> {path[i+1]} for {class_name}"
                )
