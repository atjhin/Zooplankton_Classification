
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from .extra_functions import validate_checkpoint

class FocalLoss(nn.Module):
    """
    Focal loss for binary cross-entropy on probability inputs with one-hot targets.
    Applies modulating factor (1 - pt)^gamma to down-weight easy examples.
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        """
        input: [N, C], float32 — probabilities (already softmaxed/path-product)
        target: [N, C], float32 — one-hot encoded targets
        Returns: [N, C] per-element focal BCE losses (unreduced)
        """
        bce = -(target * torch.log(input) + (1 - target) * torch.log(1 - input))
        pt = target * input + (1 - target) * (1 - input)
        focal_weight = (1 - pt) ** self.gamma
        return focal_weight * bce
        

class Expert(nn.Module):
    def __init__(self, in_dim, num_children, mode="soft", expert_type="mlp"):
        super().__init__()
        if expert_type == "linear":
            self.layer = nn.Sequential(nn.Linear(in_dim, num_children))
        elif expert_type == "mlp":
            self.layer = nn.Sequential(
                nn.Linear(in_dim, 2*in_dim),
                nn.ReLU(),
                nn.Linear(2*in_dim, num_children))
        elif expert_type == "cnn":
            self.layer = nn.Sequential(
                nn.Conv2d(in_dim, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 64, kernel_size=2),
                nn.ReLU(),
                nn.Flatten(1),
                nn.LazyLinear(num_children)) # allows different resolution images.
        else:
            raise ValueError(f"Unsupported expert_type: {expert_type}")
        self.mode = mode
        self.expert_type = expert_type

    def forward(self, x):
        logits = self.layer(x)
        if self.mode == "soft":
            weights = torch.softmax(logits, dim=-1)
            return weights, logits

        elif self.mode == "hard":
            indices = torch.argmax(logits, dim=-1)
            return indices, logits

        else:
            raise ValueError("mode must be 'soft' or 'hard'")

class HierRouteNet(nn.Module):
    def __init__(self, hierarchy, label_to_id, weights_directory=None,
                 checkpoint_dir=None, seed=123, loss_type="bce", focal_gamma=2.0,
                 backbone="efficientnet_b0", freeze_backbone=False,
                 expert_type="mlp"):
        super().__init__()
        self.hierarchy = hierarchy
        self.seed = seed
        self.label_to_id = label_to_id
        self.loss_type = loss_type
        self.backbone_type = backbone
        self.expert_type = expert_type

        if backbone == "efficientnet_b0":
            bb = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.shared = bb.features
            self.pool = bb.avgpool
            self.feature_dim = bb.classifier[1].in_features
        elif backbone == "swin_t":
            bb = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            self.shared = bb.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = bb.head.in_features
        elif backbone == "swin_s":
            bb = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
            self.shared = bb.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feature_dim = bb.head.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.shared.parameters():
                param.requires_grad = False

        self.local_classifiers = nn.ModuleDict()
        for node_id, node in hierarchy.nodes.items():
            if len(node.children) > 0:
                self.local_classifiers[str(node_id)] = Expert(self.feature_dim, len(node.children), expert_type=expert_type)
        self.leaf_index = torch.tensor(hierarchy.get_leaf_index(), dtype=torch.float32)
        if loss_type == "focal":
            self.loss = FocalLoss(gamma=focal_gamma)
        else:
            self.loss = nn.BCELoss(reduction='none')

        if checkpoint_dir is not None:
            pt_files = list(Path(checkpoint_dir).glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No .pt file found in '{checkpoint_dir}'")
            checkpoint_path = pt_files[0]
            ckpt_state = validate_checkpoint(checkpoint_path, self, backbone, expert_type)
            self.load_state_dict(ckpt_state)
            print(f"Loaded checkpoint: {checkpoint_path}")


    def forward(self, x):
        # --- Backbone ---
        x = self.shared(x)
        if self.backbone_type.startswith("swin"):
            x = x.permute(0, 3, 1, 2)              # (B, H, W, C) -> (B, C, H, W)

        if self.expert_type == "cnn":
            feat = x                                # (B, C, H, W) — pre-pool
        else:
            feat = torch.flatten(self.pool(x), 1)   # (B, feature_dim)

        # --- Run all local classifiers in one pass ---
        # Store conditional probs for every internal node
        # node_probs[node_id] -> (B, num_children) soft probabilities
        node_probs = {}
        node_logits = {}
        for node_id, classifier in self.local_classifiers.items():
            probs, logits = classifier.forward(feat) # (B, num_children)
            node_probs[node_id] = probs
            node_logits[node_id] = logits

        # --- Compute leaf probabilities via path products ---
        # For each leaf, walk root -> leaf and multiply conditional probs
        # P(leaf) = P(child_k | node_n) * P(node_n | node_n-1) * ... * P(node_1 | root)
        leaf_probs = [-1] * len(self.hierarchy.nodes)
        for node_id, node in self.hierarchy.nodes.items():
            path = self.hierarchy.get_path_to_root(node_id)

            # Start with prob=1 and multiply down the path
            B = feat.shape[0]
            prob = torch.ones(B, device=feat.device) # (B,)

            for i in range(len(path) - 1):
                parent = path[i]
                child  = path[i + 1]

                child_idx = self.hierarchy.nodes[parent].children.index(child)

                prob = prob * node_probs[str(parent)][:, child_idx]  # (B,)
            # print(prob)
            leaf_probs[node_id] = prob  # (B,)

        logit_tensor = torch.stack(leaf_probs).transpose(0,1)

        return logit_tensor, node_logits


    def predict(self, x):
        """
        Predict class labels by greedily following argmax decisions from root to leaf,
        recording the expert path taken for each sample.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            leaf_ids : LongTensor of shape (B,) — predicted leaf node_id per sample
            paths    : list of lists — paths[i] is [root, ..., leaf] for sample i,
                       reflecting the exact sequence of expert decisions made
        """
        feat = self.shared(x)
        if self.backbone_type.startswith("swin"):
            feat = feat.permute(0, 3, 1, 2)     # (B, H, W, C) -> (B, C, H, W)

        if self.expert_type == "cnn":
            pass                                 # keep (B, C, H, W) for CNN experts
        else:
            feat = torch.flatten(self.pool(feat), 1)  # (B, feature_dim)

        B = feat.shape[0]
        current_nodes = [self.hierarchy.root] * B
        paths = [[self.hierarchy.root] for _ in range(B)]

        while not all(self.hierarchy.is_leaf(n) for n in current_nodes):
            # Group sample indices by their current node
            node_to_indices = {}
            for i, node_id in enumerate(current_nodes):
                if not self.hierarchy.is_leaf(node_id):
                    node_to_indices.setdefault(node_id, []).append(i)

            for node_id, indices in node_to_indices.items():
                group_feat = feat[indices]                                  # (G, feature_dim)
                _, logits = self.local_classifiers[str(node_id)](group_feat)  # (G, num_children)
                child_idx = torch.argmax(logits, dim=-1)                    # (G,)
                children = self.hierarchy.nodes[node_id].children
                for sample_idx, cidx in zip(indices, child_idx.tolist()):
                    next_node = children[cidx]
                    current_nodes[sample_idx] = next_node
                    paths[sample_idx].append(next_node)

        leaf_ids = torch.tensor(current_nodes, dtype=torch.long, device=feat.device)
        return leaf_ids, paths


    def loss_fn(self, logits, targets):
        leaf_index = self.leaf_index.to(logits.device)
        logits = logits.clamp(1e-7, 1 - 1e-7)
        loss = self.loss(logits, targets) * leaf_index
        loss = loss.sum(dim=1) / self.leaf_index.sum()
        return loss.mean()



