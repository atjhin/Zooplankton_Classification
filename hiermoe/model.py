
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


class Expert(nn.Module):
    def __init__(self, in_dim, num_children, mode="soft", n_hidden=1):
        super().__init__()
        if n_hidden==0:
            self.layer = nn.Sequential(nn.Linear(in_dim, num_children))
        elif n_hidden==1:
            self.layer = nn.Sequential(
                nn.Linear(in_dim, 2*in_dim), 
                nn.ReLU(),
                nn.Linear(2*in_dim, num_children))
        else:
            print(f"Invalid number of hidden layers defaulting to 0 hidden layers")
            self.layer = nn.Sequential(nn.Linear(in_dim, num_children))
        self.mode = mode

    def forward(self, x):
        # x = self.relu(self.fc1(x))
        logits = self.layer(x)
        if self.mode == "soft":
            weights = torch.softmax(logits, dim=-1)
            return weights, logits

        elif self.mode == "hard":
            indices = torch.argmax(logits, dim=-1)
            return indices, logits

        else:
            raise ValueError("mode must be 'soft' or 'hard'")

class HierMoeNet(nn.Module):
    def __init__(self, hierarchy, label_to_id, weights_directory=None,
                 checkpoint_dir=None, seed=123):
        super().__init__()
        self.hierarchy = hierarchy
        self.seed = seed
        self.label_to_id = label_to_id
        backbone = models.efficientnet_b0(weights = weights_directory)
        self.shared = backbone.features
        self.pool = backbone.avgpool
        self.feature_dim = backbone.classifier[1].in_features
        self.local_classifiers = nn.ModuleDict()
        for node_id, node in hierarchy.nodes.items():
            if len(node.children) > 0:
                self.local_classifiers[str(node_id)] = Expert(self.feature_dim,len(node.children))
        self.leaf_index = torch.tensor(hierarchy.get_leaf_index(), dtype=torch.float32)
        self.bce = nn.BCELoss(reduction='none')

        if checkpoint_dir is not None:
            pt_files = list(Path(checkpoint_dir).glob("*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"No .pt file found in '{checkpoint_dir}'")
            checkpoint_path = pt_files[0]
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            print(f"Loaded checkpoint: {checkpoint_path}")


    def forward(self, x):
        # --- Backbone ---
        x = self.shared(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)                    # (B, feature_dim)

        # --- Run all local classifiers in one pass ---
        # Store conditional probs for every internal node
        # node_probs[node_id] -> (B, num_children) soft probabilities
        node_probs = {}
        node_logits = {}
        for node_id, classifier in self.local_classifiers.items():
            probs, logits = classifier.forward(x)   # (B, num_children)
            node_probs[node_id] = probs
            node_logits[node_id] = logits

        # --- Compute leaf probabilities via path products ---
        # For each leaf, walk root -> leaf and multiply conditional probs
        # P(leaf) = P(child_k | node_n) * P(node_n | node_n-1) * ... * P(node_1 | root)
        leaf_probs = [-1] * len(self.hierarchy.nodes)
        for node_id, node in self.hierarchy.nodes.items():
            path = self.hierarchy.get_path_to_root(node_id)

            # Start with prob=1 and multiply down the path
            B = x.shape[0]
            prob = torch.ones(B, device=x.device)   # (B,)

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
        feat = self.pool(feat)
        feat = torch.flatten(feat, 1)           # (B, feature_dim)

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
        loss = self.bce(logits, targets) * leaf_index
        loss = loss.sum(dim=1) / self.leaf_index.sum()
        return loss.mean()



