# Test Suite Documentation

This document describes all tests in the `tests/` directory for the `hierroute` module.

Run all tests:
```bash
cd Zooplankton_Classification && python -m pytest tests/ -v
```

## File Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── test_hierarchy.py        # Node and Hierarchy data structures
├── test_model.py            # FocalLoss, Expert, HierRouteNet
├── test_trainer.py          # Trainer (fit, predict, evaluate)
├── test_extra_functions.py  # set_seed, Visualize
└── test_integration.py      # End-to-end pipeline
```

---

## conftest.py — Shared Fixtures

Provides reusable fixtures across all test files:

| Fixture | Description |
|---------|-------------|
| `mnr_adjacency_graph` | Returns the full MNR zooplankton adjacency graph from `constants.py` |
| `hierarchy_and_labels` | Builds a `Hierarchy` object + `label_to_id` dict from the MNR graph (18 nodes, 12 leaves, 3 levels) |
| `small_adjacency_graph` | A minimal 5-node tree: `root → A, B; A → A1, A2` |
| `small_hierarchy` | Builds a `Hierarchy` from the small adjacency graph |
| `dummy_batch` | `torch.randn(4, 3, 64, 64)` — a batch of 4 random images |
| `tmp_image_dir` | Creates a temporary directory with 3 class folders, each containing 20 synthetic 64x64 `.tif` images. Cleaned up after tests. |
| `base_image_dataset` | An `ImageDataset` built from `tmp_image_dir` with Resize + ToTensor transforms |

---

## test_hierarchy.py — Node and Hierarchy

Tests the tree data structures that define the classification hierarchy.

### TestNode (4 tests)

| Test | What it verifies |
|------|-----------------|
| `test_node_creation` | Default attributes: `node_id`, `name`, `parent=None`, `children=[]`, `depth=None` |
| `test_node_with_parent` | Parent assignment via constructor |
| `test_add_child` | Adding children appends to the `children` list |
| `test_add_child_no_duplicates` | Adding the same child twice does not create duplicates |

### TestHierarchy (13 tests)

Uses a small 5-node tree (`root → A, B; A → A1, A2`).

| Test | What it verifies |
|------|-----------------|
| `test_add_nodes_and_root` | Root is correctly identified as the node with no parent |
| `test_len` | `len(hierarchy)` returns the total number of nodes (5) |
| `test_children` | `children()` returns correct child node IDs |
| `test_parent` | `parent()` returns the correct parent; root's parent is `None` |
| `test_is_leaf` | Leaf nodes return `True`; internal nodes return `False` |
| `test_depth` | Root has depth 0, children depth 1, grandchildren depth 2 |
| `test_get_path_to_root` | Path from leaf to root is ordered `[root, ..., leaf]` with correct length |
| `test_get_path_to_root_for_root` | Path for root itself is `[root]` |
| `test_descendants` | All descendants of root = 4 nodes (excludes root itself) |
| `test_subtree_leaves` | Returns only leaf nodes under a subtree |
| `test_subtree_leaves_of_leaf` | A leaf node's subtree leaves is just itself |
| `test_get_leaf_index` | Binary mask has `1` at leaf positions, `0` at internal nodes |
| `test_invalid_node_path` | Raises `ValueError` when querying a non-existent node |

### TestMNRHierarchy (4 tests)

Validates the full MNR zooplankton hierarchy from `constants.py`.

| Test | What it verifies |
|------|-----------------|
| `test_node_count` | Total of 18 nodes |
| `test_leaf_count` | Exactly 12 leaf (species-level) nodes |
| `test_max_depth` | Maximum depth is 3 (root → order → family → species) |
| `test_all_paths_start_at_root` | Every node's path begins at the root |

---

## test_model.py — FocalLoss, Expert, HierRouteNet

Tests the neural network components.

### TestFocalLoss (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_shape` | Output shape matches input shape `[N, C]` |
| `test_near_zero_on_perfect` | Loss approaches 0 when predictions match targets |
| `test_gamma_effect` | Higher gamma produces lower loss on easy (high-confidence) examples |

### TestExpert (7 tests)

Tests all three expert types: `linear`, `mlp`, and `cnn`.

| Test | What it verifies |
|------|-----------------|
| `test_output_shape[linear]` | Linear expert: flat input `(B, 1280)` produces `(B, C)` output |
| `test_output_shape[mlp]` | MLP expert: flat input `(B, 1280)` produces `(B, C)` output |
| `test_output_shape[cnn]` | CNN expert: 4D input `(B, 1280, 2, 2)` produces `(B, C)` output |
| `test_soft_mode_sums_to_one` | Soft mode weights sum to 1 across children (valid probability distribution) |
| `test_hard_mode_returns_indices` | Hard mode returns integer argmax indices |
| `test_invalid_expert_type` | Raises `ValueError` for unsupported expert type |
| `test_invalid_mode` | Raises `ValueError` for unsupported mode |

### TestHierRouteNet (12 tests)

Tests the full hierarchical mixture-of-experts model. Many tests are parametrized across all three expert types.

| Test | What it verifies |
|------|-----------------|
| `test_forward_output_shapes[linear/mlp/cnn]` | Forward pass produces `(B, num_nodes)` leaf probabilities and a dict of node logits |
| `test_forward_probs_sum_to_one[linear/mlp/cnn]` | Leaf probabilities sum to 1 per sample (path-product property) |
| `test_predict_output_shapes[linear/mlp/cnn]` | `predict()` returns `leaf_ids (B,)` and `paths` (list of lists) |
| `test_predict_paths_are_valid` | Every predicted path starts at root, ends at a leaf, and each step is a valid parent-child edge |
| `test_loss_fn_scalar` | `loss_fn()` returns a scalar tensor |
| `test_loss_fn_gradient_flows` | Gradients propagate through the loss back to model parameters |
| `test_freeze_backbone` | With `freeze_backbone=True`, all backbone parameters have `requires_grad=False` |
| `test_checkpoint_loading` | Save and reload via `checkpoint_dir` produces identical weights |
| `test_batch_size_one` | Model handles single-sample batches correctly |

### TestPredictRealImages (3 tests)

Loads one real `.tif` image from each of three classes and runs inference. Skipped if the data files are not present on the machine.

| Test | What it verifies |
|------|-----------------|
| `test_predict_real_images[linear]` | Real image inference with linear experts produces valid leaf predictions and paths |
| `test_predict_real_images[mlp]` | Real image inference with MLP experts produces valid leaf predictions and paths |
| `test_predict_real_images[cnn]` | Real image inference with CNN experts produces valid leaf predictions and paths |

Images used:
- **Cladocera**: `data/Processed Data/Cladocera/20240410_Erie_CMS0201_2mm_Rep1_000009_1.tif`
- **Rotifer**: `data/Processed Data/Rotifer/20240410_Erie_CMS0201_2mm_Rep2_000019_1.tif`
- **Bubbles**: `data/Processed Data/Bubbles/04072021_Huron_10_2mm_Rep2_AD_000003_292.tif`

---

## test_trainer.py — Trainer

Tests the training loop, prediction, and evaluation.

### TestTrainerStatics (2 tests)

| Test | What it verifies |
|------|-----------------|
| `test_compute_metrics` | Perfect predictions yield accuracy=1.0 and F1=1.0 |
| `test_clip_gradients` | After clipping, gradient norm does not exceed `max_norm` |

### TestTrainerFit (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_fit_one_epoch` | Training for 1 epoch populates `train_loss` and `valid_loss` |
| `test_model_saving` | After training, `best_model.pt` and `training_info.json` are saved to `model_dir` |
| `test_early_stopping` | With `patience=1` and an impossibly high `delta`, training stops early (within 3 epochs) |

### TestTrainerPredict (2 tests)

| Test | What it verifies |
|------|-----------------|
| `test_predict_returns_results` | `predict()` returns a dict with keys: `predictions`, `targets`, `pred_paths`, `level_results`, `mismatch_results` |
| `test_predict_save` | With `save=True`, `predictions.npz` and `eval_results.json` are written to `model_dir` |

### TestCountMismatches (1 test)

| Test | What it verifies |
|------|-----------------|
| `test_clean_paths` | Paths from `model.predict()` are structurally valid (0 mismatches, `passed=True`) |

---

## test_extra_functions.py — set_seed, Visualize

### TestSetSeed (3 tests)

| Test | What it verifies |
|------|-----------------|
| `test_reproducibility` | Same seed produces identical `torch.randn` output |
| `test_different_seeds` | Different seeds produce different output |
| `test_numpy_reproducibility` | Same seed produces identical `np.random.rand` output |

### TestVisualize (2 tests)

Uses a temporary directory with mock `training_info.json`, `eval_results.json`, and `predictions.npz`.

| Test | What it verifies |
|------|-----------------|
| `test_visualize_init` | `Visualize` loads all three artifact files without error |
| `test_plot_train_runs` | `plot_train()` executes without raising (uses `matplotlib.use("Agg")` for headless rendering) |

---

## test_integration.py — End-to-End Pipeline

A single comprehensive test (`test_full_pipeline`) that exercises the entire workflow:

| Step | What it does |
|------|-------------|
| 1. Data creation | Creates 90 synthetic `.tif` images (3 classes x 30 each) in a temp directory |
| 2. Dataset construction | Builds `ImageDataset` → `HierImageDataset` with a flat hierarchy |
| 3. Data splitting | Stratified train/val/test split + DataLoader creation (batch_size=8) |
| 4. Model instantiation | Creates `HierRouteNet` with `expert_type="cnn"` |
| 5. Training | Runs `Trainer.fit()` for 2 epochs on CPU |
| 6. Prediction | Runs `Trainer.predict()` on the test set with `save=True` |
| 7. Verification | Checks predictions shape, structural consistency, and saved artifacts |
| 8. Visualization | Loads `Visualize` and calls `plot_train()` to verify plotting works |

**Assertions verified:**
- Training produces 2 epochs of metrics
- `best_model.pt` and `training_info.json` are saved
- Prediction count matches test set size
- All predicted paths are structurally valid (0 mismatches)
- `predictions.npz` and `eval_results.json` are saved
- `Visualize` loads artifacts and renders plots without error

---

## Test Count Summary

| File | Tests |
|------|-------|
| `test_hierarchy.py` | 21 |
| `test_model.py` | 25 |
| `test_trainer.py` | 8 |
| `test_extra_functions.py` | 5 |
| `test_integration.py` | 1 |
| **Total** | **60** |

Note: Some tests are parametrized (e.g., `test_forward_output_shapes` runs once per expert type), so the actual number of test cases executed by pytest may be higher.
