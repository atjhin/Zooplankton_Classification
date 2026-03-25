# HierRouteNet — Hierarchical Zooplankton Classification

HierRouteNet is a hierarchical image classification system for zooplankton species identification. It models biological taxonomy as a tree and routes each image from coarse to fine categories using a set of per-node expert classifiers, each trained to distinguish the children of one internal node. Predictions are guaranteed to follow a valid path through the hierarchy.

---

## Datasets

Two independent datasets are supported:

### Zooplankton-MNR (Freshwater)
- **Source**: Ontario Ministry of Natural Resources
- **Images**: 64×64 grayscale TIFF
- **Classes**: 13 leaf species across 3 hierarchy levels
- **Max samples**: 6,000 per class

### Plankton-WHOI (Marine)
- **Source**: Woods Hole Oceanographic Institution
- **Images**: 64×64 grayscale PNG, sourced from years 2006–2014
- **Classes**: 16 leaf species across 3 hierarchy levels (Guinardia sub-species merged)
- **Max samples**: 6,000 per class

Data paths are configured in `hierroute/constants.py`.

---

## Hierarchy

### MNR — 3-level taxonomy

```
root
├── Zoop-yes
│   ├── Copepoda
│   │   ├── Calanoid
│   │   ├── Cyclopoid
│   │   ├── Harpacticoid
│   │   └── Nauplius_Copepod
│   ├── Cladocera
│   │   ├── Bosminidae
│   │   └── Daphnia
│   └── Rotifer
└── Zoop-No
    ├── Bubbles
    ├── Exoskeleton
    ├── Fiber
    │   ├── Fiber_Hairlike
    │   └── Fiber_Squiggly
    └── Plant_Matter
```

### WHOI — 3-level taxonomy

```
root
├── Colonial
│   ├── C-Spines
│   │   ├── Chaetoceros
│   │   ├── Lauderia
│   │   └── Asterionellopsis
│   └── C-NoSpines
│       ├── Pseudonitzschia
│       ├── Leptocylindrus
│       ├── Eucampia
│       ├── Skeletonema
│       ├── Dactyliosolen
│       ├── Thalassiosira
│       ├── Guinardia
│       └── Cerataulina
└── Unicellular
    ├── U-Spines
    │   ├── Corethron
    │   └── Ditylum
    └── U-NoSpines
        ├── Cylindrotheca
        ├── Coscinodiscus
        └── Ephemera
```

---

## Model Architecture

**Source**: `hierroute/model.py`

HierRouteNet consists of two components:

### 1. Shared Backbone

A single convolutional or transformer backbone processes each input image and produces a feature vector shared across all expert classifiers.

| Backbone | Feature Dim | Params | GFLOPs |
|---|---|---|---|
| `efficientnet_b0` | 1280 | 5.3M | 0.4 |
| `swin_t` | 768 | 28.3M | 4.5 |
| `swin_s` | 768 | 49.6M | 8.7 |

All backbones are initialised from ImageNet pretrained weights.

### 2. Expert Classifiers

One expert is assigned to each internal node in the hierarchy. Each expert receives the shared feature vector and outputs a probability distribution over that node's children.

| Expert Type | Architecture |
|---|---|
| `linear` | `Linear(feature_dim, num_children)` |
| `mlp` | `Linear → ReLU → Linear` (hidden dim = 2× feature_dim) |
| `cnn` | `Conv2d → ReLU → Conv2d → ReLU → Flatten → LazyLinear` |

### Inference

**Soft routing** (training): the probability of each leaf is the product of all conditional probabilities along its root-to-leaf path:

```
P(leaf) = P(child_1 | root) × P(child_2 | node_1) × ... × P(leaf | parent)
```

**Hard routing** (test time): greedy argmax decisions at each node, guaranteeing every prediction follows a valid path in the hierarchy (structural consistency).

### Loss

Binary cross-entropy (or Focal Loss with γ=2) applied only to leaf nodes, using the soft path-product probabilities as inputs and one-hot leaf targets.

---

## Training

**Source**: `hierroute/trainer.py`

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3×10⁻⁴ |
| Scheduler | CosineAnnealingLR (optional) |
| Early stopping | patience=5, δ=0.001 |
| Batch size | 64 |
| Max epochs | 40 |
| Gradient clipping | max_norm=1.0 |
| Data split | 70% train / 10% val / 20% test |
| Random seed | 666 |

**Augmentation** (applied to training split only):
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation(180°)
- Padding(5 px)
- Resize to target resolution

---

## Evaluation

Metrics are computed independently at each hierarchy level:

- **Accuracy** — fraction of correctly classified samples
- **Macro F1, Precision, Recall** — unweighted mean across classes
- **Per-class breakdown** — accuracy, F1, precision, recall, and sample count for every class
- **Structural consistency** — verifies that every predicted leaf can be reached from the root via the predicted intermediate nodes; any violation is reported as a mismatch

Results are saved to `eval_results.json` in each model directory.

---

## Project Structure

```
Zooplankton_Classification/
├── hierroute/
│   ├── __init__.py
│   ├── constants.py          # Class lists, hierarchy adjacency graphs, data paths
│   ├── hierarchy.py          # Node and Hierarchy tree data structures
│   ├── model.py              # HierRouteNet, Expert, FocalLoss
│   ├── trainer.py            # Training loop, validation, early stopping, evaluation
│   ├── data_setup.py         # ImageDataset, HierImageDataset
│   └── extra_functions.py    # Visualize class, set_seed utility
├── training_result/
│   ├── mnr/                  # One subdirectory per trained model
│   │   ├── <backbone>_<expert>/
│   │   │   ├── best_model.pt
│   │   │   ├── training_info.json
│   │   │   ├── eval_results.json
│   │   │   └── predictions.npz
│   │   └── ...
│   └── whoi/
├── tests/                    # Unit and integration tests
├── hardrouter.ipynb          # End-to-end training and evaluation notebook (MNR)
├── WHOI.ipynb                # End-to-end training and evaluation notebook (WHOI)
├── train_densenet121.py      # BCNN baseline training script (MNR)
├── train_densenet121_whoi.py # BCNN baseline training script (WHOI)
└── backbone_comparison.md    # Architecture comparison table
```

---

## Installation

```bash
pip install torch torchvision numpy scikit-learn seaborn pillow matplotlib
```

No `requirements.txt` is provided — dependencies are declared directly in code imports.

---

## Usage

### 1. Data Loading

```python
from hierroute.constants import whoi_adjacency_graph_s, SEED
from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.extra_functions import set_seed

set_seed(SEED)

dataset = ImageDataset(
    data_directory      = '/path/to/WHOI-Plankton',
    data_subdirectories = ['2006', '2007', ...],
    class_names         = SELECTED_CLASSES,
    max_class_size      = 6000,
    image_resolution    = 64,
    format_file         = '.png',
    seed                = SEED,
)

hier_dataset = HierImageDataset(
    base_dataset   = dataset,
    adjacency_graph = whoi_adjacency_graph_s,
    levels         = 3,
    leaves_only    = True,
)
```

### 2. Build the Model

```python
from hierroute import HierRouteNet

model = HierRouteNet(
    hierarchy      = hier_dataset.hierarchy,
    label_to_id    = hier_dataset.label_to_ids,
    backbone       = 'efficientnet_b0',   # or 'swin_t'
    expert_type    = 'mlp',               # or 'linear', 'cnn'
    freeze_backbone = False,
)
```

### 3. Train

```python
from hierroute import Trainer
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.Pad(padding=5, fill=0),
    transforms.Resize((65, 65)),
    transforms.ToTensor(),
])

hier_dataset.append_image_transforms(train_transforms, replace=True)
train_split, val_split, test_split = hier_dataset.split_train_test_val(0.7, 0.1, 0.2)
train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
    batch_size=64,
    train_indices=train_split,
    val_indices=val_split,
    test_indices=test_split,
)

trainer = Trainer(
    learning_rate = 3e-4,
    max_epochs    = 40,
    device        = device,
    model_dir     = 'training_result/whoi/efficientnet_b0_mlp',
)
trainer.fit(model, train_loader, val_loader, patience=5, delta=0.001)
```

### 4. Predict & Evaluate

```python
results = trainer.predict(model, test_loader, save=True)
# Saves eval_results.json and predictions.npz to model_dir
```

### 5. Visualise Results

```python
from hierroute.extra_functions import Visualize

vis = Visualize('training_result/whoi/efficientnet_b0_mlp')
vis.plot_train()                   # Loss / Accuracy / F1 curves
vis.plot_pred()                    # Confusion matrices + per-class bar charts
vis.plot_level_comparison()        # Cross-level Acc/F1/Prec/Rec overview
vis.plot_class_size_vs_accuracy()  # Sample count vs. accuracy scatter
```

### 6. Compare Models

```python
from training_result.result import get_results_table, plot_results_table, plot_confusion_matrix

df  = get_results_table('whoi')           # DataFrame of all runs
fig = plot_results_table('whoi', save_path='results_whoi.png')
fig = plot_confusion_matrix('whoi/swin_t_mlp', save_path='cm.png')
```

---

## Reproducibility

All experiments use `set_seed(666)`, which seeds Python, NumPy, PyTorch, and CuDNN determinism flags. Device support: Apple Silicon MPS, CUDA GPU, CPU.
