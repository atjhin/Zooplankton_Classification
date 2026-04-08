# HierRouteNet — Hierarchical Zooplankton Classification

HierRouteNet is a modular hierarchical image classification framework for zooplankton and plankton species identification. It models biological taxonomy as a tree and routes each image from coarse to fine categories using a set of per-node expert classifiers, each trained to distinguish the children of one internal node. Predictions are guaranteed to follow a valid path through the hierarchy (structural consistency).

---

## Datasets

Two independent datasets are supported:

### Zooplankton-MNR (Freshwater)
- **Source**: Ontario Ministry of Natural Resources
- **Images**: 64×64 grayscale TIFF
- **Classes**: 13 leaf species across 3 hierarchy levels (18 nodes total)
- **Samples**: ~48,600 after filtering to leaf nodes (max 6,000 per class)

### Plankton-WHOI (Marine)
- **Source**: Woods Hole Oceanographic Institution (2006–2014)
- **Images**: Variable-size grayscale PNG
- **Classes**: 16 leaf species across 3 hierarchy levels (23 nodes total, Guinardia sub-species merged)
- **Samples**: ~69,150 (max 6,000 per class)

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

### WHOI — 3-level taxonomy (morphological)

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
| `linear` | `Linear(feature_dim → num_children)` |
| `mlp` | `Linear → ReLU → Linear` (hidden dim = 2× feature_dim) |
| `cnn` | `Conv2d(1×1) → ReLU → Conv2d(2×2) → ReLU → Flatten → LazyLinear` |

### Routing

**Soft routing** (training): the probability of each leaf is the product of all conditional probabilities along its root-to-leaf path:

```
P(leaf) = P(child_1 | root) × P(child_2 | node_1) × ... × P(leaf | parent)
```

**Hard routing** (inference): greedy argmax decisions at each node, guaranteeing structural consistency.

### Loss

Binary cross-entropy (or Focal Loss with γ=2) applied only at leaf node positions, using soft path-product probabilities as inputs and one-hot leaf targets.

---

## Results

### Zooplankton-MNR

| Backbone | Expert | L1 Acc | L2 Acc | L3 Acc |
|---|---|---|---|---|
| EfficientNet-B0 | linear | 0.9974 | 0.9842 | 0.9381 |
| EfficientNet-B0 | mlp | 0.9978 | 0.9868 | 0.9444 |
| EfficientNet-B0 | cnn | 0.9979 | 0.9871 | 0.9466 |
| Swin-T | linear | 0.9980 | 0.9887 | 0.9558 |
| Swin-T | mlp | 0.9980 | 0.9888 | 0.9558 |
| **Swin-T** | **cnn** | **0.9981** | **0.9893** | **0.9591** |

### Plankton-WHOI

| Backbone | Expert | L1 Acc | L2 Acc | L3 Acc |
|---|---|---|---|---|
| EfficientNet-B0 | linear | 0.9789 | 0.9642 | 0.8968 |
| EfficientNet-B0 | mlp | 0.9814 | 0.9673 | 0.9039 |
| EfficientNet-B0 | cnn | 0.9816 | 0.9680 | 0.9059 |
| Swin-T | linear | 0.9836 | 0.9727 | 0.9179 |
| Swin-T | mlp | 0.9849 | 0.9748 | 0.9221 |
| **Swin-T** | **cnn** | **0.9850** | **0.9751** | **0.9236** |

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

## Project Structure

```
Zooplankton_Classification/
├── hierroute/
│   ├── __init__.py
│   ├── constants.py              # Class lists, hierarchy adjacency graphs, data paths
│   ├── hierarchy.py              # Node and Hierarchy tree data structures
│   ├── model.py                  # HierRouteNet, Expert, FocalLoss
│   ├── trainer.py                # Training loop, validation, early stopping, evaluation
│   ├── data_setup.py             # ImageDataset, HierImageDataset
│   └── extra_functions.py        # Visualize, set_seed, visualize_swin_attention, visualize_gradcam
├── training_result/
│   ├── result.py                 # get_results_table, plot_results_table, plot_confusion_matrix
│   ├── mnr/                      # One subdirectory per trained MNR model
│   │   └── <backbone>_<expert>/
│   │       ├── best_model.pt
│   │       ├── training_info.json
│   │       ├── eval_results.json
│   │       └── predictions.npz
│   └── whoi/                     # One subdirectory per trained WHOI model
├── runs/
│   ├── mnr.ipynb                 # End-to-end training and evaluation notebook (MNR)
│   ├── whoi.ipynb                # End-to-end training and evaluation notebook (WHOI)
│   ├── train_bcnn_baseline_mnr.py   # BCNN baseline training script (MNR)
│   └── train_bcnn_baseline_whoi.py  # BCNN baseline training script (WHOI)
├── tests/                        # Unit and integration tests (pytest)
├── .env.example                  # Template for local data path configuration
└── .env                          # Local data path (gitignored — not pushed)
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision numpy scikit-learn seaborn pillow matplotlib python-dotenv
```

### 2. Configure the data path

Copy `.env.example` to `.env` and set the path to your local data directory:

```bash
cp .env.example .env
```

Edit `.env`:
```
ZOOPLANKTON_DATA_DIR=/path/to/your/data
```

The data directory should contain the dataset folders (`Processed Data/` for MNR, `WHOI-Plankton/` for WHOI). The `.env` file is gitignored and will never be pushed.

### 3. Load the environment in notebooks

Add this at the top of any notebook, before importing `hierroute`:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Usage

### 1. Data loading

```python
from dotenv import load_dotenv
load_dotenv()

from hierroute.constants import hier_adjacency_graph, data_directory, data_subdirectories, SEED
from hierroute.data_setup import ImageDataset, HierImageDataset
from hierroute.extra_functions import set_seed

set_seed(SEED)

dataset = ImageDataset(
    data_directory      = data_directory,
    data_subdirectories = data_subdirectories,
    class_names         = ZOOPLANKTON_CLASSES,
    max_class_size      = 6000,
    image_resolution    = 64,
    format_file         = '.tif',
    seed                = SEED,
)

hier_dataset = HierImageDataset(
    base_dataset    = dataset,
    adjacency_graph = hier_adjacency_graph,
    levels          = 3,
    leaves_only     = True,
)
```

For WHOI with merged Guinardia sub-species:

```python
dataset = ImageDataset(
    data_directory      = f'{data_directory}/WHOI-Plankton',
    data_subdirectories = ['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014'],
    class_names         = SELECTED_CLASSES,
    max_class_size      = 6000,
    format_file         = '.png',
    class_folder_map    = {'Guinardia': ['Guinardia_delicatula', 'Guinardia_flaccida', 'Guinardia_striata']},
    seed                = SEED,
)
```

### 2. Build the model

```python
from hierroute import HierRouteNet

model = HierRouteNet(
    hierarchy       = hier_dataset.hierarchy,
    label_to_id     = hier_dataset.label_to_ids,
    backbone        = 'swin_t',    # or 'efficientnet_b0', 'swin_s'
    expert_type     = 'cnn',       # or 'linear', 'mlp'
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
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

hier_dataset.append_image_transforms(train_transforms, replace=True)
train_split, val_split, test_split = hier_dataset.split_train_test_val(0.7, 0.1, 0.2)
train_loader, val_loader, test_loader = hier_dataset.create_dataloaders(
    batch_size=64,
    train_indices=train_split,
    val_indices=val_split,
    test_indices=test_split,
    balanced=True,
)

trainer = Trainer(
    learning_rate = 3e-4,
    max_epochs    = 40,
    device        = 'cuda',
    model_dir     = 'training_result/mnr/swin_t_cnn',
)
trainer.fit(model, train_loader, val_loader, scheduler=True, patience=5, delta=0.001)
```

### 4. Predict and evaluate

```python
results = trainer.predict(model, test_loader, save=True)
# Saves eval_results.json and predictions.npz to model_dir
```

### 5. Load a saved checkpoint

```python
model = HierRouteNet(
    hierarchy       = hier_dataset.hierarchy,
    label_to_id     = hier_dataset.label_to_ids,
    backbone        = 'swin_t',
    expert_type     = 'cnn',
    checkpoint_dir  = 'training_result/mnr/swin_t_cnn',
)
```

### 6. Visualise training and predictions

```python
from hierroute import Visualize

vis = Visualize('training_result/mnr/swin_t_cnn')
vis.plot_train()                   # Loss / Accuracy / F1 curves
vis.plot_pred()                    # Per-level confusion matrices + per-class bar charts
vis.plot_level_comparison()        # Cross-level Acc/F1/Prec/Rec overview
vis.plot_class_size_vs_accuracy()  # Sample count vs. accuracy scatter
```

### 7. Compare all models

```python
from training_result.result import get_results_table, plot_results_table, plot_confusion_matrix

df  = get_results_table('mnr')
fig = plot_results_table('mnr', save_path='results_mnr.png')
fig = plot_confusion_matrix('mnr/swin_t_cnn', save_path='cm.png')
```

### 8. Attention visualisation (Swin-T only)

```python
from hierroute import visualize_swin_attention
from hierroute.extra_functions import visualize_gradcam

# Self-attention maps from stage 7
fig = visualize_swin_attention(model, img_tensor, stage=7, overlay=True, label='Calanoid')

# Grad-CAM weighted by hierarchy level
fig = visualize_gradcam(model, img_tensor, target_leaf='Calanoid',
                        weights=[1, 1, 1], stage=7)  # full path
fig = visualize_gradcam(model, img_tensor, target_leaf='Calanoid',
                        weights=[0, 0, 1], stage=7)  # species level only
```

---

## Tests

```bash
cd Zooplankton_Classification && python -m pytest tests/ -v
```

| File | Coverage |
|------|---------|
| `test_hierarchy.py` | `Node`, `Hierarchy` (21 tests) |
| `test_model.py` | `FocalLoss`, `Expert`, `HierRouteNet` (25 tests) |
| `test_trainer.py` | `Trainer.fit`, `predict`, `evaluate` (8 tests) |
| `test_extra_functions.py` | `set_seed`, `Visualize` (5 tests) |
| `test_integration.py` | Full end-to-end pipeline (1 test) |

---

## Reproducibility

All experiments use `set_seed(666)`, which seeds Python, NumPy, PyTorch CPU/GPU, and enables `cudnn.deterministic`. Device support: Apple Silicon MPS, CUDA GPU, CPU.
