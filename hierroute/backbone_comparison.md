# Backbone Comparison for Hierarchical Zooplankton Classification

## Model Overview

| Property | ResNet-50 | DenseNet-121 | EfficientNet-B0 | Swin-T |
|---|---|---|---|---|
| **Architecture Type** | Residual CNN | Dense CNN | Compound-scaled CNN | Vision Transformer |
| **Total Parameters** | 25.6M | 8.0M | 5.3M | 28.3M |
| **Feature Dimension** | 2048 | 1024 | 1280 | 768 |
| **ImageNet Top-1 Acc** | 76.1% | 74.4% | 77.1% | 81.3% |
| **ImageNet Top-5 Acc** | 92.9% | 91.9% | 93.3% | 95.5% |
| **Input Resolution** | 224 × 224 | 224 × 224 | 224 × 224 | 224 × 224 |
| **GFLOPs** | 4.1 | 4.1 | 0.4 | 4.5 |

## Architecture Details

| Property | ResNet-50 | DenseNet-121 | EfficientNet-B0 | Swin-T |
|---|---|---|---|---|
| **Core Block** | Bottleneck (1×1→3×3→1×1) | Dense Block (BN→ReLU→Conv) | MBConv + SE | Shifted Window Attention |
| **Skip Connections** | Additive residual | Dense concatenation | Additive residual + SE | Residual in transformer block |
| **Normalization** | BatchNorm | BatchNorm | BatchNorm | LayerNorm |
| **Activation** | ReLU | ReLU | SiLU (Swish) | GELU |
| **Downsampling** | Strided convolution | Transition layers (1×1 conv + AvgPool) | Strided depthwise conv | Patch merging |
| **Attention Mechanism** | None | None | Squeeze-and-Excitation | Multi-head self-attention (windowed) |
| **Depth (layers)** | 50 | 121 | 18 (MBConv blocks) | 12 transformer blocks (2+2+6+2) |
| **Width Progression** | 64→256→512→1024→2048 | 64→256→512→1024 | 32→16→24→40→80→112→192→320→1280 | 96→192→384→768 |

## Efficiency & Practical Considerations

| Property | ResNet-50 | DenseNet-121 | EfficientNet-B0 | Swin-T |
|---|---|---|---|---|
| **Memory Footprint** | Medium | High (dense concat) | Low | High (attention maps) |
| **Inference Speed (GPU)** | Fast | Moderate | Fast | Moderate |
| **Inference Speed (CPU)** | Fast | Slow (memory-bound) | Fast | Slow |
| **Training Stability** | High | High | High | Requires warmup |
| **Small Dataset Suitability** | Good (with pretrain) | Good (with pretrain) | Good (with pretrain) | Needs more data |
| **Fine-tuning Ease** | Simple | Simple | Simple | Needs LR tuning |

## Impact on HierRouteNet Expert Classifiers

Each expert is `Expert(feature_dim → num_children)`. The backbone choice changes every expert's input dimension:

| Expert | ResNet-50 | DenseNet-121 | EfficientNet-B0 | Swin-T |
|---|---|---|---|---|
| **Root** | 2048 → 2 | 1024 → 2 | 1280 → 2 | 768 → 2 |
| **Zoop-yes** | 2048 → 3 | 1024 → 3 | 1280 → 3 | 768 → 3 |
| **Zoop-No** | 2048 → 4 | 1024 → 4 | 1280 → 4 | 768 → 4 |
| **Copepoda** | 2048 → 4 | 1024 → 4 | 1280 → 4 | 768 → 4 |
| **Cladocera** | 2048 → 2 | 1024 → 2 | 1280 → 2 | 768 → 2 |
| **Fiber** | 2048 → 2 | 1024 → 2 | 1280 → 2 | 768 → 2 |
| **Expert Params (linear)** | 34,834 | 17,422 | 21,778 | 13,074 |

## Total Trainable Parameters (backbone + experts, linear experts)

| | ResNet-50 | DenseNet-121 | EfficientNet-B0 | Swin-T |
|---|---|---|---|---|
| **Backbone** | 25,557,032 | 6,953,856 | 5,288,548 | 28,288,354 |
| **Experts (linear)** | 34,834 | 17,422 | 21,778 | 13,074 |
| **Total** | ~25.6M | ~7.0M | ~5.3M | ~28.3M |
| **Frozen backbone** | 34,834 | 17,422 | 21,778 | 13,074 |

## Key Tradeoffs

| Consideration | Best Choice | Why |
|---|---|---|
| **Smallest model** | EfficientNet-B0 | 5.3M params, 0.4 GFLOPs — most efficient |
| **Highest ImageNet accuracy** | Swin-T | 81.3% top-1 — strongest pretrained features |
| **Best for small 64×64 images** | EfficientNet-B0 / ResNet-50 | CNNs handle low-res better than transformers; Swin-T's windowed attention has less to work with at 64×64 |
| **Fastest training** | EfficientNet-B0 | Fewest FLOPs by a large margin |
| **Feature richness** | ResNet-50 | 2048-dim gives experts the most information per sample |
| **Memory constrained** | EfficientNet-B0 | Lowest memory footprint |
| **Currently supported** | EfficientNet-B0, Swin-T | ResNet-50 and DenseNet-121 would need to be added to `model.py` |
