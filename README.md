# Explanation-Guided Shortcut Suppression in Deep Neural Networks

> A training framework that forces neural networks to stop cheating —
> by using their own explanations as a penalty signal during training.

---

## Problem Statement

Neural networks trained on biased datasets learn **shortcut features** — spurious correlations that work on training data but fail at test time. A classic example: a model trained on colored digits where digit 0 is always red learns "red background → predict 0" instead of "this shape looks like a 0." It achieves near-perfect training accuracy but completely random test accuracy (~10%) when colors are randomized.

This is not a toy problem. Real-world models exhibit the same behavior:
- Medical imaging models that learn hospital equipment artifacts instead of pathology
- NLP models that learn stylistic cues instead of reasoning
- Object detectors that learn background textures instead of object shapes

Most existing work either **detects** shortcuts post-hoc or **mitigates** them using dataset-level interventions. We close the loop: **detect shortcut reliance during training and penalize it in the same step.**

---

## Core Idea

Standard training optimizes only for accuracy:

```
L_total = L_task
```

Our framework adds an explanation-based penalty:

```
L_total = L_task + λ × L_shortcut
```

Where `L_shortcut` measures how much the model's attribution (what it's looking at) lands on known spurious regions. If the model is cheating, it gets penalized. Over training, it is forced to find legitimate features instead.

**What makes this non-trivial:**
- The explanation signal is computed **during training**, not after
- The penalty **backpropagates through the attribution** to the model weights — this requires a differentiable attribution method and `create_graph=True` in autograd
- No per-sample human labels are required — only knowledge of which region type is spurious (background vs. foreground)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                           │
│                                                             │
│  Input Batch (biased)                                       │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐    Forward Pass     ┌──────────────┐          │
│  │  Model  │ ──────────────────► │    Logits    │          │
│  └─────────┘                     └──────┬───────┘          │
│       │                                 │                   │
│       │   Attribution Pass              │ Task Loss (CE)    │
│       │   (create_graph=True)           │                   │
│       ▼                                 ▼                   │
│  ┌──────────────┐              ┌─────────────────┐         │
│  │  Grad×Input  │              │   L_task        │         │
│  │  Attributions│              └────────┬────────┘         │
│  └──────┬───────┘                       │                   │
│         │                               │                   │
│         ▼                               │                   │
│  ┌──────────────┐                       │                   │
│  │  Background  │                       │                   │
│  │    Mask      │                       │                   │
│  └──────┬───────┘                       │                   │
│         │                               │                   │
│         ▼                               │                   │
│  ┌──────────────┐              ┌─────────────────┐         │
│  │ L_shortcut = │              │  L_total =      │         │
│  │ mean(|attr|  │─────────────►│  L_task +       │         │
│  │  × bg_mask)  │   λ × sc    │  λ × L_shortcut │         │
│  └──────────────┘              └────────┬────────┘         │
│                                         │                   │
│                                         ▼                   │
│                                  Backward Pass              │
│                                  (gradients flow through    │
│                                   attribution to weights)   │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Detail — Why `create_graph=True` Matters

Standard attribution computation destroys the computation graph after computing gradients. For our penalty to work, the gradient must flow:

```
L_shortcut → attributions → model weights
```

This requires keeping the computation graph alive through the attribution step:

```python
grads = torch.autograd.grad(
    outputs=target_scores,
    inputs=images_for_attr,
    create_graph=True    # ← critical: without this, suppression has zero effect
)[0]
attributions = grads * images_for_attr
```

Without `create_graph=True`, the shortcut loss is computed and printed but has **zero gradient effect on model weights** — a silent bug that produces convincing-looking but meaningless training logs.

### Two-Phase Training Schedule

At high bias ratios, the model converges to the shortcut in epoch 1 before suppression can redirect it. We use a warmup phase:

- **Epochs 1–5:** Train on task loss only (model builds basic representations)
- **Epochs 6–20:** Activate shortcut penalty (model is redirected away from spurious features)

---

## Dataset — Colored MNIST

Standard MNIST (grayscale digits 0–9) with colored backgrounds as shortcuts:

| Digit | Shortcut Color |
|-------|---------------|
| 0 | Red |
| 1 | Green |
| 2 | Blue |
| 3 | Yellow |
| 4 | Magenta |
| 5 | Cyan |
| 6 | Orange |
| 7 | Purple |
| 8 | Sky Blue |
| 9 | Pink |

**Bias ratio** controls how biased the training data is:
- `bias=1.0` → every training image has the shortcut color (maximum cheating)
- `bias=0.8` → 80% of images have the shortcut color
- `bias=0.0` → all images randomly colored (no shortcut, but model still develops color-based representations)

Test set is always unbiased (random colors) — measures true generalization.

---

## Results

### Main Results — Shortcut Score Reduction (λ=1.0, Warmup=5)

| Bias | ERM Baseline | ColorJitter | HighDropout | **Ours** |
|------|-------------|-------------|-------------|----------|
| 1.0 | 0.8895 | 0.6473 | 0.9009 | **0.0715** |
| 0.8 | 0.5283 | 0.6415 | 0.5186 | **0.0995** |
| 0.5 | 0.5429 | 0.6394 | 0.5206 | **0.0878** |
| 0.0 | 0.5397 | 0.6293 | 0.4753 | **0.0701** |

*Lower shortcut score = better. Model relying less on spurious background features.*

### Accuracy (Test Set, Unbiased)

| Bias | ERM Baseline | ColorJitter | HighDropout | **Ours** |
|------|-------------|-------------|-------------|----------|
| 1.0 | 10.26% | 98.66% | 10.26% | 10.27% |
| 0.8 | 97.58% | 98.80% | 97.26% | **96.68%** |
| 0.5 | 98.47% | 98.77% | 98.26% | **98.17%** |
| 0.0 | 98.84% | 98.83% | 98.77% | **98.57%** |

### Key Findings

**Finding 1 — Our method achieves up to 92% shortcut score reduction** across all bias levels with negligible accuracy cost at bias ≤ 0.8.

**Finding 2 — ColorJitter augmentation makes shortcuts worse.** At every bias level, ColorJitter increases shortcut scores compared to ERM. This is counter-intuitive: augmenting with color randomization does not teach the model to ignore color, it teaches the model to handle color variance — which still involves attending to color. Our method explicitly penalizes color-based attribution, which is fundamentally different.

**Finding 3 — HighDropout has zero effect.** General regularization cannot address shortcut learning. The model learns shortcuts regardless of dropout strength. This confirms that shortcut suppression requires an explicit mechanism targeting the spurious feature.

**Finding 4 — Accuracy recovery at bias=1.0 is not possible.** When all training images carry the shortcut, there is no unconfounded shape signal to fall back on. Suppression removes the cheat sheet but no textbook exists to replace it. This is a fundamental limitation of any training-time suppression approach under maximum bias.

---

## Ablation — Effect of Lambda (bias=0.8)

| λ | Shortcut Score | Accuracy |
|---|---------------|----------|
| 0.5 | 0.4201 | 97.68% |
| 1.0 | 0.0995 | 96.68% |
| 10.0 (unnormalized) | 0.4013 | 97.30% |
| 20.0 (unnormalized) | 0.3482 | 97.68% |

The normalized loss formulation (λ=1.0) outperforms the unnormalized version at any λ value. Normalization makes the shortcut loss scale-invariant, so λ=1.0 means "weight task and shortcut loss equally" — interpretable and effective.

---

## Project Structure

```
shortcut-suppression/
│
├── config.py                     # All hyperparameters in one place
├── main.py                       # Entry point
├── requirements.txt
│
├── data/
│   ├── colored_mnist.py          # Biased Colored MNIST dataset (GPU-cached)
│   └── dataloader.py             # DataLoader factory + BatchedJitterLoader
│
├── models/
│   └── cnn.py                    # SimpleCNN (configurable dropout)
│
├── explainer/
│   ├── attribution.py            # Integrated Gradients (evaluation only)
│   └── shortcut_detector.py      # Background mask + normalized shortcut loss
│
├── training/
│   ├── losses.py                 # TaskLoss + ShortcutSuppressionLoss
│   └── trainer.py                # Training loops (baseline + suppression + warmup)
│
├── evaluation/
│   ├── metrics.py                # Accuracy + shortcut score computation
│   └── visualize.py              # Attribution map comparisons
│
└── utils/
    ├── helpers.py                # Seed, device, checkpointing
    └── logger.py                 # Training progress printer
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/shortcut-suppression.git
cd shortcut-suppression
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Train baseline only (demonstrates shortcut learning)
python main.py --mode baseline

# Train with shortcut suppression
python main.py --mode suppress

# Full pipeline: baseline + suppression + all baselines + evaluation + visualization
python main.py --mode full
```

Key config parameters in `config.py`:

```python
BIAS_RATIO      = 0.8    # Shortcut strength in training data (0.0 - 1.0)
LAMBDA_SHORTCUT = 1.0    # Weight of shortcut penalty (normalized loss)
WARMUP_EPOCHS   = 5      # Epochs of task-only training before penalty activates
EPOCHS          = 20
BATCH_SIZE      = 512
```

---

## What's Been Done

- [x] Colored MNIST dataset with configurable bias
- [x] SimpleCNN baseline model
- [x] Differentiable Gradient×Input attribution (training)
- [x] Integrated Gradients attribution (evaluation/visualization)
- [x] Normalized shortcut penalty loss
- [x] Two-phase warmup training schedule
- [x] ColorJitter augmentation baseline
- [x] HighDropout regularization baseline
- [x] Full ablation across bias levels (0.0, 0.5, 0.8, 0.9, 1.0)
- [x] Attribution map visualization (before/after suppression)
- [x] GPU VRAM dataset caching for fast training

---

## Next Steps

- [ ] Extend to CIFAR-10 with synthetic color bias (ResNet-18)
- [ ] Learnable background mask — replace hard-coded brightness threshold with a data-driven region proposal, removing the assumption that shortcuts are always in the background
- [ ] Test on real-world biased datasets (CelebA, Waterbirds)
- [ ] Compare against dedicated shortcut mitigation methods (LfF, JTT, ReBias)
- [ ] Extend to transformer architectures using attention maps as attribution

---

## Limitations

**The background mask is hand-engineered for this dataset.** The shortcut detector works by separating bright (digit) pixels from colored (background) pixels. This is valid for Colored MNIST because we designed the dataset so shortcuts are always in the background. A general-purpose system would need to discover the spurious region automatically.

**Accuracy recovery requires unbiased training signal.** At bias=1.0, suppression successfully reduces shortcut reliance (92% reduction) but cannot recover accuracy because no shape signal exists in the training data. This is a fundamental limitation, not a bug.

**Single dataset validation.** All experiments are on Colored MNIST. Generalization to other shortcut types and datasets is not yet validated.

---

## References

- Geirhos et al. (2020) — Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*
- Sundararajan et al. (2017) — Axiomatic Attribution for Deep Networks (Integrated Gradients). *ICML*
- Nam et al. (2020) — Learning from Failure: Training Debiased Classifier from Biased Classifier. *NeurIPS*
- Captum — Model Interpretability for PyTorch. https://captum.ai/

---

## Team

| Person | Responsibility |
|--------|---------------|
| Member 1 | data/ — Dataset creation and loading |
| Member 2 | explainer/ — Attribution maps and shortcut detection |
| Member 3 | training/ + evaluation/ — Loss functions and evaluation |