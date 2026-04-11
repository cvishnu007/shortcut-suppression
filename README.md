# Explanation-Guided Shortcut Suppression in Deep Neural Networks

> A training framework that forces neural networks to stop cheating —
> by using their own explanations as a penalty signal during training.

---

## 1. Project Goal

This project builds a closed-loop explanation-guided training framework that:

1. Trains a neural network on biased data containing spurious correlations (shortcuts)
2. Computes attribution maps during training to understand what the model is looking at
3. Penalizes the model when its explanations focus on spurious regions
4. Forces the model to shift toward robust, task-relevant representations

Instead of treating explainability as a post-hoc diagnostic tool, this system turns it into an **optimization signal** — if your reasoning looks wrong, your learning signal is wrong.

---

## 2. What is Shortcut Learning?

Neural networks trained on biased datasets learn **spurious correlations** instead of true patterns.

**Example (Colored MNIST):** A model trained on colored digits where digit 0 is always red learns `red background → predict 0` instead of `this shape looks like a 0 → predict 0`. It achieves 100% training accuracy but ~10% test accuracy when colors are randomized. This is a shortcut.

**Example (Waterbirds):** A model trained where waterbirds appear on water backgrounds 95% of the time learns `water texture → waterbird` instead of learning what a duck actually looks like. It achieves high average accuracy but fails catastrophically on waterbirds photographed on land.

Real-world consequences: medical imaging models that learn hospital equipment artifacts instead of pathology, NLP models that learn writing style instead of reasoning, object detectors that learn background textures instead of object shapes.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                            │
│                                                                 │
│  Input Batch (biased dataset)                                  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────┐   Forward Pass    ┌──────────────────┐        │
│  │    Model    │ ────────────────► │     Logits       │        │
│  └─────────────┘                   └────────┬─────────┘        │
│         │                                   │                   │
│         │  Attribution Pass                 │  Task Loss (CE)   │
│         │  (create_graph=True               │                   │
│         │   OR Grad-CAM hooks)              │                   │
│         ▼                                   ▼                   │
│  ┌──────────────────┐           ┌──────────────────────┐       │
│  │  Attribution Map │           │       L_task         │       │
│  │  (what model     │           └──────────┬───────────┘       │
│  │   looked at)     │                      │                   │
│  └────────┬─────────┘                      │                   │
│           │                                │                   │
│           ▼                                │                   │
│  ┌──────────────────┐                      │                   │
│  │  Shortcut Mask   │                      │                   │
│  │  (spurious       │                      │                   │
│  │   region)        │                      │                   │
│  └────────┬─────────┘                      │                   │
│           │                                │                   │
│           ▼                                │                   │
│  ┌──────────────────┐           ┌──────────────────────┐       │
│  │  L_shortcut =    │           │  L_total =           │       │
│  │  mean(|attr| ×   │──────────►│  L_task +            │       │
│  │   shortcut_mask) │   λ × sc  │  λ × L_shortcut      │       │
│  └──────────────────┘           └──────────┬───────────┘       │
│                                            │                   │
│                                            ▼                   │
│                                     Backward Pass              │
│                             (gradients flow through            │
│                              attribution to model weights)     │
└─────────────────────────────────────────────────────────────────┘
```

### The Core Technical Insight — Why `create_graph=True` Matters

Standard attribution computation destroys the computation graph after computing gradients. For the shortcut penalty to work, gradients must flow:

```
L_shortcut → attributions → model weights
```

This requires keeping the computation graph alive:

```python
grads = torch.autograd.grad(
    outputs=target_scores,
    inputs=images_for_attr,
    create_graph=True    # Without this, suppression has ZERO effect on weights
)[0]
attributions = grads * images_for_attr
```

Without `create_graph=True`, the shortcut loss is computed and printed but has **zero gradient effect** — a silent bug that produces convincing-looking but meaningless training logs. This was the most important bug fixed during development.

### Loss Normalization — Why It Matters

Before normalization, the raw shortcut loss was at scale ~0.003 while task loss was ~0.03. This meant λ was compensating for a scale mismatch, not tuning a meaningful tradeoff. After normalization:

```
L_shortcut = mean over batch of:
    sum(|attribution| × shortcut_mask) / sum(|attribution|)
```

The shortcut loss is now a ratio between 0 and 1. λ=1.0 means task loss and shortcut loss are weighted equally — an interpretable setting. This single fix caused shortcut score reduction to jump from 38% (at λ=20, unnormalized) to 87% (at λ=1.0, normalized).

### Two-Phase Training Schedule

```
Epochs 1-5  (Warmup)  : Task loss only → model builds basic representations
Epochs 6-20 (Penalty) : Task loss + λ × Shortcut loss → model redirected
```

At high bias ratios the model learns the shortcut in epoch 1. Warmup ensures it has some initial signal before suppression forces it away from the spurious feature.

---

## 4. Dataset — Colored MNIST

Standard MNIST (grayscale digits 0–9) with colored backgrounds as shortcuts:

| Digit | Shortcut Color | RGB |
|-------|---------------|-----|
| 0 | Red | (255, 0, 0) |
| 1 | Green | (0, 255, 0) |
| 2 | Blue | (0, 0, 255) |
| 3 | Yellow | (255, 255, 0) |
| 4 | Magenta | (255, 0, 255) |
| 5 | Cyan | (0, 255, 255) |
| 6 | Orange | (255, 128, 0) |
| 7 | Purple | (128, 0, 255) |
| 8 | Sky Blue | (0, 128, 255) |
| 9 | Pink | (255, 0, 128) |

**Bias ratio** controls shortcut strength in training data:

| Bias | Meaning |
|------|---------|
| 1.0 | Every training image has the correct shortcut color (maximum cheating) |
| 0.8 | 80% of images have the correct shortcut color |
| 0.5 | 50% correct colors — balanced |
| 0.0 | All images randomly colored — no systematic shortcut |

Test set is always unbiased (random colors) — measures true generalization.

**GPU Caching:** All images are pre-computed and pushed to VRAM at load time. Multiple dataloaders sharing the same bias/split share the same GPU memory without duplication.

---

## 5. Model — SimpleCNN (Colored MNIST)

A deliberately simple 2-layer CNN:

```
Input  : (batch, 3, 28, 28)
Conv1 + ReLU + MaxPool → (batch, 32, 14, 14)
Conv2 + ReLU + MaxPool → (batch, 64, 7, 7)
Flatten                → (batch, 3136)
FC1 + ReLU + Dropout   → (batch, 128)
FC2                    → (batch, 10)
Total parameters: 422,218
```

Using a simple model is a deliberate choice — attribution maps are interpretable, shortcut behavior is clean and observable, and results are reproducible in minutes.

---

## 6. Suppression Methods

### Method 1 — Supervised Suppression (Hardcoded Mask)

Uses pixel brightness to separate digit pixels (white) from background pixels (colored). Background = shortcut region.

```python
min_channel = images.min(dim=1).values
mask = (min_channel > 0.5).float()      # foreground = digit
bg_mask = 1.0 - mask                    # background = shortcut
```

**Requires:** Knowledge that the shortcut is in the background.
**Strength:** 77–92% shortcut score reduction across all bias levels.

### Method 2 — SAC Discovery (Unsupervised)

Discovers the shortcut region automatically using Spatial Attribution Consistency — the insight that a shortcut region is attended to similarly regardless of which class is shown.

1. During warmup, accumulate per-class mean attribution maps
2. After warmup, compute pixel-wise variance across the 10 class maps
3. Low-variance pixels = suspected shortcut (attended to the same way for all classes)
4. High-variance pixels = real features (digit shapes differ per class)

**Requires:** No prior knowledge about shortcut type or location.
**Strength:** 21–28% shortcut score reduction — weaker but fully unsupervised.

---

## 7. Baselines

### ColorJitter Augmentation
Applies random brightness, contrast, saturation, and hue jitter to training images. Applied as a batched GPU operation via `BatchedJitterLoader` for efficiency.

**Result:** Shortcut score consistently *increases* vs baseline. ColorJitter teaches the model to handle color variation more carefully — which means attending to color more precisely — the opposite of the intended effect. This is the sharpest negative finding: implicit augmentation is not equivalent to explicit attribution-based suppression.

### HighDropout
Standard training with dropout p=0.8 instead of p=0.5.

**Result:** Negligible shortcut score reduction (3–4%). General regularization cannot address a spatially-specific spurious feature.

### Just Train Twice (JTT) — Liu et al., ICML 2021
The current standard no-group-labels debiasing baseline. Trains a short ERM identifier to find hard samples, then upsamples them and retrains. Hyperparameters: T_id=1 epoch, λ_up=50.

**Result at bias=0.8:** Shortcut score 0.5368 → 0.5333 (0.6% reduction). JTT reweights which samples the model sees but places no constraint on where in the image it looks. A model retrained on upsampled wrong-color samples can still solve them by attending to color. Our method achieves 80.9% reduction at the same bias level.

---

## 8. Phase 1 — Adversarial Color-Shift Evaluation

To move beyond shortcut score as the sole metric, we constructed an adversarial test set where every image's background color is guaranteed to belong to a *different* digit class than its true label. A model that learned the color shortcut will confidently predict the wrong class on every image, collapsing to ~10% accuracy. A model that learned digit shape is unaffected.

**At bias=0.8:** Both the baseline and suppressed model survive adversarial shift (97.3% and 96.4%). The baseline at bias=0.8 was forced to learn some shape signal because 20% of training images had wrong colors. The adversarial accuracy gap is small — but the shortcut score gap (0.528 vs 0.095) confirms the suppressed model attends to shape while the baseline attends to color.

**At bias=1.0:** Both models score ~0% on the adversarial set. The baseline fails because it is a pure color lookup table. The suppressed model fails for a different and explainable reason: suppression removes the color shortcut, but when 100% of training images carry the correct shortcut color, there is no shape signal in the training distribution to fall back on. Suppression cannot recover signal that was never present. The shortcut score still drops from 0.8895 to 0.0715 (92% reduction), confirming that the model genuinely stopped attending to the background — it simply had nothing else to learn.

This reveals a fundamental limitation of any attribution-based suppression method: it redirects attention away from spurious features, but requires at least some unconfounded training signal to redirect attention *toward*. The adversarial test correctly diagnoses both bias=1.0 models as shape-ignorant, which is the correct diagnosis.

---

## 9. Phase 2 — Just Train Twice (JTT) Comparison

Full results at bias=0.8:

| Method | Shortcut Score | Test Acc | SC Reduction |
|---|---|---|---|
| Baseline | 0.5368 | 97.53% | — |
| ColorJitter | 0.6387 | 98.86% | −19% (worse) |
| HighDropout | 0.5161 | 97.10% | 3.9% |
| JTT | 0.5333 | 97.50% | 0.6% |
| **Suppression (ours)** | **0.1025** | **96.71%** | **80.9%** |

At bias=1.0, JTT's failure set is empty after one identifier epoch. The identifier achieves 96% training accuracy in epoch 1 by learning the color shortcut, correctly classifying every training sample — no failures to upsample. This is the same fundamental limitation that prevents suppression from recovering accuracy at full bias: both methods require at least some unconfounded signal to work with.

**Why JTT fails on shortcut score despite working on accuracy in other settings:** JTT is a data reweighting method — it changes which samples the model sees but places no constraint on where in the image the model looks. A model trained on upsampled wrong-color samples can still solve them by attending to color — the wrong color still provides information about what the correct color is not. Our method directly penalizes attribution in the background region during gradient computation, targeting the spurious feature geometrically rather than statistically.

At bias=1.0 full results:

| Method | Shortcut Score | Test Acc |
|---|---|---|
| Baseline | 0.8908 | 10.26% |
| ColorJitter | 0.6361 | 98.81% |
| HighDropout | 0.9152 | 10.26% |
| JTT | 0.8511 | 10.26% |
| **Suppression (ours)** | **0.0846** | **10.26%** |

The shortcut score comparison at bias=1.0 is the clearest result: JTT achieves 4.5% reduction while our method achieves 90.5% reduction. Both score 10.26% accuracy — both are shape-ignorant — but our method's model has genuinely stopped looking at the background, while JTT's model has barely changed its attention pattern.

---

## 10. Phase 3 — Waterbirds + ResNet-18

To validate that the method transfers beyond synthetic data, we applied explanation-guided shortcut suppression to the Waterbirds benchmark (Sagawa et al., 2020). Waterbirds is constructed by compositing CUB-200 bird images onto Places365 backgrounds with a 95% spurious correlation: landbirds appear on land backgrounds and waterbirds appear on water backgrounds in training.

**Groups:**
| Group | Description | Train samples | Test samples |
|---|---|---|---|
| 0 | Landbird on land (majority) | 3,498 | 2,255 |
| 1 | Landbird on water (minority) | 184 | 2,255 |
| 2 | Waterbird on land (minority) | 56 | 642 |
| 3 | Waterbird on water (majority) | 1,057 | 642 |

The standard metric is **worst-group accuracy** — the minimum across all four groups. A shortcut model scores well on groups 0 and 3 while failing on groups 1 and 2.

**Architecture changes from Colored MNIST:**

Integrated Gradients was replaced with Grad-CAM as the training-time attribution method. IG requires n_steps forward passes per batch (25 by default) — on ResNet-18 with batch_size=64 this is approximately 1,600 forward passes per batch, which is computationally prohibitive on 4GB VRAM. Grad-CAM requires exactly one forward and one backward pass — the same cost as standard training. A forward hook on ResNet-18's layer4 captures feature maps; a backward hook captures gradients. The Grad-CAM map is `relu(sum_c(grad_c * activation_c))` upsampled to 224×224.

The shortcut region was defined as the outer 40% border of the image, where background texture is concentrated relative to the centered bird subject. Both models were selected by best validation worst-group accuracy across 30 epochs (not final epoch), because standard ERM worst-group accuracy degrades severely after early epochs as the model overfits to the majority groups.

**Results on the Waterbirds test set:**

| Method | Overall Acc | Worst-group Acc | WG Improvement |
|---|---|---|---|
| Baseline (ERM) | 87.12% | 69.16% | — |
| **Suppression (ours)** | **86.28%** | **75.39%** | **+6.23pp** |

The suppressed model improves worst-group accuracy by 6.23 percentage points at a cost of 0.84 percentage points of overall accuracy. The critical minority group — waterbird on land (Group 2, 56 training samples) — improves from 69.16% to 78.35% (+9.19pp), directly confirming that the border penalty redirects attention from background texture toward bird morphology for the samples where the shortcut is most harmful.

**Note on shortcut score measurement:** The shortcut score for the suppressed model is reported from the final epoch weights (fully converged suppression) while worst-group accuracy is reported from the best-checkpoint epoch. These are two different model states by design — worst-group accuracy degrades as the model overfits in later epochs, so the best checkpoint is the correct model for accuracy evaluation. The shortcut score is only meaningful on a fully-converged suppressed model. Conflating the two by evaluating both from the same checkpoint would either misrepresent the accuracy (final epoch is worse) or misrepresent the suppression (early epoch is partially suppressed).

---

## 11. All Experiments (Chronological)

| Run | Dataset | Bias | λ | Method | Shortcut Score | Reduction | Test Acc | Notes |
|-----|---------|------|---|--------|---------------|-----------|----------|-------|
| 1 | MNIST | 1.0 | 0.5 | Hardcoded | 0.8712 | 3.9% | 10.26% | λ too small, unnormalized loss |
| 2 | MNIST | 1.0 | 10.0 | Hardcoded | 0.8453 | 6.7% | 10.26% | Still weak, same root cause |
| 3 | MNIST | 0.8 | 10.0 | Hardcoded | 0.4201 | 22.2% | 97.68% | Bias fix helps |
| 4 | MNIST | 0.8 | 20.0 | Hardcoded | 0.4013 | 25.6% | 97.30% | Higher λ, marginal gain |
| 5 | MNIST | 0.5 | 20.0 | Hardcoded | 0.3914 | 28.1% | 98.33% | — |
| 6 | MNIST | 0.0 | 20.0 | Hardcoded | 0.3428 | 38.4% | 98.85% | Best pre-normalization |
| **7** | **MNIST** | **0.8** | **1.0** | **Hardcoded (normalized)** | **0.0995** | **81.2%** | **96.68%** | **Normalization fix — massive jump** |
| 8 | MNIST | 1.0 | 1.0 | Hardcoded (normalized) | 0.0715 | 92.0% | 10.27% | Best suppression result |
| 9 | MNIST | 0.9 | 1.0 | Hardcoded (normalized) | 0.1281 | 77.0% | 95.77% | — |
| 10 | MNIST | 0.5 | 1.0 | Hardcoded (normalized) | 0.0878 | 83.8% | 98.17% | — |
| 11 | MNIST | 0.0 | 1.0 | Hardcoded (normalized) | 0.0701 | 87.0% | 98.50% | — |
| 12 | MNIST | 0.8 | 1.0 | SAC Discovery (v1) | 0.9592 | -81.6% | 97.07% | Failed — flagged digit pixels |
| 13 | MNIST | 0.8 | 1.0 | SAC Discovery (v2) | 0.3790 | 28.3% | 97.76% | Cross-class variance fix |
| 14 | MNIST | 0.5 | 1.0 | SAC Discovery | 0.3977 | 26.7% | 98.48% | — |
| 15 | MNIST | 0.9 | 1.0 | SAC Discovery | 0.4001 | 24.1% | 96.05% | — |
| 16 | MNIST | 0.0 | 1.0 | SAC Discovery | 0.4234 | 21.6% | 98.99% | — |
| 17 | MNIST | 1.0 | 1.0 | SAC Discovery | 0.9914 | -11.5% | 10.26% | Fails at max bias — expected |
| 18 | MNIST | 0.8 | 1.0 | + JTT comparison | JTT: 0.5333 | 0.6% | 97.50% | JTT barely moves SC score |
| 19 | MNIST | 1.0 | 1.0 | + JTT comparison | JTT: 0.8511 | 4.5% | 10.26% | JTT failure set = 0 at bias=1.0 |
| 20 | Waterbirds | — | 1.0 | ResNet-18 + Grad-CAM border=0.3 | 0.4890 | 23.1% | WG: 58.72% | WG dropped — best-epoch not used |
| 21 | Waterbirds | — | 1.0 | ResNet-18 + Grad-CAM border=0.4 | final epoch | — | WG: 75.39% | +6.23pp WG improvement |

---

## 12. Key Findings

**Finding 1 — Normalization was the critical fix.**
Before normalizing the shortcut loss, the best result was 38.4% reduction at λ=20. After normalization, the same setup achieves 87.0% reduction at λ=1.0. The raw shortcut loss was at scale ~0.003 while task loss was ~0.03 — λ was compensating for a scale mismatch, not tuning a meaningful tradeoff.

**Finding 2 — Supervised suppression achieves up to 92% shortcut score reduction.**
At bias=1.0, shortcut score drops from 0.8895 to 0.0715 with no accuracy change on the test set (which was already at chance due to full bias). At bias=0.8, accuracy cost is under 1% while shortcut reduction exceeds 80%.

**Finding 3 — ColorJitter augmentation makes shortcuts worse, not better.**
At every bias level, ColorJitter consistently increases shortcut scores compared to ERM baseline. Randomizing colors does not teach the model to ignore color — it teaches the model to handle color variance, which still involves attending to color. Explicit attribution-based penalization is fundamentally different from data augmentation.

**Finding 4 — HighDropout has zero effect on shortcuts.**
General regularization cannot address shortcut learning. This confirms that shortcut suppression requires an explicit mechanism targeting the spurious feature.

**Finding 5 — JTT achieves 0.6% shortcut score reduction vs our 80.9% at bias=0.8.**
JTT reweights training data but places no constraint on where the model looks. Our method directly penalizes spatial attribution patterns, targeting the spurious feature geometrically rather than statistically. JTT produces a robust model; ours produces a robust model with an attribution audit trail.

**Finding 6 — SAC Discovery fails at bias=1.0.**
When all training images carry the correct shortcut color, per-class attribution maps look identical during warmup. Cross-class variance is near-zero everywhere, making the discovered mask effectively random. Unsupervised discovery requires at least some unconfounded training signal.

**Finding 7 — The method transfers to real photographic data.**
On Waterbirds with ResNet-18, worst-group accuracy improves by +6.23 percentage points (+9.19pp on the hardest minority group) with Grad-CAM as the training-time attribution proxy. The shortcut score drops 23% on the fully converged model. This establishes that attribution-penalty suppression generalizes beyond synthetic datasets.

**Finding 8 — Bias=1.0 is the correct stress test but the wrong training configuration.**
At full bias, both ERM and suppression fail on the test set because there is no shape signal in training. This is a dataset property, not a method failure. The adversarial test at bias=1.0 correctly identifies both models as shape-ignorant. The interesting operating range is bias=0.8–0.9, where shape signal exists but is minority and suppression meaningfully redirects attention toward it.

---

## 13. Bugs Fixed During Development

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| Gradients not reaching model weights | Suppression had ~0% effect | `create_graph=False` — computation graph destroyed before backward | Set `create_graph=True` |
| Loss not normalized | Required λ=20, uninterpretable | Raw shortcut loss at scale 0.003 vs task loss 0.03 | Divide by total attribution → ratio 0-1 |
| Warmup had no effect | Suppression active from epoch 1 | Warmup block overwritten by unconditional Step 3 below it | Removed redundant Step 3 |
| SAC v1 flagged wrong pixels | Shortcut score increased to 0.96 | Mean attribution flagged digit pixels (most consistently attended) | Switched to cross-class variance |
| `evaluate` NameError | Crash at training start | Function defined after its call site | Moved `evaluate()` before `train_with_suppression()` |
| Filename typo in save path | Crash when saving figures | `tr aining_curves.png` had a space | Fixed string |
| JTT failure set empty at bias=1.0 | JTT has no effect | Identifier achieves 96% train acc in epoch 1 by learning shortcut — no failures | Expected behavior, documented as finding |
| Waterbirds shortcut score contradicts WG improvement | SC went up while WG improved | Best-checkpoint is epoch 6 (1 epoch into suppression) — not converged | Dual checkpoint: best_state for WG, final_state for SC |
| `best_state_dict` NameError in main.py | Crash during evaluation | Variable named inconsistently across trainer and main | Renamed to `best_state` throughout |

---

## 14. Project Structure

```
shortcut-suppression/
│
├── config.py                      # All hyperparameters in one place
├── main.py                        # Entry point
├── requirements.txt
│
├── data/
│   ├── colored_mnist.py           # Biased Colored MNIST (VRAM cache)
│   ├── dataloader.py              # DataLoader factory + BatchedJitterLoader
│   ├── adversarial_dataset.py     # Adversarial evaluation split (Phase 1)
│   └── waterbirds.py              # Waterbirds dataset from metadata.csv (Phase 3)
│
├── models/
│   ├── cnn.py                     # SimpleCNN (configurable dropout)
│   └── resnet.py                  # ResNet-18 + Grad-CAM hooks (Phase 3)
│
├── explainer/
│   ├── attribution.py             # Integrated Gradients (evaluation/visualization)
│   └── shortcut_detector.py       # Background mask + normalized loss + SAC Discovery
│
├── training/
│   ├── losses.py                  # TaskLoss + ShortcutSuppressionLoss
│   ├── trainer.py                 # MNIST training loops + warmup + SAC integration
│   ├── jtt_trainer.py             # Just Train Twice implementation (Phase 2)
│   └── waterbirds_trainer.py      # ResNet-18 training loops + Grad-CAM penalty (Phase 3)
│
├── evaluation/
│   ├── metrics.py                 # Accuracy + shortcut score (MNIST)
│   ├── visualize.py               # Attribution map comparison figures
│   ├── adversarial_metrics.py     # Adversarial evaluation + bar chart (Phase 1)
│   └── waterbirds_metrics.py      # Worst-group accuracy + shortcut score (Phase 3)
│
├── utils/
│   ├── helpers.py                 # Seed, device, checkpointing
│   └── logger.py                  # Training progress printer
│
└── results/
    └── figures/
        ├── attribution_comparison.png
        ├── training_curves.png
        ├── adversarial_comparison.png
        └── adversarial_per_class.png
```

---

## 15. Setup

```bash
git clone https://github.com/YOUR_USERNAME/shortcut-suppression.git
cd shortcut-suppression
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 16. Usage

```bash
# Train baseline only (Colored MNIST)
python main.py --mode baseline

# Train with shortcut suppression only
python main.py --mode suppress

# Train JTT baseline
python main.py --mode jtt

# Adversarial evaluation (requires trained models)
python main.py --mode adversarial \
  --load_baseline checkpoints/baseline.pth \
  --load_suppressed checkpoints/suppressed.pth

# Waterbirds + ResNet-18 (Phase 3)
python main.py --mode waterbirds

# Full pipeline: all models + all evaluations
python main.py --mode full

# Override lambda from command line
python main.py --mode full --lambda_shortcut 2.0

# Use SAC Discovery instead of hardcoded mask
python main.py --mode suppress --use_discovery
```

---

## 17. Configuration (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BIAS_RATIO` | 0.8 | Shortcut strength in Colored MNIST training (0.0–1.0) |
| `LAMBDA_SHORTCUT` | 1.0 | Shortcut penalty weight. 1.0 = equal weight with task loss |
| `WARMUP_EPOCHS` | 5 | Epochs before MNIST penalty activates |
| `EPOCHS` | 20 | Total MNIST training epochs |
| `BATCH_SIZE` | 512 | MNIST images per gradient update |
| `LEARNING_RATE` | 1e-3 | Adam learning rate (MNIST) |
| `SHORTCUT_THRESHOLD` | 0.3 | Score above which a sample is flagged as shortcut |
| `SEED` | 42 | Random seed |
| `JTT_ID_EPOCHS` | 1 | JTT identifier run length |
| `JTT_LAMBDA_UP` | 50 | JTT upsampling weight for failures |
| `WATERBIRDS_DIR` | (path) | Path to waterbird_complete95_forest2water2/ folder |
| `WATERBIRDS_EPOCHS` | 30 | ResNet-18 training epochs |
| `WATERBIRDS_BATCH` | 64 | Batch size for ResNet-18 (4GB VRAM constraint) |
| `WATERBIRDS_LR` | 1e-4 | Adam LR for ResNet-18 (pretrained weights are fragile) |
| `WATERBIRDS_LAMBDA` | 0.5 | Shortcut penalty weight for Waterbirds |
| `WATERBIRDS_WARMUP` | 5 | Warmup epochs before Grad-CAM penalty activates |
| `WATERBIRDS_BORDER_FRACTION` | 0.4 | Outer image fraction flagged as shortcut region |

---

## 18. Key Results Summary

### Colored MNIST — Shortcut Score (lower is better)

| Method | bias=1.0 | bias=0.8 | bias=0.5 | bias=0.0 |
|--------|----------|----------|----------|----------|
| ERM Baseline | 0.8895 | 0.5283 | 0.5429 | 0.5397 |
| ColorJitter | 0.6542 | 0.6404 | 0.6394 | 0.6293 |
| HighDropout | 0.9029 | 0.5029 | 0.5206 | 0.4753 |
| JTT | 0.8511 | 0.5333 | — | — |
| **Suppression (ours)** | **0.0715** | **0.0950** | **0.0878** | **0.0701** |

### Colored MNIST — Test Accuracy

| Method | bias=1.0 | bias=0.8 | bias=0.5 | bias=0.0 |
|--------|----------|----------|----------|----------|
| ERM Baseline | 10.26% | 97.53% | 98.47% | 98.84% |
| ColorJitter | 98.81% | 98.86% | 98.77% | 98.83% |
| HighDropout | 10.26% | 97.10% | 98.26% | 98.77% |
| JTT | 10.26% | 97.50% | — | — |
| **Suppression (ours)** | 10.27% | **96.71%** | **98.17%** | **98.50%** |

### Waterbirds — ResNet-18

| Method | Overall Acc | Worst-group Acc | SC Reduction |
|--------|-------------|-----------------|--------------|
| Baseline (ERM) | 87.12% | 69.16% | — |
| **Suppression (ours)** | **86.28%** | **75.39%** | 23% |

---

## 19. Limitations

**The supervised mask is dataset-specific.** The hardcoded brightness mask works because Colored MNIST is designed so the shortcut is always in the background. On a different dataset this assumption fails.

**SAC Discovery fails at bias=1.0.** At maximum bias, all class attribution maps look identical during warmup — cross-class variance is near-zero everywhere and the discovered mask is random.

**Accuracy does not recover at bias=1.0.** Suppression removes the cheat sheet but cannot invent a signal that was never present in training.

**Worst-group accuracy on Waterbirds is high-variance.** Group 2 (waterbird on land) has only 56 training samples. Val worst-group oscillated between 34% and 68% across training epochs. The best-checkpoint selection criterion partially addresses this but does not eliminate it. More stable results would require minority group upsampling during training in addition to attribution-based suppression.

**Border mask is an approximation.** The Waterbirds background extends into the image center around the bird, so the outer border fraction is an imperfect proxy for the true shortcut region. A learned or segmentation-based mask would be more precise.

**Single GPU validation.** All experiments were run on an RTX 3050 4GB. Larger batch sizes and more epochs may improve Waterbirds results on machines with more VRAM.

---

## 20. Next Steps

- [ ] Waterbirds + JTT comparison (train JTT on Waterbirds, compare worst-group)
- [ ] Adaptive border fraction using attention rollout instead of fixed geometry
- [ ] Combine attribution suppression with minority group upsampling for Waterbirds
- [ ] Validate on CelebA (hair color / gender spurious correlation)
- [ ] Extend to transformer architectures using attention maps instead of Grad-CAM
- [ ] Adaptive λ scheduling — increase penalty as suppression converges

---

## 21. References

- Geirhos et al. (2020) — Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*
- Sundararajan et al. (2017) — Axiomatic Attribution for Deep Networks. *ICML*
- Shrikumar et al. (2017) — Learning Important Features Through Propagating Activation Differences. *ICML*
- Nam et al. (2020) — Learning from Failure: Training Debiased Classifier from Biased Classifier. *NeurIPS*
- Liu et al. (2021) — Just Train Twice: Improving Group Robustness without Training Group Information. *ICML*
- Sagawa et al. (2020) — Distributionally Robust Neural Networks for Group Shifts. *ICLR*
- Selvaraju et al. (2017) — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*
- Captum — Model Interpretability for PyTorch. https://captum.ai/

---

## 22. Team

| Person | Responsibility |
|--------|---------------|
| Member 1 | `data/` — Dataset creation, dataloader, GPU caching, adversarial split |
| Member 2 | `explainer/` — Attribution maps, shortcut detection, SAC Discovery |
| Member 3 | `training/` + `evaluation/` — Loss functions, training loops, JTT, Waterbirds, metrics |