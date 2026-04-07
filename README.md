ve# Automatic Shortcut Detection & Suppression

> A deep learning framework that forces neural networks to stop cheating —
> by using their own explanations as feedback signals during training.

---

## What is Shortcut Learning?

Neural networks often learn **spurious correlations** instead of true patterns.

**Example:** A cow classifier trained on images where cows are always on grass
learns "green background → cow" instead of "cow shape → cow".
It fails completely when a cow stands on snow.

This project **automatically detects** what the model is cheating with,
and **suppresses** it during training — without requiring human labels for the shortcuts.

---

## How It Works

```
Train → Explain → Detect Shortcut → Penalize → Retrain
```

1. Train a model normally (it will cheat)
2. Generate attribution maps (Integrated Gradients)
3. Detect if explanations focus on spurious regions
4. Add a penalty loss if they do
5. Retrain — model is forced to find real features

---

## Project Structure

```
shortcut-suppression/
│
├── config.py                  # All hyperparameters in one place
├── main.py                    # Entry point — run this to train
├── requirements.txt           # All pip dependencies
├── .gitignore                 # Files to exclude from git
│
├── data/
│   ├── __init__.py
│   ├── colored_mnist.py       # Creates the biased Colored MNIST dataset
│   └── dataloader.py          # Wraps data into PyTorch DataLoaders
│
├── models/
│   ├── __init__.py
│   └── cnn.py                 # Simple CNN model definition
│
├── explainer/
│   ├── __init__.py
│   ├── attribution.py         # Generates explanation maps using Captum
│   └── shortcut_detector.py   # Detects if explanations indicate shortcuts
│
├── training/
│   ├── __init__.py
│   ├── losses.py              # Task loss + shortcut penalty loss
│   └── trainer.py             # Full training loop
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # Accuracy, shortcut score
│   └── visualize.py           # Plots attribution maps before/after
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py             # Seed, device setup, checkpoint saving
│   └── logger.py              # Training log printer
│
├── notebooks/
│   └── __init__.py            # Add .ipynb files here for experiments
│
└── results/
    └── figures/               # Saved plots go here
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/shortcut-suppression.git
cd shortcut-suppression

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Run the Project

```bash
# Train baseline model (no shortcut suppression)
python main.py --mode baseline

# Train with shortcut suppression
python main.py --mode suppress

# Evaluate and visualize
python main.py --mode evaluate
```

---

## Team

| Person | Responsibility |
|--------|---------------|
| Member 1 | data/ — Dataset creation and loading |
| Member 2 | explainer/ — Attribution maps and shortcut detection |
| Member 3 | training/ + evaluation/ — Loss functions and evaluation |

---

## Key Concepts

| Term | Meaning |
|------|---------|
| Shortcut Learning | Model uses spurious features instead of real ones |
| Attribution Map | Visual explanation of what a model focused on |
| Integrated Gradients | Method to compute attribution per pixel |
| Shortcut Score | How much attribution lands in spurious regions |
| Suppression Loss | Penalty added to training loss for shortcut use |

---

## References

- ICCV 2025 — Efficient Unsupervised Shortcut Detection in Transformers
- ICML 2020 — Automatic Shortcut Removal for SSL
- Captum — Model Interpretability for PyTorch (https://captum.ai/)
