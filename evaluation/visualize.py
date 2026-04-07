# =============================================================================
# evaluation/visualize.py — Visualization of Attribution Maps
# =============================================================================
# WHAT THIS FILE DOES:
#   Creates visual comparisons of attribution maps:
#   - Before suppression: attributions likely highlight background (color)
#   - After suppression:  attributions should highlight digit shape
#
# WHY VISUALIZATIONS ARE CRITICAL FOR THIS PROJECT:
#   Numbers alone (accuracy) don't tell the full story.
#   Showing SIDE-BY-SIDE attribution maps is your strongest evidence.
#   "Look — the baseline model looks at the background.
#    Our model looks at the actual digit."
#   That's a conference-quality result.
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from captum.attr import IntegratedGradients
from explainer.attribution import attribution_to_heatmap
import config
import os


def visualize_attribution_comparison(
    baseline_model,
    suppressed_model,
    images,
    labels,
    device,
    save_path=None,
    n_samples=5
):
    """
    Creates a 3-row visualization for n_samples images:
      Row 1: Original colored image
      Row 2: Baseline model attribution heatmap
      Row 3: Suppression model attribution heatmap

    Args:
        baseline_model   : trained baseline (no suppression)
        suppressed_model : trained with shortcut suppression
        images           : Tensor (batch, 3, H, W) — sample images to visualize
        labels           : Tensor (batch,) — true labels
        device           : torch.device
        save_path        : str — path to save figure (optional)
        n_samples        : int — how many images to show side by side

    Example output:
        A figure with n_samples columns:
        [ img1 ]  [ img2 ]  ...
        [heat1a]  [heat2a]  ...  ← Baseline looks at background color
        [heat1b]  [heat2b]  ...  ← Suppressed looks at digit shape
    """

    # Limit samples
    images = images[:n_samples].to(device)
    labels = labels[:n_samples].to(device)

    ig_baseline   = IntegratedGradients(baseline_model)
    ig_suppressed = IntegratedGradients(suppressed_model)

    baseline_model.eval()
    suppressed_model.eval()

    # Storage for heatmaps
    heatmaps_baseline   = []
    heatmaps_suppressed = []

    for i in range(n_samples):
        img   = images[i].unsqueeze(0).requires_grad_(True)  # (1, 3, H, W)
        label = labels[i].unsqueeze(0)
        base  = torch.zeros_like(img)

        # Baseline attribution
        attr_b = ig_baseline.attribute(img, base, target=label, n_steps=25)
        heatmaps_baseline.append(attribution_to_heatmap(attr_b.detach()))

        # Suppression model attribution
        img2   = images[i].unsqueeze(0).requires_grad_(True)
        attr_s = ig_suppressed.attribute(img2, base, target=label, n_steps=25)
        heatmaps_suppressed.append(attribution_to_heatmap(attr_s.detach()))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))
    fig.suptitle("Attribution Map Comparison: Baseline vs. Shortcut Suppression",
                 fontsize=14, fontweight='bold', y=1.02)

    row_labels = ["Original Image", "Baseline Attribution\n(What it cheats with)",
                  "Suppression Attribution\n(What it actually learned)"]

    for i in range(n_samples):
        # Row 0: Original image
        img_display = images[i].cpu().permute(1, 2, 0).numpy()
        img_display = (img_display * 0.5 + 0.5)   # Undo normalization
        img_display = np.clip(img_display, 0, 1)

        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Label: {labels[i].item()}", fontsize=9)
        axes[0, i].axis('off')

        # Row 1: Baseline attribution (likely highlights background)
        axes[1, i].imshow(heatmaps_baseline[i], cmap='hot', vmin=0, vmax=1)
        axes[1, i].axis('off')

        # Row 2: Suppression attribution (should highlight digit)
        axes[2, i].imshow(heatmaps_suppressed[i], cmap='hot', vmin=0, vmax=1)
        axes[2, i].axis('off')

    # Add row labels on the left
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=0,
                                     labelpad=80, va='center')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[Viz] Saved to {save_path}")

    plt.show()
    return fig


def plot_training_curves(baseline_history, suppression_history, save_path=None):
    """
    Plots training accuracy curves for baseline vs. suppression model.
    Shows the tradeoff between accuracy and shortcut reduction.

    Args:
        baseline_history   : dict from train_baseline()
        suppression_history: dict from train_with_suppression()
        save_path          : str — path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training Curves: Baseline vs. Shortcut Suppression", fontsize=13)

    epochs = range(1, len(baseline_history['train_acc']) + 1)

    # ── Plot 1: Test Accuracy ─────────────────────────────────────────────────
    axes[0].plot(epochs, baseline_history['test_acc'],
                 label='Baseline', color='tomato', linewidth=2)
    axes[0].plot(epochs, suppression_history['test_acc'],
                 label='Suppression', color='steelblue', linewidth=2)
    axes[0].set_title("Test Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # ── Plot 2: Suppression Loss ──────────────────────────────────────────────
    if 'shortcut_loss' in suppression_history:
        axes[1].plot(epochs, suppression_history['shortcut_loss'],
                     color='steelblue', linewidth=2)
        axes[1].set_title("Shortcut Penalty Loss (Suppression Model)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Shortcut Loss")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[Viz] Saved to {save_path}")

    plt.show()
    return fig
