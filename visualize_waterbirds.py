# =============================================================================
# visualize_waterbirds.py — Grad-CAM Heatmap Comparison for Waterbirds
# =============================================================================
# WHAT THIS DOES:
#   Generates a side-by-side visualization showing:
#     Row 1: Original bird image
#     Row 2: Baseline model Grad-CAM heatmap (what it cheats with)
#     Row 3: Suppressed model Grad-CAM heatmap (what it actually learned)
#
# USAGE:
#   python visualize_waterbirds.py
#
# REQUIRES:
#   - Trained baseline checkpoint:   ./checkpoints/waterbirds_baseline.pth
#   - Trained suppressed checkpoint: ./checkpoints/waterbirds_suppressed.pth
#   - Waterbirds dataset at config.WATERBIRDS_DIR
#
# OUTPUT:
#   ./results/figures/waterbirds_heatmap_comparison.png
# =============================================================================

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F

from data.waterbirds import WaterbirdsDataset, get_transforms
from models.resnet import get_waterbirds_model
import config


# =============================================================================
# Group metadata
# =============================================================================

GROUP_NAMES = {
    0: 'Landbird / Land bg\n(majority)',
    1: 'Landbird / Water bg\n(minority ← shortcut failure)',
    2: 'Waterbird / Land bg\n(minority ← shortcut failure)',
    3: 'Waterbird / Water bg\n(majority)',
}

LABEL_NAMES = {0: 'Landbird', 1: 'Waterbird'}


# =============================================================================
# Grad-CAM extraction
# =============================================================================

def get_gradcam(model, image_tensor, device):
    """
    Computes Grad-CAM heatmap for a single image.

    Args:
        model        : ResNet18Waterbirds (hooks already registered)
        image_tensor : Tensor (1, 3, 224, 224) — single image, normalized
        device       : torch.device

    Returns:
        cam   : numpy array (224, 224) in [0, 1] — heatmap
        pred  : int — predicted class (0=landbird, 1=waterbird)
        conf  : float — confidence in prediction
    """
    model.eval()
    image_tensor = image_tensor.to(device).requires_grad_(True)

    logits = model(image_tensor)
    pred   = logits.argmax(dim=1).item()
    conf   = torch.softmax(logits, dim=1)[0, pred].item()

    # Backward on the predicted class score
    model.zero_grad()
    logits[0, pred].backward()

    # Get Grad-CAM from model hooks
    cam = model.get_gradcam(target_size=224)   # (1, 1, 224, 224)
    cam = cam[0, 0].detach().cpu().numpy()     # (224, 224)

    return cam, pred, conf


# =============================================================================
# Image denormalization for display
# =============================================================================

def denormalize(tensor):
    """
    Reverses ImageNet normalization for display.

    Args:
        tensor : Tensor (3, 224, 224) — ImageNet normalized

    Returns:
        numpy array (224, 224, 3) in [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


# =============================================================================
# Sample selection — one image per group
# =============================================================================

def get_one_per_group(dataset, n_per_group=1):
    """
    Returns one sample per group (groups 0-3).
    Prioritises minority groups (1 and 2) for visual impact.

    Returns:
        samples : list of (image_tensor, label, group) tuples
    """
    collected = {g: [] for g in range(4)}

    for idx in range(len(dataset)):
        img, label, group = dataset[idx]
        if len(collected[group]) < n_per_group:
            collected[group].append((img, label, group))
        if all(len(v) >= n_per_group for v in collected.values()):
            break

    # Order: minority groups first (more interesting), then majority
    order   = [2, 1, 0, 3]
    samples = []
    for g in order:
        samples.extend(collected[g][:n_per_group])
    return samples


# =============================================================================
# Main visualization
# =============================================================================

def visualize_waterbirds_heatmaps(
    baseline_ckpt   = None,
    suppressed_ckpt = None,
    save_path       = None,
    n_samples       = 4,
    alpha           = 0.5,
):
    """
    Generates a 3-row × n_samples figure:
        Row 1: Original image
        Row 2: Baseline Grad-CAM overlay (red = high attention)
        Row 3: Suppressed Grad-CAM overlay

    Args:
        baseline_ckpt   : str — path to baseline .pth checkpoint
        suppressed_ckpt : str — path to suppressed .pth checkpoint
        save_path       : str — where to save the figure
        n_samples       : int — number of images to show (one per group up to 4)
        alpha           : float — heatmap overlay transparency (0=invisible, 1=opaque)
    """
    # ── Paths ─────────────────────────────────────────────────────────────────
    baseline_ckpt   = baseline_ckpt   or os.path.join(
        config.CHECKPOINT_DIR, 'waterbirds_baseline.pth'
    )
    suppressed_ckpt = suppressed_ckpt or os.path.join(
        config.CHECKPOINT_DIR, 'waterbirds_suppressed.pth'
    )
    save_path = save_path or os.path.join(
        config.RESULTS_DIR, 'waterbirds_heatmap_comparison.png'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Viz] Using device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[Viz] Loading baseline model ...")
    baseline_model = get_waterbirds_model(pretrained=False).to(device)
    ckpt = torch.load(baseline_ckpt, map_location=device)
    baseline_model.load_state_dict(ckpt['model_state_dict'])
    baseline_model.eval()
    print(f"      Loaded from {baseline_ckpt}  (epoch={ckpt['epoch']}, "
          f"acc={ckpt['accuracy']:.4f})")

    print("[Viz] Loading suppressed model ...")
    suppressed_model = get_waterbirds_model(pretrained=False).to(device)
    ckpt = torch.load(suppressed_ckpt, map_location=device)
    suppressed_model.load_state_dict(ckpt['model_state_dict'])
    suppressed_model.eval()
    print(f"      Loaded from {suppressed_ckpt}  (epoch={ckpt['epoch']}, "
          f"acc={ckpt['accuracy']:.4f})")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("[Viz] Loading Waterbirds test set ...")
    dataset = WaterbirdsDataset(
        root      = config.WATERBIRDS_DIR,
        split     = 'test',
        transform = get_transforms(train=False),
    )

    samples = get_one_per_group(dataset, n_per_group=1)
    samples = samples[:n_samples]
    actual_n = len(samples)

    print(f"[Viz] Generating heatmaps for {actual_n} samples ...")

    # ── Compute heatmaps ──────────────────────────────────────────────────────
    images_display  = []
    cams_baseline   = []
    cams_suppressed = []
    preds_baseline  = []
    preds_suppressed= []
    confs_baseline  = []
    confs_suppressed= []
    true_labels     = []
    group_ids       = []

    for img_tensor, label, group in samples:
        inp = img_tensor.unsqueeze(0)   # (1, 3, 224, 224)

        cam_b, pred_b, conf_b = get_gradcam(baseline_model,   inp, device)
        cam_s, pred_s, conf_s = get_gradcam(suppressed_model, inp, device)

        images_display.append(denormalize(img_tensor))
        cams_baseline.append(cam_b)
        cams_suppressed.append(cam_s)
        preds_baseline.append(pred_b)
        preds_suppressed.append(pred_s)
        confs_baseline.append(conf_b)
        confs_suppressed.append(conf_s)
        true_labels.append(label)
        group_ids.append(group)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, actual_n,
        figsize=(4.5 * actual_n, 13),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.08}
    )

    # Handle single-column case
    if actual_n == 1:
        axes = axes.reshape(3, 1)

    row_titles = [
        'Original Image',
        'Baseline Grad-CAM\n(what it cheats with)',
        'Suppressed Grad-CAM\n(what it actually learned)',
    ]

    for col in range(actual_n):
        img   = images_display[col]
        cam_b = cams_baseline[col]
        cam_s = cams_suppressed[col]

        true_name  = LABEL_NAMES[true_labels[col]]
        group_name = GROUP_NAMES[group_ids[col]]
        is_minority = group_ids[col] in (1, 2)

        # ── Row 0: Original image ─────────────────────────────────────────────
        axes[0, col].imshow(img)
        axes[0, col].axis('off')

        title_color = '#C0392B' if is_minority else '#2C3E50'
        axes[0, col].set_title(
            f"True: {true_name}\n{group_name}",
            fontsize=8.5, color=title_color,
            fontweight='bold' if is_minority else 'normal',
            pad=6,
        )

        # ── Row 1: Baseline Grad-CAM overlay ─────────────────────────────────
        axes[1, col].imshow(img)
        axes[1, col].imshow(cam_b, cmap='jet', alpha=alpha, vmin=0, vmax=1)
        axes[1, col].axis('off')

        pred_name_b = LABEL_NAMES[preds_baseline[col]]
        correct_b   = '✓' if preds_baseline[col] == true_labels[col] else '✗'
        axes[1, col].set_title(
            f"Pred: {pred_name_b} {correct_b}  ({confs_baseline[col]:.0%})",
            fontsize=8, color='#27AE60' if correct_b == '✓' else '#E74C3C',
            pad=4,
        )

        # ── Row 2: Suppressed Grad-CAM overlay ───────────────────────────────
        axes[2, col].imshow(img)
        axes[2, col].imshow(cam_s, cmap='jet', alpha=alpha, vmin=0, vmax=1)
        axes[2, col].axis('off')

        pred_name_s = LABEL_NAMES[preds_suppressed[col]]
        correct_s   = '✓' if preds_suppressed[col] == true_labels[col] else '✗'
        axes[2, col].set_title(
            f"Pred: {pred_name_s} {correct_s}  ({confs_suppressed[col]:.0%})",
            fontsize=8, color='#27AE60' if correct_s == '✓' else '#E74C3C',
            pad=4,
        )

    # ── Row labels on the left ────────────────────────────────────────────────
    for row_idx, row_title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(
            row_title, fontsize=10, rotation=0,
            labelpad=110, va='center', fontweight='bold',
        )

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        'Waterbirds Grad-CAM: Baseline vs. Shortcut Suppression\n'
        'Red = high attention  |  Blue = low attention  |  '
        'Minority groups highlighted in red',
        fontsize=12, fontweight='bold', y=1.01,
    )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.5])
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attribution strength', fontsize=9, rotation=270, labelpad=14)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'], fontsize=8)

    # ── Legend for minority indicator ─────────────────────────────────────────
    minority_patch  = mpatches.Patch(color='#C0392B', label='Minority group (shortcut failure)')
    majority_patch  = mpatches.Patch(color='#2C3E50', label='Majority group')
    fig.legend(
        handles=[minority_patch, majority_patch],
        loc='lower center', ncol=2,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.04),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Viz] Saved → {save_path}")
    plt.show()
    return fig


# =============================================================================
# Extended version: show multiple samples per group
# =============================================================================

def visualize_per_group_grid(
    baseline_ckpt   = None,
    suppressed_ckpt = None,
    save_path       = None,
    n_per_group     = 2,
    alpha           = 0.5,
):
    """
    Extended grid showing n_per_group samples for each of the 4 Waterbirds groups.
    Total columns = 4 × n_per_group.

    More comprehensive than visualize_waterbirds_heatmaps but also larger.
    Good for appendix figures.
    """
    baseline_ckpt   = baseline_ckpt   or os.path.join(
        config.CHECKPOINT_DIR, 'waterbirds_baseline.pth'
    )
    suppressed_ckpt = suppressed_ckpt or os.path.join(
        config.CHECKPOINT_DIR, 'waterbirds_suppressed.pth'
    )
    save_path = save_path or os.path.join(
        config.RESULTS_DIR, 'waterbirds_heatmap_grid.png'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseline_model   = get_waterbirds_model(pretrained=False).to(device)
    suppressed_model = get_waterbirds_model(pretrained=False).to(device)

    ckpt = torch.load(baseline_ckpt, map_location=device)
    baseline_model.load_state_dict(ckpt['model_state_dict'])
    baseline_model.eval()

    ckpt = torch.load(suppressed_ckpt, map_location=device)
    suppressed_model.load_state_dict(ckpt['model_state_dict'])
    suppressed_model.eval()

    dataset = WaterbirdsDataset(
        root=config.WATERBIRDS_DIR, split='test',
        transform=get_transforms(train=False),
    )

    # Collect n_per_group samples per group
    collected = {g: [] for g in range(4)}
    for idx in range(len(dataset)):
        img, label, group = dataset[idx]
        if len(collected[group]) < n_per_group:
            collected[group].append((img, label, group))
        if all(len(v) >= n_per_group for v in collected.values()):
            break

    # Flatten: group 0 samples, group 1 samples, group 2 samples, group 3 samples
    all_samples = []
    group_boundaries = []   # column index where each group starts
    for g in range(4):
        group_boundaries.append(len(all_samples))
        all_samples.extend(collected[g])

    n_cols = len(all_samples)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 12))

    if n_cols == 1:
        axes = axes.reshape(3, 1)

    images_display   = []
    cams_baseline    = []
    cams_suppressed  = []
    preds_b          = []
    preds_s          = []
    true_labels_list = []
    group_ids_list   = []

    for img_tensor, label, group in all_samples:
        inp = img_tensor.unsqueeze(0)
        cam_b, pred_b, _ = get_gradcam(baseline_model,   inp, device)
        cam_s, pred_s, _ = get_gradcam(suppressed_model, inp, device)
        images_display.append(denormalize(img_tensor))
        cams_baseline.append(cam_b)
        cams_suppressed.append(cam_s)
        preds_b.append(pred_b)
        preds_s.append(pred_s)
        true_labels_list.append(label)
        group_ids_list.append(group)

    for col in range(n_cols):
        img   = images_display[col]
        cam_b = cams_baseline[col]
        cam_s = cams_suppressed[col]
        true  = true_labels_list[col]
        group = group_ids_list[col]

        axes[0, col].imshow(img)
        axes[0, col].axis('off')
        axes[0, col].set_title(
            f"{LABEL_NAMES[true]}\nGrp {group}",
            fontsize=7.5,
            color='#C0392B' if group in (1, 2) else '#2C3E50',
        )

        axes[1, col].imshow(img)
        axes[1, col].imshow(cam_b, cmap='jet', alpha=alpha, vmin=0, vmax=1)
        axes[1, col].axis('off')
        c = '✓' if preds_b[col] == true else '✗'
        axes[1, col].set_title(
            f"{LABEL_NAMES[preds_b[col]]} {c}",
            fontsize=7.5,
            color='#27AE60' if c == '✓' else '#E74C3C',
        )

        axes[2, col].imshow(img)
        axes[2, col].imshow(cam_s, cmap='jet', alpha=alpha, vmin=0, vmax=1)
        axes[2, col].axis('off')
        c = '✓' if preds_s[col] == true else '✗'
        axes[2, col].set_title(
            f"{LABEL_NAMES[preds_s[col]]} {c}",
            fontsize=7.5,
            color='#27AE60' if c == '✓' else '#E74C3C',
        )

    # Group separators
    for boundary in group_boundaries[1:]:
        for row in range(3):
            axes[row, boundary].spines['left'].set_visible(True)
            axes[row, boundary].spines['left'].set_linewidth(2)
            axes[row, boundary].spines['left'].set_color('#7F8C8D')

    row_labels = ['Original', 'Baseline\nGrad-CAM', 'Suppressed\nGrad-CAM']
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(
            label, fontsize=9, rotation=0,
            labelpad=80, va='center', fontweight='bold',
        )

    group_label_x = [
        (group_boundaries[g] + (group_boundaries[g+1] if g+1 < 4 else n_cols)) / 2 / n_cols
        for g in range(4)
    ]
    for g, gx in enumerate(group_label_x):
        color = '#C0392B' if g in (1, 2) else '#2C3E50'
        fig.text(
            gx, 1.005,
            f"Group {g}: {['Landbird/Land', 'Landbird/Water', 'Waterbird/Land', 'Waterbird/Water'][g]}",
            ha='center', va='bottom', fontsize=9,
            color=color, fontweight='bold' if g in (1, 2) else 'normal',
            transform=fig.transFigure,
        )

    fig.suptitle(
        'Waterbirds Grad-CAM Grid — Baseline vs. Suppression\n'
        'Minority groups (1, 2) in red — these are where shortcut failures occur',
        fontsize=11, fontweight='bold', y=1.03,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Viz] Grid saved → {save_path}")
    plt.show()
    return fig


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Waterbirds Grad-CAM Visualization')
    parser.add_argument(
        '--mode', type=str, default='comparison',
        choices=['comparison', 'grid'],
        help='comparison = 4 samples (one per group) | grid = 2 per group'
    )
    parser.add_argument('--baseline',   type=str, default=None)
    parser.add_argument('--suppressed', type=str, default=None)
    parser.add_argument('--alpha',      type=float, default=0.5,
                        help='Heatmap overlay opacity (0-1)')
    args = parser.parse_args()

    if args.mode == 'comparison':
        visualize_waterbirds_heatmaps(
            baseline_ckpt   = args.baseline,
            suppressed_ckpt = args.suppressed,
            alpha           = args.alpha,
        )
    else:
        visualize_per_group_grid(
            baseline_ckpt   = args.baseline,
            suppressed_ckpt = args.suppressed,
            alpha           = args.alpha,
        )