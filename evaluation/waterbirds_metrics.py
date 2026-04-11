# =============================================================================
# evaluation/waterbirds_metrics.py — Waterbirds Evaluation
# =============================================================================
#
# PRIMARY METRIC: Worst-group accuracy
#   Waterbirds has 4 groups (y × place).  Groups 1 and 2 are minorities:
#     Group 1: landbird on water  (~184 train samples)
#     Group 2: waterbird on land  (~56 train samples)
#   A shortcut model scores ~10-20% on these groups while overall accuracy
#   stays high.  Worst-group accuracy is min(per_group_accuracies).
#
# SECONDARY METRICS:
#   - Overall accuracy (average across all test samples)
#   - Per-group accuracy (all 4 groups)
#   - Shortcut score: fraction of Grad-CAM attribution in the border region
#     (analogous to the background shortcut score used for Colored MNIST)
#
# PLACEMENT:  save to  evaluation/waterbirds_metrics.py
# =============================================================================

import torch
import numpy as np

from models.resnet import get_border_mask
import config


GROUP_NAMES = {
    0: 'Landbird on land  (majority)',
    1: 'Landbird on water (minority ← shortcut failure)',
    2: 'Waterbird on land (minority ← shortcut failure)',
    3: 'Waterbird on water(majority)',
}


# =============================================================================
# Full evaluation
# =============================================================================

def evaluate_waterbirds(model, test_loader, device):
    """
    Computes overall accuracy, per-group accuracy, and worst-group accuracy
    on the Waterbirds test set.

    Args:
        model       : ResNet18Waterbirds
        test_loader : DataLoader (split='test')
        device      : torch.device

    Returns:
        overall_acc  : float
        worst_group  : float
        per_group    : dict {0: acc, 1: acc, 2: acc, 3: acc}
    """
    model.eval()
    group_correct = {g: 0 for g in range(4)}
    group_total   = {g: 0 for g in range(4)}

    with torch.no_grad():
        for images, labels, groups in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)

            for pred, label, group in zip(preds, labels, groups):
                g = int(group.item())
                group_total[g]   += 1
                if pred.item() == label.item():
                    group_correct[g] += 1

    per_group = {
        g: group_correct[g] / group_total[g]
        if group_total[g] > 0 else 0.0
        for g in range(4)
    }
    overall     = sum(group_correct.values()) / sum(group_total.values())
    worst_group = min(per_group.values())

    return overall, worst_group, per_group


# =============================================================================
# Grad-CAM shortcut score
# =============================================================================

def compute_waterbirds_shortcut_score(
    model, test_loader, device, n_batches=10, border_fraction=0.3
):
    """
    Computes average Grad-CAM attribution in the border (background) region.

    Higher score = model attends to background = shortcut reliant.
    Lower score  = model attends to bird center = shape reliant.

    Args:
        model           : ResNet18Waterbirds
        test_loader     : DataLoader
        device          : torch.device
        n_batches       : int — number of batches to evaluate
        border_fraction : float — outer fraction flagged as background

    Returns:
        avg_score : float in [0, 1]
    """
    model.eval()
    all_scores = []

    for batch_idx, (images, labels, _) in enumerate(test_loader):
        if batch_idx >= n_batches:
            break

        images, labels = images.to(device), labels.to(device)
        images.requires_grad_(True)

        # Forward pass — populates activation hook
        logits = model(images)

        # Backward pass on predicted class scores — populates gradient hook
        target_scores = logits.gather(1, labels.view(-1, 1)).squeeze(1).sum()
        target_scores.backward()

        # Get Grad-CAM heatmap
        cam  = model.get_gradcam(target_size=images.shape[-1])  # (B,1,224,224)
        mask = get_border_mask(
            images.shape[0], images.shape[-1],
            border_fraction=border_fraction,
            device=device
        )

        border_attr = (cam * mask).sum(dim=(1, 2, 3))
        total_attr  = cam.sum(dim=(1, 2, 3)) + 1e-8
        scores      = (border_attr / total_attr).detach().cpu().numpy()
        all_scores.extend(scores.tolist())

    return float(np.mean(all_scores))


# =============================================================================
# Pretty-print report
# =============================================================================

def print_waterbirds_report(model_name, overall, worst_group, per_group, shortcut_score):
    """
    Prints a formatted evaluation report for Waterbirds.

    Args:
        model_name     : str
        overall        : float
        worst_group    : float
        per_group      : dict {0-3: acc}
        shortcut_score : float
    """
    print(f"\n{'='*60}")
    print(f"  WATERBIRDS EVALUATION: {model_name}")
    print(f"{'='*60}")
    print(f"  Overall accuracy    : {overall:.2%}")
    print(f"  Worst-group accuracy: {worst_group:.2%}  ← primary metric")
    print(f"  Shortcut score      : {shortcut_score:.4f}  (lower = better)")
    print(f"\n  Per-group accuracy:")
    for g in range(4):
        acc  = per_group[g]
        bar  = '█' * int(acc * 20)
        flag = ' ← MINORITY' if g in (1, 2) else ''
        print(f"    Group {g} ({GROUP_NAMES[g][:28]}): {acc:.2%}  {bar}{flag}")
    print(f"{'='*60}")


# =============================================================================
# Comparison summary
# =============================================================================

def print_waterbirds_comparison(results):
    """
    Prints a side-by-side comparison table for all evaluated models.

    Args:
        results : dict {model_name: {'overall': float, 'worst_group': float,
                                     'shortcut': float}}
    """
    print(f"\n{'='*72}")
    print(f"  WATERBIRDS COMPARISON SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Method':<26} {'Overall':>9}  {'Worst-group':>12}  {'SC Score':>10}")
    print(f"  {'-'*62}")
    for name, vals in results.items():
        marker = ' ←' if 'Suppression' in name or 'ours' in name.lower() else ''
        print(
            f"  {name:<26} {vals['overall']:>8.2%}  "
            f"{vals['worst_group']:>11.2%}  "
            f"{vals['shortcut']:>9.4f}{marker}"
        )
    print(f"{'='*72}")

    # Highlight worst-group improvement
    names = list(results.keys())
    if len(names) >= 2:
        baseline_wg    = results[names[0]]['worst_group']
        suppressed_wg  = results[names[-1]]['worst_group']
        wg_improvement = suppressed_wg - baseline_wg
        print(f"\n  Worst-group improvement (suppressed vs baseline): "
              f"{wg_improvement:+.2%}")
