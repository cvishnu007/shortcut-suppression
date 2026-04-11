# =============================================================================
# evaluation/adversarial_metrics.py — Adversarial Evaluation Metrics
# =============================================================================
#
# WHAT THIS FILE DOES:
#   Extends the standard metrics.py with three new functions specifically
#   for adversarial evaluation:
#
#   1. compute_adversarial_accuracy()
#      Runs a model on the adversarial split and returns overall + per-class
#      accuracy.  This is the core number: baseline collapses, suppressed holds.
#
#   2. compute_adversarial_report()
#      Runs both models (baseline and suppressed) on both splits (standard
#      and adversarial) and returns a structured dict of all four numbers.
#      This produces the 2×2 table in your paper/poster.
#
#   3. print_adversarial_comparison()
#      Formats and prints the full comparison table to stdout.
#      Ready to paste into a report or workshop submission.
#
#   4. plot_adversarial_bar_chart()
#      Generates a publication-quality bar chart comparing the four
#      accuracy values.  This is Figure 1 of your paper.
#
# PLACEMENT:  save this file to  evaluation/adversarial_metrics.py
# =============================================================================

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless backend — safe on any machine
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from evaluation.metrics import compute_accuracy


# =============================================================================
# 1. Core adversarial accuracy
# =============================================================================

def compute_adversarial_accuracy(model, adv_loader, device):
    """
    Computes overall and per-class accuracy on the adversarial split.

    Interpretation guide:
        overall_acc ≈ 0.10  →  model is a pure shortcut learner
        overall_acc ≈ 0.96  →  model learned shape (shortcut suppressed)

    Args:
        model      : PyTorch model in eval or train mode (set to eval inside)
        adv_loader : DataLoader from get_adversarial_loader()
        device     : torch.device

    Returns:
        overall_acc : float
        per_class   : dict {digit: accuracy}
    """
    # compute_accuracy is already written and tested — reuse it directly.
    # The adversarial loader returns (image, label, color) exactly like the
    # standard loader, so no adapter layer is needed.
    return compute_accuracy(model, adv_loader, device)


# =============================================================================
# 2. Full 2×2 report: two models × two splits
# =============================================================================

def compute_adversarial_report(
    baseline_model,
    suppressed_model,
    standard_loader,
    adv_loader,
    device,
):
    """
    Evaluates both models on both splits and returns a structured results dict.

    The resulting 2×2 table is the central empirical claim of the project:

                        Standard test   Adversarial test
        Baseline            high             low          ← shortcut reliant
        Suppressed          high             high         ← shape reliant

    Args:
        baseline_model   : trained baseline (no suppression)
        suppressed_model : trained with shortcut suppression
        standard_loader  : standard randomized-color test DataLoader
        adv_loader       : adversarial DataLoader (guaranteed-wrong colors)
        device           : torch.device

    Returns:
        results : dict with keys
            'baseline_standard'   : (overall_acc, per_class)
            'baseline_adv'        : (overall_acc, per_class)
            'suppressed_standard' : (overall_acc, per_class)
            'suppressed_adv'      : (overall_acc, per_class)
    """
    print("\n[AdversarialEval] Evaluating baseline on standard test set ...")
    b_std  = compute_accuracy(baseline_model, standard_loader, device)

    print("[AdversarialEval] Evaluating baseline on adversarial test set ...")
    b_adv  = compute_adversarial_accuracy(baseline_model, adv_loader, device)

    print("[AdversarialEval] Evaluating suppressed model on standard test set ...")
    s_std  = compute_accuracy(suppressed_model, standard_loader, device)

    print("[AdversarialEval] Evaluating suppressed model on adversarial test set ...")
    s_adv  = compute_adversarial_accuracy(suppressed_model, adv_loader, device)

    return {
        'baseline_standard'   : b_std,
        'baseline_adv'        : b_adv,
        'suppressed_standard' : s_std,
        'suppressed_adv'      : s_adv,
    }


# =============================================================================
# 3. Pretty-print the comparison table
# =============================================================================

def print_adversarial_comparison(results, bias_ratio=None):
    """
    Prints a formatted comparison table to stdout.

    Args:
        results    : dict from compute_adversarial_report()
        bias_ratio : float — optional, for display context in header
    """
    b_std_acc = results['baseline_standard'][0]
    b_adv_acc = results['baseline_adv'][0]
    s_std_acc = results['suppressed_standard'][0]
    s_adv_acc = results['suppressed_adv'][0]

    # How much does each model degrade under adversarial shift?
    b_drop = b_std_acc - b_adv_acc
    s_drop = s_std_acc - s_adv_acc

    # Core metric: adversarial accuracy gap between the two models
    adv_gap = s_adv_acc - b_adv_acc

    bias_str = f"  (bias={bias_ratio:.0%})" if bias_ratio is not None else ""

    print(f"\n{'='*62}")
    print(f"  ADVERSARIAL EVALUATION RESULTS{bias_str}")
    print(f"{'='*62}")
    print(f"  {'Model':<22} {'Standard':>12}  {'Adversarial':>12}  {'Drop':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Baseline':<22} {b_std_acc:>11.2%}  {b_adv_acc:>11.2%}  {b_drop:>7.2%}")
    print(f"  {'Suppressed (ours)':<22} {s_std_acc:>11.2%}  {s_adv_acc:>11.2%}  {s_drop:>7.2%}")
    print(f"  {'-'*58}")
    print(f"  Adversarial accuracy gap (suppressed - baseline): {adv_gap:+.2%}")
    print(f"{'='*62}")

    # Interpretation
    if b_adv_acc < 0.20:
        print(f"\n  [✓] Baseline COLLAPSES under adversarial shift ({b_adv_acc:.1%})")
        print(f"      → Confirmed shortcut learner: model relied on color.")
    if s_adv_acc > 0.50:
        print(f"  [✓] Suppressed model SURVIVES adversarial shift ({s_adv_acc:.1%})")
        print(f"      → Confirmed shape learner: model robust to color shift.")
    if adv_gap > 0.30:
        print(f"  [✓] Gap of {adv_gap:.1%} is a STRONG empirical result.")

    return {
        'baseline_standard'   : b_std_acc,
        'baseline_adversarial': b_adv_acc,
        'suppressed_standard' : s_std_acc,
        'suppressed_adversarial': s_adv_acc,
        'adversarial_gap'     : adv_gap,
        'baseline_drop'       : b_drop,
        'suppressed_drop'     : s_drop,
    }


# =============================================================================
# 4. Publication-quality bar chart
# =============================================================================

def plot_adversarial_bar_chart(
    results,
    save_path=None,
    bias_ratio=None,
    extra_models=None,
):
    """
    Generates a grouped bar chart comparing model performance on standard
    and adversarial test sets.  This is Figure 1 of the paper/poster.

    Args:
        results      : dict from compute_adversarial_report()
        save_path    : str — path to save PNG (default: config.RESULTS_DIR)
        bias_ratio   : float — shown in title for context
        extra_models : dict of {model_name: (std_acc, adv_acc)} — optional
                       additional models (e.g. ColorJitter, HighDropout) to
                       add as extra bar groups for a richer comparison figure.

    Returns:
        fig : matplotlib Figure

    Layout:
        Each model gets two bars: Standard (solid) and Adversarial (hatched).
        The dramatic height difference between Baseline's two bars is the
        visual proof that it is a shortcut learner.
    """
    import config   # import here to avoid circular dependency at module level

    save_path = save_path or os.path.join(
        config.RESULTS_DIR, "adversarial_comparison.png"
    )

    # ── Gather data ────────────────────────────────────────────────────────
    model_names = ['Baseline', 'Suppressed\n(ours)']
    std_accs    = [
        results['baseline_standard'][0],
        results['suppressed_standard'][0],
    ]
    adv_accs    = [
        results['baseline_adv'][0],
        results['suppressed_adv'][0],
    ]

    # Optional extra models (ColorJitter, HighDropout, JTT, etc.)
    if extra_models:
        for name, (std_acc, adv_acc) in extra_models.items():
            model_names.append(name)
            std_accs.append(std_acc)
            adv_accs.append(adv_acc)

    n_models = len(model_names)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(7, n_models * 2.2), 5.5))

    x     = np.arange(n_models)
    width = 0.38
    gap   = 0.04

    # Standard test bars (solid)
    bars_std = ax.bar(
        x - width / 2 - gap / 2,
        std_accs,
        width,
        label='Standard test (random colors)',
        color='#4C8CBF',
        edgecolor='#2C5F8A',
        linewidth=0.8,
        zorder=3,
    )

    # Adversarial test bars (hatched — visually distinct in greyscale print)
    bars_adv = ax.bar(
        x + width / 2 + gap / 2,
        adv_accs,
        width,
        label='Adversarial test (wrong colors)',
        color='#E87040',
        edgecolor='#A84010',
        linewidth=0.8,
        hatch='//',
        zorder=3,
    )

    # ── Annotations — value labels on top of every bar ─────────────────────
    for bar in bars_std:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f'{h:.1%}', ha='center', va='bottom', fontsize=9, color='#2C5F8A',
        )
    for bar in bars_adv:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.012,
            f'{h:.1%}', ha='center', va='bottom', fontsize=9, color='#A84010',
        )

    # ── Styling ────────────────────────────────────────────────────────────
    bias_str = f" (bias={bias_ratio:.0%})" if bias_ratio is not None else ""
    ax.set_title(
        f'Adversarial Color Shift Evaluation{bias_str}\n'
        f'A shortcut-reliant model collapses; a shape-reliant model survives.',
        fontsize=11, pad=14,
    )
    ax.set_ylabel('Test Accuracy', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.axhline(0.10, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6, zorder=2, label='Random chance (10%)')
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"[AdversarialEval] Figure saved → {save_path}")

    return fig


# =============================================================================
# 5. Per-class accuracy breakdown on adversarial split
# =============================================================================

def plot_per_class_adversarial(results, save_path=None):
    """
    Plots per-class accuracy for both models on the adversarial split.

    This reveals whether suppression works uniformly across all digit classes
    or only for some.  Uneven per-class performance is worth discussing.

    Args:
        results   : dict from compute_adversarial_report()
        save_path : str — path to save PNG (optional)

    Returns:
        fig : matplotlib Figure
    """
    import config

    save_path = save_path or os.path.join(
        config.RESULTS_DIR, "adversarial_per_class.png"
    )

    b_per_class = results['baseline_adv'][1]        # dict {0: acc, ..., 9: acc}
    s_per_class = results['suppressed_adv'][1]

    digits = list(range(10))
    b_accs = [b_per_class[d] for d in digits]
    s_accs = [s_per_class[d] for d in digits]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    x     = np.arange(10)
    width = 0.38

    ax.bar(x - width / 2, b_accs, width,
           label='Baseline', color='#4C8CBF', edgecolor='#2C5F8A',
           linewidth=0.8, zorder=3)
    ax.bar(x + width / 2, s_accs, width,
           label='Suppressed (ours)', color='#E87040', edgecolor='#A84010',
           linewidth=0.8, zorder=3)

    ax.set_title(
        'Per-class Accuracy on Adversarial Test Set\n'
        '(colors guaranteed to be wrong for every sample)',
        fontsize=11, pad=12,
    )
    ax.set_xlabel('Digit class', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    ax.axhline(0.10, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.6, zorder=2, label='Random chance')
    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f"[AdversarialEval] Per-class figure saved → {save_path}")

    return fig
