# =============================================================================
# main.py — Entry Point
# =============================================================================
# Run this file to start the project.
#
# USAGE:
#   python main.py --mode baseline     # Train baseline (no suppression)
#   python main.py --mode suppress     # Train with shortcut suppression
#   python main.py --mode evaluate     # Evaluate both + standard visualizations
#   python main.py --mode adversarial  # ← NEW: adversarial evaluation only
#   python main.py --mode full         # Run everything end-to-end (recommended)
#
# ADVERSARIAL MODE:
#   Loads pre-trained checkpoints (or trains both models if none exist) and
#   runs the adversarial color-shift evaluation.  Produces two figures:
#     results/figures/adversarial_comparison.png  ← main result bar chart
#     results/figures/adversarial_per_class.png   ← per-class breakdown
#
#   The adversarial test set guarantees that every image's background color
#   belongs to a DIFFERENT digit class.  A shortcut-reliant model collapses
#   to ~10% accuracy; a shape-reliant model holds at 96%+.
# =============================================================================

import argparse
import torch
import gc
import os
import config
from utils.helpers import set_seed, get_device, save_checkpoint, load_checkpoint
from data.dataloader import get_dataloaders
from models.cnn import get_model
from training.trainer import train_baseline, train_with_suppression
from evaluation.metrics import (
    compute_accuracy,
    compute_average_shortcut_score,
    print_evaluation_report,
)
from evaluation.visualize import visualize_attribution_comparison, plot_training_curves

# ── Phase 1 imports ────────────────────────────────────────────────────────────
from data.adversarial_dataset import get_adversarial_loader
from evaluation.adversarial_metrics import (
    compute_adversarial_report,
    print_adversarial_comparison,
    plot_adversarial_bar_chart,
    plot_per_class_adversarial,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shortcut Detection & Suppression — Explanation-Guided Training"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['baseline', 'suppress', 'evaluate', 'adversarial', 'full'],
        help=(
            "Which mode to run.  "
            "'adversarial' runs the color-shift stress test on trained models. "
            "'full' runs everything end-to-end."
        ),
    )
    parser.add_argument(
        '--lambda_shortcut',
        type=float,
        default=None,
        help="Override lambda for the shortcut penalty (default: from config.py)",
    )
    parser.add_argument(
        '--use_discovery',
        action='store_true',
        default=False,
        help="Use SAC Discovery (unsupervised mask) instead of hardcoded mask.",
    )
    parser.add_argument(
        '--load_baseline',
        type=str,
        default=None,
        help="Path to a saved baseline checkpoint (.pth) — skip re-training.",
    )
    parser.add_argument(
        '--load_suppressed',
        type=str,
        default=None,
        help="Path to a saved suppressed checkpoint (.pth) — skip re-training.",
    )
    return parser.parse_args()


# =============================================================================
# Adversarial evaluation block
# =============================================================================

def run_adversarial_evaluation(
    baseline_model,
    suppressed_model,
    standard_test_loader,
    device,
    extra_models=None,
):
    """
    Runs the full Phase 1 adversarial evaluation and saves all figures.

    Args:
        baseline_model       : trained baseline model
        suppressed_model     : trained suppressed model
        standard_test_loader : standard DataLoader (random colors)
        device               : torch.device
        extra_models         : dict {name: (std_acc, adv_acc)} for extra bars
                               (pass ColorJitter / HighDropout accuracies here)

    Returns:
        summary : dict of scalar results from print_adversarial_comparison()
    """
    print("\n" + "="*62)
    print("  PHASE 1 — ADVERSARIAL COLOR-SHIFT EVALUATION")
    print("="*62)
    print(
        "\n  Test protocol:\n"
        "    Standard split  : colors are random (10% hit rate by luck)\n"
        "    Adversarial split: every digit gets a WRONG class's color\n"
        "    Shortcut model  → accuracy collapses to ~10%\n"
        "    Shape model     → accuracy holds at ~96%+\n"
    )

    # Build adversarial loader (VRAM-cached, reuses MNIST already on disk)
    adv_loader = get_adversarial_loader()

    # Free unreferenced tensors before running IG attribution
    torch.cuda.empty_cache()
    gc.collect()

    # Run 2×2 evaluation
    results = compute_adversarial_report(
        baseline_model=baseline_model,
        suppressed_model=suppressed_model,
        standard_loader=standard_test_loader,
        adv_loader=adv_loader,
        device=device,
    )

    # Print table and get scalar summary
    summary = print_adversarial_comparison(results, bias_ratio=config.BIAS_RATIO)

    # Figure 1: grouped bar chart (main paper figure)
    plot_adversarial_bar_chart(
        results=results,
        save_path=os.path.join(config.RESULTS_DIR, "adversarial_comparison.png"),
        bias_ratio=config.BIAS_RATIO,
        extra_models=extra_models,
    )

    # Figure 2: per-class breakdown on adversarial split
    plot_per_class_adversarial(
        results=results,
        save_path=os.path.join(config.RESULTS_DIR, "adversarial_per_class.png"),
    )

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    set_seed(config.SEED)
    device = get_device()

    print(f"\n[Config] Mode            : {args.mode}")
    print(f"[Config] Bias Ratio      : {config.BIAS_RATIO}")
    print(f"[Config] Epochs          : {config.EPOCHS}")
    print(f"[Config] Lambda shortcut : {config.LAMBDA_SHORTCUT}")
    print(f"[Config] SAC Discovery   : {args.use_discovery}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders()

    # ── Models ─────────────────────────────────────────────────────────────────
    baseline_model   = get_model().to(device)
    suppressed_model = get_model().to(device)

    # ── Optionally load pre-trained weights ────────────────────────────────────
    if args.load_baseline and os.path.exists(args.load_baseline):
        baseline_model, _, _ = load_checkpoint(
            baseline_model, args.load_baseline, device
        )
        print(f"[Main] Loaded baseline from {args.load_baseline}")

    if args.load_suppressed and os.path.exists(args.load_suppressed):
        suppressed_model, _, _ = load_checkpoint(
            suppressed_model, args.load_suppressed, device
        )
        print(f"[Main] Loaded suppressed model from {args.load_suppressed}")

    # ── Training ───────────────────────────────────────────────────────────────
    baseline_history    = None
    suppression_history = None

    if args.mode in ['baseline', 'full'] and not args.load_baseline:
        baseline_history = train_baseline(
            baseline_model, train_loader, test_loader, device
        )
        if config.SAVE_BEST_MODEL:
            save_checkpoint(
                baseline_model, None,
                epoch=config.EPOCHS,
                accuracy=baseline_history['test_acc'][-1],
                path=os.path.join(config.CHECKPOINT_DIR, "baseline.pth"),
            )

    if args.mode in ['suppress', 'full'] and not args.load_suppressed:
        if args.lambda_shortcut is not None:
            config.LAMBDA_SHORTCUT = args.lambda_shortcut

        suppression_history = train_with_suppression(
            suppressed_model, train_loader, test_loader, device,
            use_discovery=args.use_discovery,
        )
        if config.SAVE_BEST_MODEL:
            save_checkpoint(
                suppressed_model, None,
                epoch=config.EPOCHS,
                accuracy=suppression_history['test_acc'][-1],
                path=os.path.join(config.CHECKPOINT_DIR, "suppressed.pth"),
            )

    # ── Adversarial evaluation (Phase 1) ───────────────────────────────────────
    if args.mode in ['adversarial', 'full']:
        # If running in 'adversarial' mode without prior training, we need
        # both models to exist.  Warn clearly instead of crashing silently.
        if args.mode == 'adversarial':
            if args.load_baseline is None and baseline_history is None:
                print(
                    "\n[Warning] --mode adversarial: no baseline model trained "
                    "or loaded.\n"
                    "  Either run --mode full, or pass --load_baseline <path>.\n"
                    "  Proceeding with a randomly-initialized baseline "
                    "(results will be ~10% for both models).\n"
                )
            if args.load_suppressed is None and suppression_history is None:
                print(
                    "\n[Warning] --mode adversarial: no suppressed model trained "
                    "or loaded.\n"
                    "  Proceeding with a randomly-initialized suppressed model.\n"
                )

        run_adversarial_evaluation(
            baseline_model=baseline_model,
            suppressed_model=suppressed_model,
            standard_test_loader=test_loader,
            device=device,
            extra_models=None,   # populate with ColorJitter/HighDropout in Phase 2
        )

    # ── Standard evaluation (unchanged from original) ──────────────────────────
    if args.mode in ['evaluate', 'full']:
        print("\n[Evaluation] Computing standard metrics ...")

        # ── ColorJitter baseline ───────────────────────────────────────────────
        from data.dataloader import get_dataloaders_jitter
        jitter_train_loader, jitter_test_loader = get_dataloaders_jitter()
        jitter_model = get_model().to(device)
        print("\n[Baseline] Training ColorJitter model ...")
        jitter_history = train_baseline(
            jitter_model, jitter_train_loader, jitter_test_loader, device
        )

        # ── HighDropout baseline ───────────────────────────────────────────────
        dropout_model = get_model(dropout=0.8).to(device)
        print("\n[Baseline] Training HighDropout model ...")
        dropout_history = train_baseline(
            dropout_model, train_loader, test_loader, device
        )

        torch.cuda.empty_cache()
        gc.collect()

        # ── Evaluate all four models ───────────────────────────────────────────
        b_acc, b_per_class = compute_accuracy(baseline_model, test_loader, device)
        b_shortcut = compute_average_shortcut_score(
            baseline_model, test_loader, device, n_batches=5
        )
        print_evaluation_report("Baseline", b_acc, b_per_class, b_shortcut)

        j_acc, j_per_class = compute_accuracy(jitter_model, test_loader, device)
        j_shortcut = compute_average_shortcut_score(
            jitter_model, test_loader, device, n_batches=5
        )
        print_evaluation_report("ColorJitter", j_acc, j_per_class, j_shortcut)

        d_acc, d_per_class = compute_accuracy(dropout_model, test_loader, device)
        d_shortcut = compute_average_shortcut_score(
            dropout_model, test_loader, device, n_batches=5
        )
        print_evaluation_report("HighDropout", d_acc, d_per_class, d_shortcut)

        s_acc, s_per_class = compute_accuracy(suppressed_model, test_loader, device)
        s_shortcut = compute_average_shortcut_score(
            suppressed_model, test_loader, device, n_batches=5
        )
        print_evaluation_report("Suppression", s_acc, s_per_class, s_shortcut)

        print("\n[Summary]")
        print(f"  Shortcut Score: {b_shortcut:.4f} → {s_shortcut:.4f}")
        delta = b_shortcut - s_shortcut
        print(f"  Reduction     : {delta:.4f} ({delta/b_shortcut:.1%})")

        # ── Attribution visualization ──────────────────────────────────────────
        print("\n[Visualization] Generating attribution comparison ...")
        sample_images, sample_labels, _ = next(iter(test_loader))

        visualize_attribution_comparison(
            baseline_model=baseline_model,
            suppressed_model=suppressed_model,
            images=sample_images,
            labels=sample_labels,
            device=device,
            save_path=os.path.join(config.RESULTS_DIR, "attribution_comparison.png"),
            n_samples=5,
        )

        # ── Training curves (only when we actually trained both) ───────────────
        if args.mode == 'full' and baseline_history and suppression_history:
            plot_training_curves(
                baseline_history, suppression_history,
                save_path=os.path.join(config.RESULTS_DIR, "training_curves.png"),
            )

    print("\n[Done] All tasks completed.")
    print(f"       Figures saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
