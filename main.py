# =============================================================================
# main.py — Entry Point
# =============================================================================
# Run this file to start the project.
#
# USAGE:
#   python main.py --mode baseline    # Train baseline (no suppression)
#   python main.py --mode suppress    # Train with shortcut suppression
#   python main.py --mode evaluate    # Evaluate both and visualize
#   python main.py --mode full        # Run everything end-to-end
# =============================================================================

import argparse
import torch

import config
from utils.helpers import set_seed, get_device
from data.dataloader import get_dataloaders
from models.cnn import get_model
from training.trainer import train_baseline, train_with_suppression
from evaluation.metrics import (
    compute_accuracy,
    compute_average_shortcut_score,
    print_evaluation_report
)
from evaluation.visualize import visualize_attribution_comparison, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Shortcut Detection & Suppression")
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['baseline', 'suppress', 'evaluate', 'full'],
        help="Which mode to run"
    )
    parser.add_argument(
        '--lambda_shortcut',
        type=float,
        default=None,
        help="Override lambda for shortcut penalty (default: from config.py)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(config.SEED)
    device = get_device()

    print(f"\n[Config] Mode: {args.mode}")
    print(f"[Config] Bias Ratio: {config.BIAS_RATIO}")
    print(f"[Config] Epochs: {config.EPOCHS}")
    print(f"[Config] Lambda: {config.LAMBDA_SHORTCUT}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders()

    # ── Models ────────────────────────────────────────────────────────────────
    baseline_model    = get_model().to(device)
    suppressed_model  = get_model().to(device)

    if args.mode in ['baseline', 'full']:
        # ── Train Baseline ─────────────────────────────────────────────────────
        baseline_history = train_baseline(
            baseline_model, train_loader, test_loader, device
        )

    if args.mode in ['suppress', 'full']:
        # Override lambda if specified via command line
        if args.lambda_shortcut is not None:
            config.LAMBDA_SHORTCUT = args.lambda_shortcut

        # ── Train with Suppression ─────────────────────────────────────────────
        suppression_history = train_with_suppression(
            suppressed_model, train_loader, test_loader, device
        )

    if args.mode in ['evaluate', 'full']:
        print("\n\n[Evaluation] Computing metrics...")

        # ── Evaluate Baseline ──────────────────────────────────────────────────
        b_acc, b_per_class = compute_accuracy(baseline_model, test_loader, device)
        b_shortcut = compute_average_shortcut_score(
            baseline_model, test_loader, device, n_batches=5
        )
        print_evaluation_report("Baseline", b_acc, b_per_class, b_shortcut)

        # ── Evaluate Suppression ───────────────────────────────────────────────
        s_acc, s_per_class = compute_accuracy(suppressed_model, test_loader, device)
        s_shortcut = compute_average_shortcut_score(
            suppressed_model, test_loader, device, n_batches=5
        )
        print_evaluation_report("Suppression", s_acc, s_per_class, s_shortcut)

        # ── Comparison Summary ─────────────────────────────────────────────────
        print("\n[Summary]")
        print(f"  Shortcut Score Reduction : {b_shortcut:.4f} → {s_shortcut:.4f}")
        delta = b_shortcut - s_shortcut
        print(f"  Improvement              : {delta:.4f} ({delta/b_shortcut:.1%} reduction)")

        # ── Visualize Attribution Maps ─────────────────────────────────────────
        print("\n[Visualization] Generating attribution comparison...")
        sample_images, sample_labels, _ = next(iter(test_loader))

        visualize_attribution_comparison(
            baseline_model=baseline_model,
            suppressed_model=suppressed_model,
            images=sample_images,
            labels=sample_labels,
            device=device,
            save_path=f"{config.RESULTS_DIR}/attribution_comparison.png",
            n_samples=5
        )

        if args.mode == 'full':
            plot_training_curves(
                baseline_history, suppression_history,
                save_path=f"{config.RESULTS_DIR}/training_curves.png"
            )

    print("\n[Done] All tasks completed.")


if __name__ == "__main__":
    main()
