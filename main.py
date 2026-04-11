# =============================================================================
# main.py — Entry Point  (Phase 2 update: JTT added)
# =============================================================================
# USAGE:
#   python main.py --mode baseline
#   python main.py --mode suppress
#   python main.py --mode adversarial --load_baseline X --load_suppressed Y
#   python main.py --mode jtt                          ← NEW
#   python main.py --mode full                         ← runs everything
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

# Phase 1
from data.adversarial_dataset import get_adversarial_loader
from evaluation.adversarial_metrics import (
    compute_adversarial_report,
    print_adversarial_comparison,
    plot_adversarial_bar_chart,
    plot_per_class_adversarial,
)

# Phase 2
from training.jtt_trainer import train_jtt


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shortcut Detection & Suppression"
    )
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['baseline', 'suppress', 'evaluate', 'adversarial', 'jtt', 'full'],
        help="Which mode to run."
    )
    parser.add_argument('--lambda_shortcut', type=float, default=None)
    parser.add_argument('--use_discovery', action='store_true', default=False)
    parser.add_argument('--load_baseline', type=str, default=None)
    parser.add_argument('--load_suppressed', type=str, default=None)
    parser.add_argument('--load_jtt', type=str, default=None)
    return parser.parse_args()


# =============================================================================
# Adversarial evaluation block (Phase 1 — unchanged)
# =============================================================================

def run_adversarial_evaluation(
    baseline_model, suppressed_model, standard_test_loader,
    device, extra_models=None
):
    print("\n" + "="*62)
    print("  PHASE 1 — ADVERSARIAL COLOR-SHIFT EVALUATION")
    print("="*62)

    adv_loader = get_adversarial_loader()
    torch.cuda.empty_cache()
    gc.collect()

    results = compute_adversarial_report(
        baseline_model=baseline_model,
        suppressed_model=suppressed_model,
        standard_loader=standard_test_loader,
        adv_loader=adv_loader,
        device=device,
    )

    summary = print_adversarial_comparison(results, bias_ratio=config.BIAS_RATIO)

    plot_adversarial_bar_chart(
        results=results,
        save_path=os.path.join(config.RESULTS_DIR, "adversarial_comparison.png"),
        bias_ratio=config.BIAS_RATIO,
        extra_models=extra_models,
    )
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

    set_seed(config.SEED)
    device = get_device()

    print(f"\n[Config] Mode            : {args.mode}")
    print(f"[Config] Bias Ratio      : {config.BIAS_RATIO}")
    print(f"[Config] Epochs          : {config.EPOCHS}")
    print(f"[Config] Lambda shortcut : {config.LAMBDA_SHORTCUT}")
    print(f"[Config] JTT T_id        : {getattr(config, 'JTT_ID_EPOCHS', 1)}")
    print(f"[Config] JTT lambda_up   : {getattr(config, 'JTT_LAMBDA_UP', 50)}")

    train_loader, test_loader = get_dataloaders()

    baseline_model   = get_model().to(device)
    suppressed_model = get_model().to(device)
    jtt_model        = get_model().to(device)

    # Load checkpoints if supplied
    if args.load_baseline and os.path.exists(args.load_baseline):
        baseline_model, _, _ = load_checkpoint(baseline_model, args.load_baseline, device)
    if args.load_suppressed and os.path.exists(args.load_suppressed):
        suppressed_model, _, _ = load_checkpoint(suppressed_model, args.load_suppressed, device)
    if args.load_jtt and os.path.exists(args.load_jtt):
        jtt_model, _, _ = load_checkpoint(jtt_model, args.load_jtt, device)

    baseline_history    = None
    suppression_history = None
    jtt_history         = None

    # ── Baseline ───────────────────────────────────────────────────────────
    if args.mode in ['baseline', 'full'] and not args.load_baseline:
        baseline_history = train_baseline(
            baseline_model, train_loader, test_loader, device
        )
        if config.SAVE_BEST_MODEL:
            save_checkpoint(
                baseline_model, None, epoch=config.EPOCHS,
                accuracy=baseline_history['test_acc'][-1],
                path=os.path.join(config.CHECKPOINT_DIR, "baseline.pth"),
            )

    # ── Suppression ────────────────────────────────────────────────────────
    if args.mode in ['suppress', 'full'] and not args.load_suppressed:
        if args.lambda_shortcut is not None:
            config.LAMBDA_SHORTCUT = args.lambda_shortcut
        suppression_history = train_with_suppression(
            suppressed_model, train_loader, test_loader, device,
            use_discovery=args.use_discovery,
        )
        if config.SAVE_BEST_MODEL:
            save_checkpoint(
                suppressed_model, None, epoch=config.EPOCHS,
                accuracy=suppression_history['test_acc'][-1],
                path=os.path.join(config.CHECKPOINT_DIR, "suppressed.pth"),
            )

    # ── JTT ────────────────────────────────────────────────────────────────
    if args.mode in ['jtt', 'full'] and not args.load_jtt:
        jtt_model, jtt_history = train_jtt(train_loader, test_loader, device)
        if config.SAVE_BEST_MODEL:
            save_checkpoint(
                jtt_model, None, epoch=config.EPOCHS,
                accuracy=jtt_history['test_acc'][-1],
                path=os.path.join(config.CHECKPOINT_DIR, "jtt.pth"),
            )

    # ── Adversarial evaluation (Phase 1) ───────────────────────────────────
    if args.mode in ['adversarial', 'full']:
        run_adversarial_evaluation(
            baseline_model=baseline_model,
            suppressed_model=suppressed_model,
            standard_test_loader=test_loader,
            device=device,
            extra_models=None,
        )

    # ── Full evaluation: all four models ───────────────────────────────────
    if args.mode in ['evaluate', 'full']:
        print("\n[Evaluation] Computing metrics for all models ...")

        from data.dataloader import get_dataloaders_jitter
        jitter_train_loader, jitter_test_loader = get_dataloaders_jitter()
        jitter_model = get_model().to(device)
        print("\n[Baseline] Training ColorJitter model ...")
        jitter_history = train_baseline(
            jitter_model, jitter_train_loader, jitter_test_loader, device
        )

        dropout_model = get_model(dropout=0.8).to(device)
        print("\n[Baseline] Training HighDropout model ...")
        dropout_history = train_baseline(
            dropout_model, train_loader, test_loader, device
        )

        torch.cuda.empty_cache()
        gc.collect()

        # Evaluate all five models
        models_to_eval = [
            ("Baseline",          baseline_model),
            ("ColorJitter",       jitter_model),
            ("HighDropout",       dropout_model),
            ("JTT",               jtt_model),
            ("Suppression (ours)", suppressed_model),
        ]

        results_table = {}
        for name, model in models_to_eval:
            acc, per_class = compute_accuracy(model, test_loader, device)
            sc = compute_average_shortcut_score(model, test_loader, device, n_batches=5)
            print_evaluation_report(name, acc, per_class, sc)
            results_table[name] = {'acc': acc, 'shortcut': sc}

        # Summary comparison table
        print(f"\n{'='*62}")
        print(f"  FULL COMPARISON SUMMARY  (bias={config.BIAS_RATIO:.0%})")
        print(f"{'='*62}")
        print(f"  {'Method':<22} {'Test Acc':>10}  {'Shortcut Score':>15}")
        print(f"  {'-'*50}")
        for name, vals in results_table.items():
            marker = " ←" if name == "Suppression (ours)" else ""
            print(f"  {name:<22} {vals['acc']:>9.2%}  {vals['shortcut']:>14.4f}{marker}")
        print(f"{'='*62}")

        b_sc = results_table['Baseline']['shortcut']
        s_sc = results_table['Suppression (ours)']['shortcut']
        j_sc = results_table['JTT']['shortcut']
        print(f"\n  Shortcut score vs Baseline:")
        print(f"    JTT               : {b_sc:.4f} → {j_sc:.4f}  ({(b_sc-j_sc)/b_sc:.1%} reduction)")
        print(f"    Suppression (ours): {b_sc:.4f} → {s_sc:.4f}  ({(b_sc-s_sc)/b_sc:.1%} reduction)")

        # Attribution visualization
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

        if args.mode == 'full' and baseline_history and suppression_history:
            plot_training_curves(
                baseline_history, suppression_history,
                save_path=os.path.join(config.RESULTS_DIR, "training_curves.png"),
            )

    print("\n[Done] All tasks completed.")
    print(f"       Figures saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
