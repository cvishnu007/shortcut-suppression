# =============================================================================
# main.py — Entry Point  (Phase 3 update: Waterbirds added)
# =============================================================================
# USAGE:
#   python main.py --mode baseline
#   python main.py --mode suppress
#   python main.py --mode adversarial --load_baseline X --load_suppressed Y
#   python main.py --mode jtt
#   python main.py --mode waterbirds    ← NEW: Phase 3
#   python main.py --mode full
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

# Phase 3
from data.waterbirds import get_waterbirds_loaders
from models.resnet import get_waterbirds_model
from training.waterbirds_trainer import (
    train_waterbirds_baseline,
    train_waterbirds_suppression,
)
from evaluation.waterbirds_metrics import (
    evaluate_waterbirds,
    compute_waterbirds_shortcut_score,
    print_waterbirds_report,
    print_waterbirds_comparison,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shortcut Detection & Suppression"
    )
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=[
            'baseline', 'suppress', 'evaluate',
            'adversarial', 'jtt', 'waterbirds', 'full'
        ],
    )
    parser.add_argument('--lambda_shortcut', type=float, default=None)
    parser.add_argument('--use_discovery', action='store_true', default=False)
    parser.add_argument('--load_baseline', type=str, default=None)
    parser.add_argument('--load_suppressed', type=str, default=None)
    parser.add_argument('--load_jtt', type=str, default=None)
    return parser.parse_args()


# =============================================================================
# Adversarial evaluation block (Phase 1)
# =============================================================================

def run_adversarial_evaluation(
    baseline_model, suppressed_model, standard_test_loader, device
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
    )
    plot_per_class_adversarial(
        results=results,
        save_path=os.path.join(config.RESULTS_DIR, "adversarial_per_class.png"),
    )
    return summary


# =============================================================================
# Waterbirds block (Phase 3)
# =============================================================================

def run_waterbirds(device):
    """
    Trains baseline and suppressed ResNet-18 on Waterbirds and reports
    overall accuracy, worst-group accuracy, and shortcut score for both.
    """
    print("\n" + "="*64)
    print("  PHASE 3 — WATERBIRDS + RESNET-18")
    print("="*64)

    # Check dataset exists before spending time training
    meta = os.path.join(
        getattr(config, 'WATERBIRDS_DIR', ''), 'metadata.csv'
    )
    if not os.path.exists(meta):
        print(f"\n[ERROR] metadata.csv not found at {meta}")
        print("  Set WATERBIRDS_DIR in config.py to the folder containing metadata.csv")
        return

    train_loader, val_loader, test_loader = get_waterbirds_loaders()

    # ── Baseline ──────────────────────────────────────────────────────────
    wb_baseline = get_waterbirds_model(pretrained=True).to(device)
    baseline_history= train_waterbirds_baseline(
        wb_baseline, train_loader, val_loader, device
    )
    save_checkpoint(
        wb_baseline, None,
        epoch=getattr(config, 'WATERBIRDS_EPOCHS', 30),
        accuracy=max(baseline_history['val_worst_group']),
        path=os.path.join(config.CHECKPOINT_DIR, "waterbirds_baseline.pth"),
    )

    torch.cuda.empty_cache()
    gc.collect()

    # ── Suppression ───────────────────────────────────────────────────────
    wb_suppressed = get_waterbirds_model(pretrained=True).to(device)
    suppression_history,final_state,best_state = train_waterbirds_suppression(
        wb_suppressed, train_loader, val_loader, device
    )
    save_checkpoint(
        wb_suppressed, None,
        epoch=getattr(config, 'WATERBIRDS_EPOCHS', 30),
        accuracy=max(suppression_history['val_worst_group']),
        path=os.path.join(config.CHECKPOINT_DIR, "waterbirds_suppressed.pth"),
    )

    torch.cuda.empty_cache()
    gc.collect()

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[Waterbirds] Evaluating on test set ...")

    b_overall, b_wg, b_pg = evaluate_waterbirds(wb_baseline, test_loader, device)
    b_sc = compute_waterbirds_shortcut_score(
        wb_baseline, test_loader, device, n_batches=10
    )
    print_waterbirds_report(
        "Baseline (ERM)", b_overall, b_wg, b_pg, b_sc
    )
    wb_suppressed.load_state_dict(best_state)
    s_overall, s_wg, s_pg = evaluate_waterbirds(wb_suppressed, test_loader, device)
    # Shortcut score — from final converged epoch (suppression fully applied)
    wb_suppressed.load_state_dict(final_state)
    s_sc = compute_waterbirds_shortcut_score(
        wb_suppressed, test_loader, device, n_batches=10
    )
    wb_suppressed.load_state_dict(best_state)
    print_waterbirds_report(
        "Suppression (ours)", s_overall, s_wg, s_pg, s_sc
    )

    # ── Comparison table ──────────────────────────────────────────────────
    print_waterbirds_comparison({
        'Baseline (ERM)':      {'overall': b_overall, 'worst_group': b_wg, 'shortcut': b_sc},
        'Suppression (ours)':  {'overall': s_overall, 'worst_group': s_wg, 'shortcut': s_sc},
    })

    return {
        'baseline':   {'overall': b_overall, 'worst_group': b_wg, 'shortcut': b_sc},
        'suppressed': {'overall': s_overall, 'worst_group': s_wg, 'shortcut': s_sc},
    }


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

    # ── Waterbirds-only mode ───────────────────────────────────────────────
    if args.mode == 'waterbirds':
        run_waterbirds(device)
        print("\n[Done] Waterbirds evaluation complete.")
        return

    # ── MNIST modes (unchanged from Phase 2) ──────────────────────────────
    train_loader, test_loader = get_dataloaders()

    baseline_model   = get_model().to(device)
    suppressed_model = get_model().to(device)
    jtt_model        = get_model().to(device)

    if args.load_baseline and os.path.exists(args.load_baseline):
        baseline_model, _, _ = load_checkpoint(baseline_model, args.load_baseline, device)
    if args.load_suppressed and os.path.exists(args.load_suppressed):
        suppressed_model, _, _ = load_checkpoint(suppressed_model, args.load_suppressed, device)
    if args.load_jtt and os.path.exists(args.load_jtt):
        jtt_model, _, _ = load_checkpoint(jtt_model, args.load_jtt, device)

    baseline_history    = None
    suppression_history = None

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

    if args.mode in ['jtt', 'full'] and not args.load_jtt:
        jtt_model, jtt_history = train_jtt(train_loader, test_loader, device)
        if config.SAVE_BEST_MODEL:
            save_checkpoint(
                jtt_model, None, epoch=config.EPOCHS,
                accuracy=jtt_history['test_acc'][-1],
                path=os.path.join(config.CHECKPOINT_DIR, "jtt.pth"),
            )

    if args.mode in ['adversarial', 'full']:
        run_adversarial_evaluation(
            baseline_model, suppressed_model, test_loader, device
        )

    if args.mode in ['evaluate', 'full']:
        print("\n[Evaluation] Computing MNIST metrics ...")

        from data.dataloader import get_dataloaders_jitter
        jitter_train_loader, jitter_test_loader = get_dataloaders_jitter()
        jitter_model = get_model().to(device)
        print("\n[Baseline] Training ColorJitter model ...")
        train_baseline(jitter_model, jitter_train_loader, jitter_test_loader, device)

        dropout_model = get_model(dropout=0.8).to(device)
        print("\n[Baseline] Training HighDropout model ...")
        train_baseline(dropout_model, train_loader, test_loader, device)

        torch.cuda.empty_cache()
        gc.collect()

        models_to_eval = [
            ("Baseline",           baseline_model),
            ("ColorJitter",        jitter_model),
            ("HighDropout",        dropout_model),
            ("JTT",                jtt_model),
            ("Suppression (ours)", suppressed_model),
        ]

        results_table = {}
        for name, model in models_to_eval:
            acc, per_class = compute_accuracy(model, test_loader, device)
            sc = compute_average_shortcut_score(model, test_loader, device, n_batches=5)
            print_evaluation_report(name, acc, per_class, sc)
            results_table[name] = {'acc': acc, 'shortcut': sc}

        print(f"\n{'='*62}")
        print(f"  MNIST COMPARISON SUMMARY  (bias={config.BIAS_RATIO:.0%})")
        print(f"{'='*62}")
        print(f"  {'Method':<22} {'Test Acc':>10}  {'Shortcut Score':>15}")
        print(f"  {'-'*50}")
        for name, vals in results_table.items():
            marker = " ←" if name == "Suppression (ours)" else ""
            print(f"  {name:<22} {vals['acc']:>9.2%}  {vals['shortcut']:>14.4f}{marker}")
        print(f"{'='*62}")

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

    # ── Waterbirds in full mode ────────────────────────────────────────────
    if args.mode == 'full':
        run_waterbirds(device)

    print("\n[Done] All tasks completed.")
    print(f"       Figures saved to: {config.RESULTS_DIR}/")


if __name__ == "__main__":
    main()
