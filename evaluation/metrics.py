# =============================================================================
# evaluation/metrics.py — Evaluation Metrics
# =============================================================================
# WHAT THIS FILE DOES:
#   Computes all the metrics we need to evaluate our project:
#   1. Standard accuracy (how correct is the model)
#   2. Shortcut score (is the model cheating)
#   3. Per-class accuracy (is the model fair across all digit classes)
# =============================================================================

import torch
import numpy as np
from captum.attr import IntegratedGradients
from explainer.shortcut_detector import compute_shortcut_score, get_background_mask


def compute_accuracy(model, data_loader, device):
    """
    Computes overall and per-class accuracy.

    Args:
        model       : PyTorch model
        data_loader : DataLoader
        device      : torch.device

    Returns:
        overall_acc  : float — overall accuracy across all classes
        per_class    : dict  — {class_id: accuracy} for each digit 0–9
    """
    model.eval()

    class_correct = {i: 0 for i in range(10)}
    class_total   = {i: 0 for i in range(10)}

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            for pred, label in zip(predictions, labels):
                label_id = label.item()
                class_total[label_id] += 1
                if pred.item() == label_id:
                    class_correct[label_id] += 1

    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc   = total_correct / total_samples

    per_class = {
        i: class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0.0
        for i in range(10)
    }

    return overall_acc, per_class


def compute_average_shortcut_score(model, data_loader, device, n_batches=10):
    """
    Computes average shortcut score across multiple batches.

    A high score = model is relying on background (shortcut) region.
    A low score  = model is relying on digit shape (real feature).

    Args:
        model       : PyTorch model
        data_loader : DataLoader
        device      : torch.device
        n_batches   : int — how many batches to evaluate (attribution is slow)

    Returns:
        avg_score : float — average shortcut score (0 to 1)
    """
    model.eval()
    ig = IntegratedGradients(model)
    all_scores = []

    for batch_idx, (images, labels, _) in enumerate(data_loader):
        if batch_idx >= n_batches:
            break   # Don't evaluate all batches — attribution is expensive

        images = images.to(device)
        labels = labels.to(device)
        images_attr = images.detach().clone().requires_grad_(True)

        baseline = torch.zeros_like(images_attr)

        with torch.no_grad():
            # We use no_grad here since we don't need gradients for evaluation
            pass

        # Compute attributions for this batch
        attributions = ig.attribute(
            inputs=images_attr,
            baselines=baseline,
            target=labels,
            n_steps=25
        )

        scores = compute_shortcut_score(attributions.detach(), images)
        all_scores.extend(scores.cpu().numpy().tolist())

    return float(np.mean(all_scores))


def print_evaluation_report(model_name, overall_acc, per_class, shortcut_score):
    """
    Prints a formatted evaluation report.

    Args:
        model_name    : str — "Baseline" or "Suppression"
        overall_acc   : float
        per_class     : dict {class: accuracy}
        shortcut_score: float
    """
    print(f"\n{'='*50}")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"{'='*50}")
    print(f"  Overall Accuracy   : {overall_acc:.2%}")
    print(f"  Avg Shortcut Score : {shortcut_score:.4f}  (lower = better)")
    print(f"\n  Per-Class Accuracy:")
    for cls, acc in per_class.items():
        bar = '█' * int(acc * 20)
        print(f"    Digit {cls}: {acc:.2%}  {bar}")
    print(f"{'='*50}")
