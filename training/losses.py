# =============================================================================
# training/losses.py — Loss Functions
# =============================================================================
# WHAT THIS FILE DOES:
#   Defines the combined loss used during shortcut-suppression training.
#
# OUR TOTAL LOSS:
#   L_total = L_task + λ × L_shortcut
#
#   L_task     = CrossEntropyLoss (standard classification loss)
#   L_shortcut = Penalty for attribution falling in background region
#   λ (lambda) = Controls how much we penalize shortcut use (from config)
#
# WHY TWO LOSSES?
#   L_task alone makes the model accurate but allows cheating.
#   L_shortcut alone makes the model ignore backgrounds but may lose accuracy.
#   Together, they push the model toward accurate AND honest predictions.
# =============================================================================

import torch
import torch.nn as nn
from explainer.shortcut_detector import compute_shortcut_loss
import config


class TaskLoss(nn.Module):
    """
    Standard Cross-Entropy Loss for classification.

    Cross-Entropy measures how wrong the model's class probabilities are
    compared to the true label. Minimizing this makes predictions more accurate.

    Lower CE loss = model is more confident in the correct class.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits : Tensor (batch, num_classes) — raw model output (before softmax)
            labels : Tensor (batch,) — ground truth class indices

        Returns:
            Scalar loss value
        """
        return self.ce(logits, labels)


class ShortcutSuppressionLoss(nn.Module):
    """
    Combined loss = Task Loss + λ × Shortcut Penalty.

    This is the MAIN loss used during shortcut-suppression training.

    Args:
        lambda_shortcut : float — weight of the shortcut penalty
                          Higher = stronger suppression (but may hurt accuracy)
                          Lower  = lighter suppression (but shortcuts may persist)
    """

    def __init__(self, lambda_shortcut=None):
        super().__init__()
        self.lambda_shortcut = lambda_shortcut or config.LAMBDA_SHORTCUT
        self.task_loss_fn = TaskLoss()

    def forward(self, logits, labels, attributions, images):
        """
        Computes the total combined loss.

        Args:
            logits       : Tensor (batch, num_classes) — model output
            labels       : Tensor (batch,) — true labels
            attributions : Tensor (batch, 3, H, W) — explanation maps (WITH gradients)
            images       : Tensor (batch, 3, H, W) — input images (for mask generation)

        Returns:
            total_loss     : Scalar — the combined loss for backprop
            task_loss_val  : float — task loss value (for logging)
            sc_loss_val    : float — shortcut loss value (for logging)
        """
        # ── Task Loss ─────────────────────────────────────────────────────────
        task_loss = self.task_loss_fn(logits, labels)

        # ── Shortcut Penalty Loss ─────────────────────────────────────────────
        sc_loss = compute_shortcut_loss(attributions, images)

        # ── Combine ───────────────────────────────────────────────────────────
        total_loss = task_loss + self.lambda_shortcut * sc_loss

        # Return all three for logging purposes
        return total_loss, task_loss.item(), sc_loss.item()
