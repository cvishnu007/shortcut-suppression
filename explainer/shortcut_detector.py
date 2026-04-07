# =============================================================================
# explainer/shortcut_detector.py — Detect Shortcuts from Attributions
# =============================================================================
# WHAT THIS FILE DOES:
#   Given an attribution map (what the model focused on),
#   determines whether the model is using a shortcut (spurious feature).
#
# HOW WE DETECT SHORTCUTS FOR COLORED MNIST:
#   In Colored MNIST:
#     - The DIGIT PIXELS are the real signal (what we WANT the model to use)
#     - The BACKGROUND PIXELS are the shortcut (what we DON'T want)
#
#   We compute:
#     shortcut_score = (attribution in background) / (total attribution)
#
#   If shortcut_score > threshold → model is cheating
#   If shortcut_score < threshold → model is behaving
# =============================================================================

import torch
import numpy as np
import config


def get_foreground_mask(images):
    """
    Digit pixels are WHITE (all 3 channels high).
    Background pixels are COLORED (at least one channel is low).
    So foreground = pixels where ALL channels are high.
    """
    # In normalized space (mean=0.5, std=0.5), white pixels → 1.0 on all channels
    # Colored pixels → at least one channel is -1.0
    # Take the minimum across channels — white pixels have min ≈ 1.0
    min_channel = images.min(dim=1).values        # (batch, H, W)
    mask = (min_channel > 0.5).float().unsqueeze(1)  # (batch, 1, H, W)
    return mask


def get_background_mask(images):
    """
    Opposite of foreground mask — 1 = background (shortcut region).

    Args:
        images : Tensor of shape (batch, 3, H, W)

    Returns:
        mask : Tensor of shape (batch, 1, H, W) — float, 0 or 1
               1 = background (shortcut), 0 = digit (real signal)
    """
    return 1.0 - get_foreground_mask(images)


def compute_shortcut_score(attributions, images):
    """
    Computes how much attribution lands in the background (shortcut region).

    A HIGH score means the model is heavily relying on background color → shortcut.
    A LOW score means the model is focusing on the digit shape → good behavior.

    Args:
        attributions : Tensor of shape (batch, 3, H, W) — from compute_attribution_batch()
        images       : Tensor of shape (batch, 3, H, W) — the input images

    Returns:
        shortcut_scores : Tensor of shape (batch,) — one score per image
                          Values between 0 and 1.
                          0 = all attribution on digit
                          1 = all attribution on background

    MATH:
        score_i = sum(|attr_i| * bg_mask_i) / (sum(|attr_i|) + ε)
    """
    # Get background mask: 1 = background, 0 = digit
    bg_mask = get_background_mask(images)    # (batch, 1, H, W)

    # Expand mask to cover all 3 channels
    bg_mask = bg_mask.expand_as(attributions)   # (batch, 3, H, W)

    # Compute absolute attribution (we care about magnitude, not direction)
    abs_attr = attributions.abs()                # (batch, 3, H, W)

    # Attribution in background region
    bg_attribution = (abs_attr * bg_mask).sum(dim=(1, 2, 3))   # (batch,)

    # Total attribution
    total_attribution = abs_attr.sum(dim=(1, 2, 3)) + 1e-8     # (batch,) + small ε

    # Shortcut score = fraction of attribution in background
    shortcut_scores = bg_attribution / total_attribution         # (batch,)

    return shortcut_scores


def is_shortcut(shortcut_scores, threshold=None):
    """
    Returns a boolean mask: True = this sample uses a shortcut.

    Args:
        shortcut_scores : Tensor of shape (batch,) — from compute_shortcut_score()
        threshold       : float — if score > threshold, it's a shortcut
                          (default: uses config.SHORTCUT_THRESHOLD)

    Returns:
        mask : Tensor of shape (batch,) — True where shortcut is detected
    """
    if threshold is None:
        threshold = config.SHORTCUT_THRESHOLD

    return shortcut_scores > threshold


def compute_shortcut_loss(attributions, images):
    """
    Computes the shortcut penalty loss for training.

    This is the KEY contribution of our project.
    By adding this to the training loss, we punish the model for
    having high attribution in background (shortcut) regions.

    The model is then forced to find other features (the digit shape)
    to minimize this penalty.

    Args:
        attributions : Tensor of shape (batch, 3, H, W) — MUST have gradients
        images       : Tensor of shape (batch, 3, H, W)

    Returns:
        loss : Scalar tensor — the shortcut penalty
               This can be directly backpropagated.

    FORMULA:
        L_shortcut = mean over batch of:
                     sum of (|attribution| * background_mask)

    WHY MEAN AND NOT SUM?
        Mean keeps the loss scale consistent regardless of batch size.
    """
    # Get background mask
    bg_mask = get_background_mask(images)              # (batch, 1, H, W)
    bg_mask = bg_mask.expand_as(attributions)          # (batch, 3, H, W)

    # How much attribution lands on background, per image
    bg_attribution = (attributions.abs() * bg_mask)    # (batch, 3, H, W)

    # Average over the whole batch → scalar
    loss = bg_attribution.mean()

    return loss
