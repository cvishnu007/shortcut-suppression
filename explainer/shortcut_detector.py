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
    bg_mask = get_background_mask(images)
    bg_mask = bg_mask.expand_as(attributions)
    abs_attr = attributions.abs()
    bg_attr  = (abs_attr * bg_mask).sum(dim=(1,2,3))
    total    = abs_attr.sum(dim=(1,2,3)) + 1e-8
    return (bg_attr / total).mean()
class ShortcutRegionDiscovery:
    """
    Discovers shortcut regions by finding pixels with LOW attribution
    variance across classes.

    The insight:
        Shortcut region = attended to similarly regardless of which class
                          is shown → LOW cross-class variance
        Real feature    = attended to differently per class (digit shapes
                          differ) → HIGH cross-class variance

    Algorithm:
        1. During warmup, accumulate per-class mean attribution maps
        2. After warmup, compute variance across the 10 class maps
        3. Bottom percentile of variance = suspected shortcut pixels
    """

    def __init__(self, image_size=28, num_classes=10, bottom_percent=0.4):
        """
        Args:
            image_size     : int   — spatial size (28 for MNIST)
            num_classes    : int   — number of classes (10 for MNIST)
            bottom_percent : float — fraction of LOWEST variance pixels
                             to flag as shortcut. 0.4 = bottom 40%.
        """
        self.image_size     = image_size
        self.num_classes    = num_classes
        self.bottom_percent = bottom_percent
        self.mask           = None

        # Per-class accumulators
        # class_sum[c]   : running sum of attribution maps for class c → (H, W)
        # class_count[c] : number of samples seen for class c
        self.class_sum   = {}
        self.class_count = {}
        for c in range(num_classes):
            self.class_sum[c]   = None
            self.class_count[c] = 0

    def accumulate(self, attributions, labels):
        """
        Accumulates per-class attribution maps during warmup.

        Args:
            attributions : Tensor (batch, 3, H, W) — detached, no grad
            labels       : Tensor (batch,)          — class labels
        """
        # Collapse RGB → single channel: sum of absolute values → (batch, H, W)
        abs_attr = attributions.detach().abs().sum(dim=1)

        for c in range(self.num_classes):
            # Find indices in batch belonging to class c
            mask_c = (labels == c)
            if mask_c.sum() == 0:
                continue

            # Mean attribution for class c in this batch → (H, W)
            class_attr = abs_attr[mask_c].mean(dim=0)

            if self.class_sum[c] is None:
                self.class_sum[c] = class_attr.clone()
            else:
                self.class_sum[c] += class_attr

            self.class_count[c] += 1

    def compute_mask(self, device):
        """
        Computes the shortcut mask after warmup ends.

        Steps:
            1. Compute per-class mean attribution maps
            2. Stack into (num_classes, H, W)
            3. Compute pixel-wise variance across classes
            4. Flag bottom_percent lowest-variance pixels as shortcut

        Returns:
            mask : Tensor (1, 1, H, W) — 1 = suspected shortcut pixel
        """
        class_means = []
        for c in range(self.num_classes):
            if self.class_sum[c] is None or self.class_count[c] == 0:
                # No samples seen for this class — use zeros
                h = self.image_size
                class_means.append(torch.zeros(h, h, device=device))
            else:
                mean_c = self.class_sum[c] / self.class_count[c]
                class_means.append(mean_c.to(device))

        # Stack → (num_classes, H, W)
        stacked = torch.stack(class_means, dim=0)

        # Pixel-wise variance across classes → (H, W)
        variance = stacked.var(dim=0)

        # Flatten to find threshold
        flat = variance.flatten()
        k    = int(len(flat) * self.bottom_percent)
        k    = max(k, 1)
        threshold = flat.kthvalue(k).values

        # Low variance = shortcut
        mask = (variance <= threshold).float()

        # Shape (1, 1, H, W) for broadcasting
        self.mask = mask.unsqueeze(0).unsqueeze(0).to(device)

        n_pixels = mask.sum().item()
        print(f"[Discovery] Cross-class variance mask computed — "
              f"{n_pixels:.0f}/{self.image_size**2} pixels flagged "
              f"({n_pixels/self.image_size**2:.1%})")

        return self.mask

    def get_mask(self, batch_size, device):
        """
        Returns the discovered mask expanded to match a batch.

        Args:
            batch_size : int
            device     : torch.device

        Returns:
            mask : Tensor (batch, 3, H, W)
        """
        if self.mask is None:
            raise RuntimeError("Mask not computed yet. Call compute_mask() first.")

        return self.mask.expand(
            batch_size, 3,
            self.image_size, self.image_size
        ).to(device)
def compute_shortcut_loss_discovered(attributions, discovered_mask):
    """
    Shortcut penalty loss using a DISCOVERED mask instead of brightness threshold.

    Args:
        attributions    : Tensor (batch, 3, H, W) — WITH gradients
        discovered_mask : Tensor (batch, 3, H, W) — from discovery.get_mask()

    Returns:
        loss : Scalar tensor — normalized shortcut penalty
    """
    abs_attr = attributions.abs()
    bg_attr  = (abs_attr * discovered_mask).sum(dim=(1, 2, 3))
    total    = abs_attr.sum(dim=(1, 2, 3)) + 1e-8
    return (bg_attr / total).mean()