# =============================================================================
# explainer/attribution.py — Generate Explanation Maps
# =============================================================================
# WHAT THIS FILE DOES:
#   Uses Captum's Integrated Gradients to compute attribution maps.
#   An attribution map answers: "Which pixels contributed most to this prediction?"
#
# INTEGRATED GRADIENTS (simple explanation):
#   Imagine walking from a black image (baseline) to the actual image in
#   small steps. At each step, measure how much the prediction changes.
#   Sum all those changes → you get each pixel's total contribution.
#
# OUTPUT:
#   A tensor the same size as the input image, where each value represents
#   how much that pixel "influenced" the model's decision.
#   High value = that pixel matters a lot.
#   Low value  = that pixel is ignored.
# =============================================================================

import torch
from captum.attr import IntegratedGradients, NoiseTunnel
import numpy as np


def compute_attribution(model, image, label, device, n_steps=50):
    """
    Computes Integrated Gradients attribution for a single image.

    Args:
        model  : PyTorch model (must be in eval mode before calling)
        image  : Tensor of shape (1, 3, 28, 28) — ONE image (batch size = 1)
        label  : int — the true label (or predicted label) to explain
        device : torch.device (cpu or cuda)
        n_steps: int — more steps = more accurate, but slower (50 is fine)

    Returns:
        attribution : Tensor of shape (1, 3, 28, 28)
                      Each value = how much that pixel contributed
                      Positive = pushed toward the class
                      Negative = pushed away from the class

    USAGE:
        attr = compute_attribution(model, img.unsqueeze(0), label=3, device=device)
        # attr shape: (1, 3, 28, 28)
    """
    model.eval()   # Ensure model is in eval mode (no dropout, etc.)

    # Move image to the correct device (CPU or GPU)
    image = image.to(device)
    image.requires_grad_(True)   # Enable gradient computation through the image

    # Create IntegratedGradients explainer
    # It takes your model's forward function as input
    ig = IntegratedGradients(model)

    # Baseline = all-zeros image (black — represents "no information")
    # Attribution measures the difference from baseline to actual input
    baseline = torch.zeros_like(image)

    # Compute attributions
    # internal_batch_size: process n_steps in batches to save memory
    attribution = ig.attribute(
        inputs=image,
        baselines=baseline,
        target=label,                   # Which output neuron to explain
        n_steps=n_steps,
        internal_batch_size=20          # Lower if you run out of memory
    )

    return attribution.detach()         # Detach from computation graph


def compute_attribution_batch(model, images, labels, device, n_steps=25):
    """
    Computes attributions for a batch of images (more efficient).

    Args:
        model  : PyTorch model
        images : Tensor of shape (batch, 3, 28, 28)
        labels : Tensor of shape (batch,) — one label per image
        device : torch.device

    Returns:
        attributions : Tensor of shape (batch, 3, 28, 28)

    NOTE:
        Computing attributions is expensive (many forward passes).
        For training, we compute them for each batch.
        For evaluation, we might compute for fewer samples.
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad_(True)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(images)

    attributions = ig.attribute(
        inputs=images,
        baselines=baseline,
        target=labels,
        n_steps=n_steps,
        internal_batch_size=10
    )

    return attributions.detach()


def attribution_to_heatmap(attribution):
    """
    Converts an attribution tensor to a single-channel heatmap for visualization.

    Attribution tensors have 3 color channels (RGB).
    We collapse them to one channel by taking the absolute value and summing.
    This tells us WHERE attention is, not which direction it goes.

    Args:
        attribution : Tensor of shape (1, 3, H, W) or (3, H, W)

    Returns:
        heatmap : numpy array of shape (H, W) — values in [0, 1]
    """
    # Handle both (1, 3, H, W) and (3, H, W) input shapes
    if attribution.dim() == 4:
        attribution = attribution.squeeze(0)   # Remove batch dim → (3, H, W)

    # Collapse RGB channels: sum of absolute values
    heatmap = attribution.abs().sum(dim=0)     # → (H, W)

    # Convert to numpy for matplotlib
    heatmap = heatmap.cpu().numpy()

    # Normalize to [0, 1] for display
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap
