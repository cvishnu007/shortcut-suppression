# =============================================================================
# models/resnet.py — ResNet-18 for Waterbirds with Grad-CAM
# =============================================================================
#
# WHY RESNET-18 INSTEAD OF SIMPLECNN:
#   Waterbirds contains real photographs (224×224 RGB).  A SimpleCNN with
#   400K parameters cannot learn meaningful features from real images in
#   30 epochs.  ResNet-18 pretrained on ImageNet already knows edges,
#   textures, shapes — we fine-tune its final layer to distinguish
#   landbirds from waterbirds.
#
# WHY GRAD-CAM INSTEAD OF INTEGRATED GRADIENTS:
#   IG requires n_steps forward passes per batch (25 by default).
#   On ResNet-18 with batch_size=64, that is 25 × 64 = 1,600 forward
#   passes per batch — about 40 minutes per epoch on an RTX 3050 4GB.
#   Grad-CAM requires exactly ONE forward pass and ONE backward pass,
#   same cost as regular training.  It produces a spatially-equivalent
#   attribution map (heatmap over spatial locations) that serves the
#   same purpose for the shortcut penalty.
#
# HOW GRAD-CAM WORKS HERE:
#   We register a forward hook on the last conv layer (layer4).
#   During the backward pass, we also capture the gradients at that layer.
#   The Grad-CAM map is: relu(sum over channels of (grad * activation))
#   Upsampled to 224×224, this tells us which spatial regions the model
#   used to make its prediction — equivalent to an attribution map.
#
# SHORTCUT PENALTY WITH GRAD-CAM:
#   Waterbirds images have the bird composited onto the background.
#   The bird occupies the center of the image (CUB bounding boxes are
#   centered).  The background fills the periphery.
#   We define the shortcut region as the BORDER of the image (outer 30%)
#   and penalize attribution landing there.  This is analogous to the
#   background mask in Colored MNIST.
#
# PLACEMENT:  save to  models/resnet.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18Waterbirds(nn.Module):
    """
    ResNet-18 fine-tuned for Waterbirds binary classification.

    Attributes:
        backbone     : ResNet-18 up to (and including) layer4
        classifier   : replaced final FC layer (512 → 2)
        _activations : dict holding the last layer4 feature map (set by hook)
        _gradients   : dict holding the last layer4 gradients (set by hook)
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Keep everything except the final FC layer
        self.backbone   = nn.Sequential(*list(backbone.children())[:-2])
        # backbone output: (batch, 512, 7, 7) for 224×224 input

        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        # Storage for Grad-CAM hooks
        self._activations = {}
        self._gradients   = {}

        # Register hooks on the last conv block (layer4 = backbone[-1])
        self.backbone[-1].register_forward_hook(self._save_activation)
        self.backbone[-1].register_full_backward_hook(self._save_gradient)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ResNet18] {total:,} trainable parameters  "
              f"({'pretrained' if pretrained else 'random init'})")

    # ── Hooks ─────────────────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        """Forward hook — stores layer4 feature maps."""
        self._activations['layer4'] = output  # (batch, 512, 7, 7)

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook — stores gradients w.r.t. layer4 output."""
        self._gradients['layer4'] = grad_output[0]  # (batch, 512, 7, 7)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x : Tensor (batch, 3, 224, 224)
        Returns:
            logits : Tensor (batch, 2)
        """
        features = self.backbone(x)          # (batch, 512, 7, 7)
        pooled   = self.avgpool(features)    # (batch, 512, 1, 1)
        pooled   = pooled.flatten(1)         # (batch, 512)
        logits   = self.classifier(pooled)   # (batch, 2)
        return logits

    # ── Grad-CAM ──────────────────────────────────────────────────────────

    def get_gradcam(self, target_size=224):
        """
        Computes Grad-CAM heatmap from stored activations and gradients.

        Must be called AFTER a backward pass has been run.
        Returns a heatmap in [0,1] upsampled to target_size × target_size.

        Args:
            target_size : int — spatial size to upsample to (default 224)

        Returns:
            heatmap : Tensor (batch, 1, target_size, target_size)
                      Values in [0, 1].  High = model attended here.
        """
        acts  = self._activations['layer4']   # (batch, 512, 7, 7)
        grads = self._gradients['layer4']     # (batch, 512, 7, 7)

        # Global average pool the gradients over spatial dims → weights
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (batch, 512, 1, 1)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (batch, 1, 7, 7)
        cam = F.relu(cam)                                 # only positive influence

        # Upsample to image resolution
        cam = F.interpolate(
            cam, size=(target_size, target_size),
            mode='bilinear', align_corners=False
        )                                                 # (batch, 1, H, W)

        # Normalize per-image to [0, 1]
        b = cam.shape[0]
        cam_flat = cam.view(b, -1)
        cam_min  = cam_flat.min(dim=1).values.view(b, 1, 1, 1)
        cam_max  = cam_flat.max(dim=1).values.view(b, 1, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam   # (batch, 1, H, H) in [0,1]


# =============================================================================
# Background mask for Waterbirds
# =============================================================================

def get_border_mask(batch_size, image_size=224, border_fraction=0.3, device='cuda'):
    """
    Returns a binary mask where 1 = border region (shortcut) and 0 = center.

    In Waterbirds, birds are composited at the center of the image.
    The background (shortcut) fills the periphery.
    We flag the outer `border_fraction` of the image on each side as
    the shortcut region.

    Args:
        batch_size      : int
        image_size      : int   (default 224)
        border_fraction : float (default 0.3 = outer 30% on each side)
        device          : torch.device or str

    Returns:
        mask : Tensor (batch, 1, image_size, image_size) — float, 0 or 1
               1 = border (background shortcut region)
               0 = center (bird region)
    """
    border = int(image_size * border_fraction)

    mask = torch.ones(image_size, image_size, device=device)
    mask[border:image_size - border, border:image_size - border] = 0.0

    return mask.view(1, 1, image_size, image_size).expand(
        batch_size, 1, image_size, image_size
    )


# =============================================================================
# Shortcut loss using Grad-CAM
# =============================================================================

def compute_gradcam_shortcut_loss(model, images, labels, border_fraction=0.3):
    """
    Computes the shortcut penalty using Grad-CAM attributions.

    This is called INSIDE the training loop after the backward pass on
    task loss, so gradients and activations are already populated.

    Strategy:
        1. Get Grad-CAM heatmap from model hooks (already populated)
        2. Get border mask (shortcut region)
        3. Loss = mean attribution in border / total attribution

    Args:
        model           : ResNet18Waterbirds (hooks already fired)
        images          : Tensor (batch, 3, 224, 224)
        labels          : Tensor (batch,) — not used directly but kept
                          for interface consistency
        border_fraction : float — outer fraction flagged as shortcut

    Returns:
        sc_loss : scalar Tensor — normalized shortcut penalty
    """
    cam  = model.get_gradcam(target_size=images.shape[-1])  # (batch,1,224,224)
    mask = get_border_mask(
        images.shape[0], images.shape[-1],
        border_fraction=border_fraction,
        device=images.device
    )

    border_attr = (cam * mask).sum(dim=(1, 2, 3))
    total_attr  = cam.sum(dim=(1, 2, 3)) + 1e-8

    return (border_attr / total_attr).mean()


# =============================================================================
# Factory
# =============================================================================

def get_waterbirds_model(pretrained=True):
    """Returns a ResNet18Waterbirds model ready for training."""
    return ResNet18Waterbirds(num_classes=2, pretrained=pretrained)
