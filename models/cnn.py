# =============================================================================
# models/cnn.py — Simple CNN Model
# =============================================================================
# WHAT THIS FILE DOES:
#   Defines a basic Convolutional Neural Network for classifying MNIST digits.
#
# WHY SIMPLE?
#   We intentionally use a simple model so:
#   1. Training is fast (2-month project)
#   2. Explanations are more interpretable
#   3. The shortcut behavior is easy to observe
#
# ARCHITECTURE:
#   Input (3,28,28) → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Output
# =============================================================================

import torch
import torch.nn as nn
import config


class SimpleCNN(nn.Module):
    """
    A simple 2-layer CNN for digit classification.

    Forward pass shape trace:
        Input  : (batch, 3, 28, 28)    ← batch of RGB images
        After conv1 + pool : (batch, 32, 13, 13)
        After conv2 + pool : (batch, 64, 5,  5)
        After flatten      : (batch, 1600)
        After fc1          : (batch, 128)
        After fc2          : (batch, 10)    ← 10 digit classes
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(SimpleCNN, self).__init__()

        # ── Convolutional Block 1 ─────────────────────────────────────────────
        # Learns low-level features: edges, colors, textures
        self.conv1 = nn.Conv2d(
            in_channels=3,       # RGB input
            out_channels=32,     # 32 different feature detectors
            kernel_size=3,       # 3x3 filters
            padding=1            # Keeps spatial size the same
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # MaxPool halves spatial dims: 28x28 → 14x14

        # ── Convolutional Block 2 ─────────────────────────────────────────────
        # Learns higher-level features: shapes, digit parts
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # MaxPool: 14x14 → 7x7

        # ── Activation Function ───────────────────────────────────────────────
        self.relu = nn.ReLU()

        # ── Dropout for Regularization ────────────────────────────────────────
        # Randomly zeros out 50% of neurons during training.
        # Prevents the model from overfitting to noise.
        self.dropout = nn.Dropout(p=0.5)

        # ── Fully Connected Layers ────────────────────────────────────────────
        # These combine all extracted features to make the final class prediction
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # 64 channels × 7×7 spatial
        self.fc2 = nn.Linear(128, num_classes)   # Final output: 10 class scores

    def forward(self, x):
        """
        Defines the forward pass (how input flows through the network).

        Args:
            x : Tensor of shape (batch_size, 3, 28, 28)

        Returns:
            Tensor of shape (batch_size, 10) — raw class scores (logits)
        """
        # Block 1: Detect low-level features
        x = self.relu(self.conv1(x))     # (batch, 32, 28, 28)
        x = self.pool1(x)                 # (batch, 32, 14, 14)

        # Block 2: Detect higher-level features
        x = self.relu(self.conv2(x))     # (batch, 64, 14, 14)
        x = self.pool2(x)                 # (batch, 64,  7,  7)

        # Flatten: convert 3D feature maps to 1D vector for FC layers
        x = x.reshape(x.size(0), -1)        # (batch, 64*7*7) = (batch, 3136)

        # Fully connected classification head
        x = self.dropout(self.relu(self.fc1(x)))   # (batch, 128)
        x = self.fc2(x)                             # (batch, 10)

        return x

    def get_feature_maps(self, x):
        """
        Returns intermediate feature maps (useful for visualization).
        This is used by the explainer module to understand what the model sees.
        """
        features = {}

        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        features['block1'] = x.detach()

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        features['block2'] = x.detach()

        return features


def get_model(dropout=0.5):
    model = SimpleCNN(num_classes=config.NUM_CLASSES)
    model.dropout = nn.Dropout(p=dropout)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] SimpleCNN created — {total_params:,} trainable parameters")
    return model
