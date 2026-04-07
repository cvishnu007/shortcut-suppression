# =============================================================================
# data/colored_mnist.py — Biased Colored MNIST Dataset
# =============================================================================
# WHAT THIS FILE DOES:
#   Takes standard MNIST (black & white digit images) and colors them.
#   Each digit class (0–9) gets assigned a "shortcut color" (e.g., digit 0 → red).
#   During TRAINING: most images are colored with the shortcut color.
#   During TESTING:  images are colored randomly (no shortcut).
#
# WHY: The model will learn "red → class 0" instead of "shape → class 0".
#      Our project will detect and suppress this cheating.
# =============================================================================

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import config


# Define one distinct color per digit class (0–9)
# These are RGB values in range [0, 255]
CLASS_COLORS = {
    0: (255, 0,   0  ),    # Red
    1: (0,   255, 0  ),    # Green
    2: (0,   0,   255),    # Blue
    3: (255, 255, 0  ),    # Yellow
    4: (255, 0,   255),    # Magenta
    5: (0,   255, 255),    # Cyan
    6: (255, 128, 0  ),    # Orange
    7: (128, 0,   255),    # Purple
    8: (0,   128, 255),    # Sky Blue
    9: (255, 0,   128),    # Pink
}

def colorize_image(grayscale_img, color_rgb):
    """
    Colors the BACKGROUND of the image, keeps the digit white.
    Background color = shortcut. Digit shape = real signal.
    """
    img_array = np.array(grayscale_img, dtype=np.float32) / 255.0

    # digit_mask: True where digit pixels are, False where background is
    digit_mask = img_array > 0.1

    rgb_array = np.zeros((*img_array.shape, 3), dtype=np.float32)

    # Background pixels → fill with shortcut color
    for c, val in enumerate(color_rgb):
        rgb_array[:, :, c] = np.where(digit_mask, 0.0, val / 255.0)

    # Digit pixels → keep white
    for c in range(3):
        rgb_array[:, :, c] = np.where(digit_mask, 1.0, rgb_array[:, :, c])

    rgb_array = (rgb_array * 255).astype(np.uint8)
    return Image.fromarray(rgb_array, mode='RGB')


class ColoredMNIST(Dataset):
    """
    Custom PyTorch Dataset that wraps MNIST with color shortcuts.

    Args:
        root     : Path to download/store raw MNIST data
        train    : True = training set, False = test set
        bias     : Fraction of training samples with shortcut color (0.0 to 1.0)
                   At test time, bias is always 0 (random colors, no shortcut).
        download : Whether to download MNIST if not found
    """

    def __init__(self, root, train=True, bias=0.95, download=True):
        # Load raw MNIST dataset (grayscale, no transforms yet)
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=None   # We'll handle transforms manually
        )

        self.train = train
        self.bias = bias if train else 0.0   # No bias at test time

        # Store the transform we'll apply after colorizing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),       # Converts PIL Image to (C, H, W) tensor
            transforms.Normalize(        # Normalize each channel to mean=0.5, std=0.5
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        # Pre-generate color assignments for each sample
        self.color_assignments = self._assign_colors()

    def _assign_colors(self):
        """
        Decide which color each image gets, based on the bias ratio.

        For biased training:
          - With probability `bias`: use the shortcut color (class color)
          - With probability `1 - bias`: use a random color

        For unbiased testing:
          - Always use a random color
        """
        color_assignments = []

        for idx in range(len(self.mnist)):
            label = int(self.mnist.targets[idx])

            if self.train:
                # Flip a biased coin
                use_shortcut = np.random.random() < self.bias

                if use_shortcut:
                    # Assign the class's shortcut color
                    color = CLASS_COLORS[label]
                else:
                    # Assign a random color (not necessarily this class's color)
                    random_class = np.random.randint(0, 10)
                    color = CLASS_COLORS[random_class]
            else:
                # At test time: always random color
                random_class = np.random.randint(0, 10)
                color = CLASS_COLORS[random_class]

            color_assignments.append(color)

        return color_assignments

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Returns one sample from the dataset.

        Returns:
            image : Tensor of shape (3, 28, 28) — the colored digit
            label : int — the digit class (0–9)
            color : tuple — the RGB color used (useful for debugging/analysis)
        """
        # Get the raw grayscale image and label
        raw_img, label = self.mnist[idx]

        # Get the pre-assigned color for this image
        color = self.color_assignments[idx]

        # Color the image
        colored_img = colorize_image(raw_img, color)

        # Apply normalization and convert to tensor
        image_tensor = self.to_tensor(colored_img)

        return image_tensor, label, color
