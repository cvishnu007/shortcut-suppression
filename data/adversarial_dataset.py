# =============================================================================
# data/adversarial_dataset.py — Adversarial Evaluation Split
# =============================================================================
#
# PURPOSE:
#   This dataset is the empirical "killer test" for shortcut learning.
#
#   Standard test set:  colors are random → model can sometimes get lucky
#                       by guessing the right color by chance (~10% hit rate).
#
#   Adversarial split:  every digit gets a color belonging to a DIFFERENT
#                       digit class, guaranteed.  A model that learned the
#                       color shortcut will confidently predict the WRONG class
#                       on every image.  Accuracy collapses to ~10% (chance).
#
#   A model that learned digit SHAPE is completely unaffected.
#   Accuracy holds at ~96%+ because shape information is unchanged.
#
#   THE GAP IS YOUR PROOF.
#
# HOW THE ADVERSARIAL MAPPING WORKS:
#   For each image with true label c, we sample uniformly from
#   {CLASS_COLORS[k] : k ≠ c}.  This is a per-sample guarantee: the
#   assigned color is never the class's own shortcut color.
#
#   We do this per-sample rather than with a single global swap because a
#   fixed swap (0→1, 1→2, …, 9→0) still allows a model to score 100% by
#   learning the remapped color→class association.  Per-sample random
#   assignment removes every deterministic escape hatch.
#
# PLACEMENT:  save this file to  data/adversarial_dataset.py
#
# VRAM CACHING:
#   Uses its own class-level cache (separate from ColoredMNIST) so it never
#   overwrites the standard test set already sitting in VRAM.
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np

from data.colored_mnist import CLASS_COLORS
import config


class AdversarialColoredMNIST(Dataset):
    """
    MNIST test set where every image's background color is guaranteed to
    belong to a DIFFERENT class than the image's true label.

    Stress test interpretation:
        Shortcut-reliant model  →  ~10% accuracy (predicts by color = wrong)
        Shape-reliant model     →  ~96%+ accuracy (color is irrelevant)

    The accuracy gap between these two numbers is the central empirical
    result of this project.
    """

    # Separate VRAM cache — never collides with ColoredMNIST._shared_vram_cache
    _adv_vram_cache = None
    _adv_colors     = None

    def __init__(self, root=None, download=True, seed=None):
        """
        Args:
            root     : str  — path to MNIST data (default: config.DATA_DIR)
            download : bool — download MNIST if not present
            seed     : int  — for reproducibility (default: config.SEED + 1,
                               intentionally differs from the training seed)
        """
        root = root or config.DATA_DIR
        seed = seed if seed is not None else (config.SEED + 1)

        self.mnist  = datasets.MNIST(
            root=root, train=False, download=download, transform=None
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed   = seed

        if AdversarialColoredMNIST._adv_vram_cache is not None:
            # Already computed — share the pointer, no re-allocation
            self.color_assignments = AdversarialColoredMNIST._adv_colors
            self.cached_images     = AdversarialColoredMNIST._adv_vram_cache
        else:
            self.color_assignments = self._assign_adversarial_colors()
            AdversarialColoredMNIST._adv_colors = self.color_assignments

            self.cached_images = self._precompute_images()
            AdversarialColoredMNIST._adv_vram_cache = self.cached_images

    # -------------------------------------------------------------------------
    # Color assignment
    # -------------------------------------------------------------------------

    def _assign_adversarial_colors(self):
        """
        For each image sample a color uniformly from all colors except
        the image's own class color.

        Returns:
            color_assignments : list[(R,G,B)] — one tuple per image
        """
        rng    = np.random.RandomState(self.seed)
        labels = self.mnist.targets.numpy()

        color_assignments = []
        for label in labels:
            # 9 eligible colors: all classes except this label's own
            eligible = [CLASS_COLORS[k] for k in range(10) if k != int(label)]
            chosen   = eligible[rng.randint(0, len(eligible))]
            color_assignments.append(chosen)

        # Hard verification — surface the bug if the guarantee ever breaks
        violations = sum(
            1 for label, color in zip(labels, color_assignments)
            if color == CLASS_COLORS[int(label)]
        )
        if violations > 0:
            raise RuntimeError(
                f"[AdversarialDataset] BUG: {violations} images received "
                f"their correct shortcut color.  Adversarial guarantee broken."
            )

        print(
            f"[AdversarialDataset] Color assignment verified — "
            f"0 / {len(labels):,} images have their own shortcut color."
        )
        return color_assignments

    # -------------------------------------------------------------------------
    # VRAM precomputation  (identical pipeline to ColoredMNIST._precompute_images)
    # -------------------------------------------------------------------------

    def _precompute_images(self):
        print("[AdversarialDataset] Pre-computing adversarial images → VRAM ...")

        raw_data = self.mnist.data.float() / 255.0        # (N, 28, 28)
        N        = raw_data.shape[0]

        # Digit pixels are bright (stroke), background pixels are dark
        mask = (raw_data > 0.1).unsqueeze(1)              # (N, 1, 28, 28)

        colors_tensor = torch.tensor(
            self.color_assignments, dtype=torch.float32
        ) / 255.0                                          # (N, 3)
        colors_tensor = colors_tensor.view(N, 3, 1, 1)

        background   = colors_tensor.expand(N, 3, 28, 28) # adversarial color
        digits       = torch.ones(N, 3, 28, 28, dtype=torch.float32)  # white stroke

        colored_data = torch.where(mask, digits, background)
        return colored_data.to(self.device)

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img_tensor = self.cached_images[idx]
        label      = int(self.mnist.targets[idx])
        color      = self.color_assignments[idx]

        # Same normalization as ColoredMNIST: [0,1] → [-1,1]
        return (img_tensor - 0.5) / 0.5, label, color

    # -------------------------------------------------------------------------
    # Cache management
    # -------------------------------------------------------------------------

    @classmethod
    def clear_cache(cls):
        """Force a rebuild on next instantiation (useful when changing seed)."""
        cls._adv_vram_cache = None
        cls._adv_colors     = None


# =============================================================================
# Convenience factory
# =============================================================================

def get_adversarial_loader(batch_size=None, seed=None):
    """
    Returns a DataLoader over the adversarial evaluation split.

    Args:
        batch_size : int  (default: config.BATCH_SIZE)
        seed       : int  (default: config.SEED + 1)

    Returns:
        DataLoader with shuffle=False (deterministic evaluation order)
    """
    batch_size = batch_size or config.BATCH_SIZE

    dataset = AdversarialColoredMNIST(
        root=config.DATA_DIR, download=True, seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,   # data already lives in VRAM
    )

    print(f"[AdversarialLoader] {len(dataset):,} images | "
          f"guarantee: 0% correct shortcut colors")
    return loader
