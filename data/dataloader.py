# =============================================================================
# data/dataloader.py — DataLoader Factory
# =============================================================================
# WHAT THIS FILE DOES:
#   Creates PyTorch DataLoaders (train/test) from our ColoredMNIST dataset.
#   DataLoaders handle batching, shuffling, and parallel data loading.
# =============================================================================

from torch.utils.data import DataLoader
from data.colored_mnist import ColoredMNIST
import config


def get_dataloaders(bias=None):
    """
    Creates and returns training and test DataLoaders.

    Args:
        bias : float — shortcut bias for training (default: from config.py)

    Returns:
        train_loader : DataLoader for biased training set
        test_loader  : DataLoader for unbiased test set

    USAGE:
        train_loader, test_loader = get_dataloaders()
        for images, labels, colors in train_loader:
            ...  # your training code here
    """

    # Use config value if no bias is manually specified
    if bias is None:
        bias = config.BIAS_RATIO

    # Create training dataset (biased — model will learn shortcuts here)
    train_dataset = ColoredMNIST(
        root=config.DATA_DIR,
        train=True,
        bias=bias,
        download=True
    )

    # Create test dataset (unbiased — no shortcuts, tests true generalization)
    test_dataset = ColoredMNIST(
        root=config.DATA_DIR,
        train=False,
        bias=0.0,       # Always 0 at test time
        download=True
    )

    # Wrap in DataLoaders
    # shuffle=True for training → prevents model from memorizing order
    # shuffle=False for testing → consistent evaluation
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # Parallel data loading workers
        pin_memory=True         # Faster GPU transfers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"[Data] Training samples : {len(train_dataset):,}")
    print(f"[Data] Test samples     : {len(test_dataset):,}")
    print(f"[Data] Bias ratio       : {bias:.0%} shortcut colors in training")

    return train_loader, test_loader
