# =============================================================================
# data/dataloader.py — DataLoader Factory
# =============================================================================
from torch.utils.data import DataLoader
from torchvision import transforms
from data.colored_mnist import ColoredMNIST
import config
import torch

def get_dataloaders(bias=None):
    if bias is None:
        bias = config.BIAS_RATIO

    train_dataset = ColoredMNIST(root=config.DATA_DIR, train=True, bias=bias, download=True)
    test_dataset = ColoredMNIST(root=config.DATA_DIR, train=False, bias=0.0, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False  # Disabled because data is already in VRAM!
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"[Data] Training samples : {len(train_dataset):,}")
    print(f"[Data] Test samples     : {len(test_dataset):,}")
    print(f"[Data] Bias ratio       : {bias:.0%} shortcut colors in training")

    return train_loader, test_loader

# =============================================================================
# THE MAGIC BATCH WRAPPER FOR COLOR JITTER
# =============================================================================
class BatchedJitterLoader:
    """
    Wraps a standard DataLoader to apply ColorJitter to the ENTIRE batch at once.
    This prevents the GPU from stalling on 60,000 individual operations.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.jitter = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, 
            saturation=0.5, hue=0.5
        )
        
    def __iter__(self):
        for images, labels, colors in self.dataloader:
            # 1. Images come out of the dataset as [-1.0, 1.0]. Revert to [0.0, 1.0] for Jitter.
            images = (images * 0.5) + 0.5
            
            # 2. Apply jitter to ALL 512 IMAGES SIMULTANEOUSLY on the GPU
            images = self.jitter(images)
            
            # 3. Normalize back to [-1.0, 1.0] for the CNN
            images = (images - 0.5) / 0.5
            
            yield images, labels, colors
            
    def __len__(self):
        return len(self.dataloader)

def get_dataloaders_jitter(bias=None):
    if bias is None:
        bias = config.BIAS_RATIO

    train_dataset = ColoredMNIST(root=config.DATA_DIR, train=True, bias=bias, download=True)
    test_dataset = ColoredMNIST(root=config.DATA_DIR, train=False, bias=0.0, download=True)

    # Standard loaders (getting data directly from VRAM)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False
    )
    
    # Wrap them in our lightning-fast batched jitter!
    return BatchedJitterLoader(train_loader), BatchedJitterLoader(test_loader)