# =============================================================================
# data/waterbirds.py — Waterbirds Dataset
# =============================================================================
#
# DATASET OVERVIEW:
#   Waterbirds is constructed by pasting CUB-200 bird images onto Places365
#   backgrounds.  The spurious correlation is background texture:
#     - Landbirds (y=0) appear on land backgrounds (place=0) 95% of training
#     - Waterbirds (y=1) appear on water backgrounds (place=1) 95% of training
#
#   A shortcut model learns "water background → waterbird" without learning
#   what a waterbird actually looks like.  It achieves high average accuracy
#   but collapses on the two minority groups:
#     - Group 1: landbird on water background  (model predicts waterbird)
#     - Group 2: waterbird on land background  (model predicts landbird)
#
# GROUPS (4 total):
#   group = y * 2 + place
#   0 = landbird  on land   (majority)
#   1 = landbird  on water  (minority — shortcut failure)
#   2 = waterbird on land   (minority — shortcut failure)
#   3 = waterbird on water  (majority)
#
# SPLITS (from metadata.csv):
#   0 = train, 1 = val, 2 = test
#
# PLACEMENT:  save to  data/waterbirds.py
# =============================================================================

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# =============================================================================
# Transforms
# =============================================================================

# Standard ImageNet normalization — ResNet-18 pretrained weights expect this
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(train=True):
    """
    Returns torchvision transforms for Waterbirds.

    Training: random crop + horizontal flip (standard for CUB-based datasets)
    Eval:     center crop only — deterministic

    ResNet-18 expects 224×224 ImageNet-normalized input.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])


# =============================================================================
# Dataset class
# =============================================================================

class WaterbirdsDataset(Dataset):
    """
    Waterbirds dataset loaded from metadata.csv.

    Returns:
        image  : Tensor (3, 224, 224) — ImageNet normalized
        label  : int — 0=landbird, 1=waterbird
        group  : int — 0-3, where 1 and 2 are the minority (shortcut-failure) groups
    """

    def __init__(self, root=None, split='train', transform=None):
        """
        Args:
            root      : str  — path to waterbird_complete95_forest2water2/ folder
                               (default: config.WATERBIRDS_DIR)
            split     : str  — 'train', 'val', or 'test'
            transform : torchvision transform (default: get_transforms)
        """
        root = root or config.WATERBIRDS_DIR

        split_map = {'train': 0, 'val': 1, 'test': 2}
        if split not in split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        split_id = split_map[split]

        # Load and filter metadata
        meta_path = os.path.join(root, 'metadata.csv')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.csv not found at {meta_path}.\n"
                f"Check that WATERBIRDS_DIR in config.py points to the folder "
                f"containing metadata.csv."
            )

        df = pd.read_csv(meta_path)
        df = df[df['split'] == split_id].reset_index(drop=True)

        self.root      = root
        self.df        = df
        self.transform = transform or get_transforms(train=(split == 'train'))

        # Derived group label: y * 2 + place
        # 0=(land,land) 1=(land,water) 2=(water,land) 3=(water,water)
        self.groups = (df['y'].values * 2 + df['place'].values).astype(int)
        self.labels = df['y'].values.astype(int)

        # Print split summary
        group_counts = {g: (self.groups == g).sum() for g in range(4)}
        print(f"[Waterbirds] {split:5s} split: {len(df):,} samples  "
              f"| groups: {group_counts}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = os.path.join(self.root, row['img_filename'])
        image     = Image.open(img_path).convert('RGB')
        label     = int(self.labels[idx])
        group     = int(self.groups[idx])

        if self.transform:
            image = self.transform(image)

        return image, label, group


# =============================================================================
# Convenience factory
# =============================================================================

def get_waterbirds_loaders(root=None):
    """
    Returns (train_loader, val_loader, test_loader) for Waterbirds.

    Batch size comes from config.WATERBIRDS_BATCH (default 64).
    num_workers=2 for disk-based loading (unlike MNIST which is VRAM-cached).

    Returns:
        train_loader : DataLoader — shuffled, augmented
        val_loader   : DataLoader — deterministic
        test_loader  : DataLoader — deterministic
    """
    root       = root or config.WATERBIRDS_DIR
    batch_size = getattr(config, 'WATERBIRDS_BATCH', 64)

    train_dataset = WaterbirdsDataset(root=root, split='train')
    val_dataset   = WaterbirdsDataset(root=root, split='val')
    test_dataset  = WaterbirdsDataset(root=root, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
