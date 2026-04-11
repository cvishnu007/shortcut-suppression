# =============================================================================
# data/colored_mnist.py — Biased Colored MNIST Dataset (VRAM CACHE + CRASH FIX)
# =============================================================================
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
import config

CLASS_COLORS = {
    0: (255, 0,   0  ), 1: (0,   255, 0  ), 2: (0,   0,   255),
    3: (255, 255, 0  ), 4: (255, 0,   255), 5: (0,   255, 255),
    6: (255, 128, 0  ), 7: (128, 0,   255), 8: (0,   128, 255),
    9: (255, 0,   128),
}

class ColoredMNIST(Dataset):
    # ── THE FIX: Global VRAM dictionaries to prevent memory duplication ──
    _shared_vram_cache = {}
    _shared_colors = {}

    def __init__(self, root, train=True, bias=0.95, download=True):
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=None)
        self.train = train
        self.bias = bias if train else 0.0  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to_tensor = None

        # Generate a unique key for this dataset (e.g., "True_0.8")
        self.cache_key = f"{self.train}_{self.bias}"

        # If the data is already on the GPU, just point to it! Do not duplicate.
        if self.cache_key in ColoredMNIST._shared_vram_cache:
            self.color_assignments = ColoredMNIST._shared_colors[self.cache_key]
            self.cached_images = ColoredMNIST._shared_vram_cache[self.cache_key]
        else:
            # If this is the first time, compute and push to VRAM
            self.color_assignments = self._assign_colors()
            ColoredMNIST._shared_colors[self.cache_key] = self.color_assignments
            
            self.cached_images = self._precompute_images()
            ColoredMNIST._shared_vram_cache[self.cache_key] = self.cached_images

    def _assign_colors(self):
        color_assignments = []
        for idx in range(len(self.mnist)):
            label = int(self.mnist.targets[idx])
            if self.train:
                color = CLASS_COLORS[label] if np.random.random() < self.bias else CLASS_COLORS[np.random.randint(0, 10)]
            else:
                color = CLASS_COLORS[np.random.randint(0, 10)]
            color_assignments.append(color)
        return color_assignments

    def _precompute_images(self):
        print(f"[Data] Pre-computing and pushing {'training' if self.train else 'test'} dataset directly to VRAM...")
        raw_data = self.mnist.data.float() / 255.0
        N = raw_data.shape[0]
        mask = (raw_data > 0.1).unsqueeze(1)
        
        colors_tensor = torch.tensor(self.color_assignments, dtype=torch.float32) / 255.0
        colors_tensor = colors_tensor.view(N, 3, 1, 1)
        
        background = colors_tensor.expand(N, 3, 28, 28)
        digits = torch.ones(N, 3, 28, 28, dtype=torch.float32)
        
        colored_data = torch.where(mask, digits, background)
        return colored_data.to(self.device)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img_tensor = self.cached_images[idx]
        label = int(self.mnist.targets[idx])
        color = self.color_assignments[idx]

        if self.to_tensor is not None:
            jittered = self.to_tensor.transforms[0](img_tensor)
            return (jittered - 0.5) / 0.5, label, color

        return (img_tensor - 0.5) / 0.5, label, color