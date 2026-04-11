# =============================================================================
# utils/helpers.py — Utility Functions
# =============================================================================

import torch
import numpy as np
import random
import os
import config


def set_seed(seed=None):
    """
    Sets random seeds for reproducibility.
    Without this, results change every run (hard to debug or compare).

    Args:
        seed : int — the seed value (default: from config.py)
    """
    seed = seed or config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # Ensures reproducibility on GPU
    print(f"[Seed] Set to {seed}")


def get_device():
    """
    Returns the best available device.
    Uses GPU (CUDA) if available, otherwise falls back to CPU.

    Returns:
        device : torch.device — either 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU detected: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU found, using CPU (training will be slower)")

    return device


def save_checkpoint(model, optimizer, epoch, accuracy, path):
    """
    Saves model state to disk so we can resume or reload later.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
    }
    
    # Only try to save optimizer state if an optimizer was actually passed
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved at epoch {epoch} (acc={accuracy:.4f}) → {path}")


def load_checkpoint(model, path, device):
    """
    Loads a saved model checkpoint from disk.

    Args:
        model  : PyTorch model (same architecture as when saved)
        path   : str — path to the .pth file
        device : torch.device

    Returns:
        model   : model with loaded weights
        epoch   : int — epoch the checkpoint was saved at
        accuracy: float — accuracy at checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch    = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    print(f"[Checkpoint] Loaded from {path} (epoch={epoch}, acc={accuracy:.4f})")
    return model, epoch, accuracy
