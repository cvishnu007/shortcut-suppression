# =============================================================================
# training/trainer.py — Training Loop
# =============================================================================
# WHAT THIS FILE DOES:
#   Contains the core training logic for both:
#     1. Baseline training    (standard CE loss, no suppression)
#     2. Suppression training (CE loss + shortcut penalty)
#
# HOW SUPPRESSION TRAINING WORKS (the key loop):
#   For each batch:
#     a) Forward pass: get predictions (logits)
#     b) Compute attribution maps (explanations) using Integrated Gradients
#     c) Compute shortcut score (how much attribution is on background)
#     d) Total loss = CE loss + λ × shortcut loss
#     e) Backward pass: update model weights
#
# IMPORTANT NOTE ON GRADIENTS AND ATTRIBUTIONS:
#   Normally, attribution computation is done in no_grad() mode.
#   For training suppression, attributions must have gradients so that
#   the shortcut loss can backpropagate all the way to the model weights.
# =============================================================================

import torch
import torch.optim as optim
from captum.attr import IntegratedGradients
from training.losses import TaskLoss, ShortcutSuppressionLoss
from utils.logger import TrainingLogger
from utils.helpers import save_checkpoint
import config


def train_baseline(model, train_loader, test_loader, device):
    """
    Standard training — no shortcut suppression.
    This is our "before" state. The model will learn to cheat.

    Args:
        model        : PyTorch model (SimpleCNN)
        train_loader : DataLoader for biased training data
        test_loader  : DataLoader for unbiased test data
        device       : torch.device

    Returns:
        history : dict with 'train_loss', 'train_acc', 'test_acc' lists
    """
    print("\n" + "="*60)
    print("  BASELINE TRAINING (no shortcut suppression)")
    print("="*60)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = TaskLoss()
    logger = TrainingLogger()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        model.train()    # Enable dropout and batch norm for training
        epoch_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          # Clear old gradients

            logits = model(images)         # Forward pass
            loss = loss_fn(logits, labels) # Compute CE loss

            loss.backward()                # Compute gradients
            optimizer.step()               # Update weights

            # Track metrics
            epoch_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Compute epoch stats
        avg_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        logger.log_epoch(epoch, config.EPOCHS, avg_loss, train_acc, test_acc)

    return history


def train_with_suppression(model, train_loader, test_loader, device):
    """
    Training WITH shortcut suppression — our core contribution.

    The model is penalized whenever its attribution maps focus on
    background (shortcut) regions instead of digit (real signal) regions.

    Args:
        model        : PyTorch model
        train_loader : DataLoader
        test_loader  : DataLoader
        device       : torch.device

    Returns:
        history : dict with training metrics
    """
    print("\n" + "="*60)
    print("  SUPPRESSION TRAINING (shortcut penalty active)")
    print(f"  Lambda = {config.LAMBDA_SHORTCUT}")
    print("="*60)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = ShortcutSuppressionLoss(lambda_shortcut=config.LAMBDA_SHORTCUT)
    ig = IntegratedGradients(model)      # Attribution method
    logger = TrainingLogger()

    history = {
        "train_loss": [], "task_loss": [], "shortcut_loss": [],
        "train_acc": [], "test_acc": []
    }

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss, epoch_task, epoch_sc = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # ── Step 1: Forward pass for predictions ──────────────────────────
            logits = model(images)

            # ── Step 2: Compute attribution maps ──────────────────────────────
            # We need attributions WITH gradients so the shortcut loss
            # can flow back through them to the model weights.
            #
            # Note: images need requires_grad=True for attribution computation.
            images_for_attr = images.detach().clone().requires_grad_(True)

            baseline = torch.zeros_like(images_for_attr)

            # Compute attributions (this does many mini forward passes internally)
            # n_steps=10 is lower than evaluation mode for speed during training
            attributions = ig.attribute(
                inputs=images_for_attr,
                baselines=baseline,
                target=labels,
                n_steps=10,              # Lower for training speed
                internal_batch_size=8
            )

            # ── Step 3: Compute combined loss ──────────────────────────────────
            total_loss, task_val, sc_val = loss_fn(
                logits, labels, attributions, images
            )

            # ── Step 4: Backprop and update ────────────────────────────────────
            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += total_loss.item()
            epoch_task += task_val
            epoch_sc   += sc_val

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # ── Epoch Summary ──────────────────────────────────────────────────────
        avg_loss  = epoch_loss / len(train_loader)
        avg_task  = epoch_task / len(train_loader)
        avg_sc    = epoch_sc   / len(train_loader)
        train_acc = correct / total
        test_acc  = evaluate(model, test_loader, device)

        history["train_loss"].append(avg_loss)
        history["task_loss"].append(avg_task)
        history["shortcut_loss"].append(avg_sc)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        logger.log_epoch_suppression(
            epoch, config.EPOCHS, avg_loss, avg_task, avg_sc, train_acc, test_acc
        )

    return history


def evaluate(model, data_loader, device):
    """
    Evaluates model accuracy on a dataset.

    Args:
        model       : PyTorch model
        data_loader : DataLoader
        device      : torch.device

    Returns:
        accuracy : float — fraction of correct predictions
    """
    model.eval()    # Disable dropout for evaluation
    correct, total = 0, 0

    with torch.no_grad():    # No gradient computation needed for eval
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.train()   # Switch back to train mode
    return correct / total
