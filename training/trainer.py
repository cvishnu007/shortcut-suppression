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
from explainer.shortcut_detector import ShortcutRegionDiscovery, compute_shortcut_loss_discovered
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

def evaluate(model, data_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    model.train()
    return correct / total

def train_with_suppression(model, train_loader, test_loader, device,
                           use_discovery=True):
    print("\n" + "="*60)
    print("  SUPPRESSION TRAINING (shortcut penalty active)")
    print(f"  Lambda = {config.LAMBDA_SHORTCUT}")
    print(f"  Mode   = {'SAC Discovery' if use_discovery else 'Hardcoded Mask'}")
    print("="*60)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn   = ShortcutSuppressionLoss(lambda_shortcut=config.LAMBDA_SHORTCUT)
    logger    = TrainingLogger()

    # ── Shortcut Region Discovery — initialised before training ──────────────
    discovery = ShortcutRegionDiscovery(
        image_size=config.IMAGE_SIZE,
        bottom_percent=0.4          # Flag top 40% most consistently attended pixels
    )
    discovered_mask = None       # Will be set after warmup ends

    history = {
        "train_loss": [], "task_loss": [], "shortcut_loss": [],
        "train_acc": [], "test_acc": []
    }

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss, epoch_task, epoch_sc = 0.0, 0.0, 0.0
        correct, total = 0, 0

        # ── Compute discovered mask at the END of warmup ──────────────────────
        if epoch == config.WARMUP_EPOCHS + 1 and use_discovery:
            discovered_mask = discovery.compute_mask(device)

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # ── Step 1: Forward pass ──────────────────────────────────────────
            logits = model(images)

            # ── Step 2: Differentiable attributions ───────────────────────────
            images_for_attr = images.detach().clone().requires_grad_(True)
            logits_for_attr = model(images_for_attr)

            target_scores = logits_for_attr.gather(
                1, labels.view(-1, 1)
            ).squeeze(1).sum()

            grads = torch.autograd.grad(
                outputs=target_scores,
                inputs=images_for_attr,
                create_graph=True
            )[0]

            attributions = grads * images_for_attr   # (batch, 3, H, W)

            # ── Step 3: Loss computation ──────────────────────────────────────
            if epoch <= config.WARMUP_EPOCHS:
                # Warmup phase — task loss only, accumulate attributions
                total_loss = loss_fn.task_loss_fn(logits, labels)
                task_val   = total_loss.item()
                sc_val     = 0.0

                # Accumulate detached attributions for mask discovery
                if use_discovery:
                    discovery.accumulate(attributions.detach(),labels)

            else:
                # Suppression phase
                if use_discovery and discovered_mask is not None:
                    # Use data-driven discovered mask
                    batch_mask = discovered_mask.expand(
                        images.size(0), 3,
                        config.IMAGE_SIZE, config.IMAGE_SIZE
                    )
                    task_loss  = loss_fn.task_loss_fn(logits, labels)
                    sc_loss    = compute_shortcut_loss_discovered(
                        attributions, batch_mask
                    )
                    total_loss = task_loss + config.LAMBDA_SHORTCUT * sc_loss
                    task_val   = task_loss.item()
                    sc_val     = sc_loss.item()
                else:
                    # Fallback to hardcoded mask
                    total_loss, task_val, sc_val = loss_fn(
                        logits, labels, attributions, images
                    )

            # ── Step 4: Backprop ──────────────────────────────────────────────
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_task += task_val
            epoch_sc   += sc_val

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total   += labels.size(0)

        # ── Epoch Summary ─────────────────────────────────────────────────────
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