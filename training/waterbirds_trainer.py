# =============================================================================
# training/waterbirds_trainer.py — Waterbirds Training Loops
# =============================================================================
#
# TWO TRAINING MODES:
#   train_waterbirds_baseline()   — standard ERM fine-tuning on ResNet-18
#   train_waterbirds_suppression() — ERM + Grad-CAM shortcut penalty
#
# KEY DIFFERENCE FROM MNIST TRAINING:
#   1. Grad-CAM instead of Integrated Gradients for attribution.
#      Grad-CAM fires during the normal backward pass — no extra forward
#      passes needed.  The hook on layer4 captures activations and
#      gradients automatically.
#
#   2. Two-pass structure per batch for suppression:
#      Pass 1: forward + backward for task loss (populates Grad-CAM hooks)
#      Pass 2: compute Grad-CAM shortcut loss + backward
#      This keeps gradients clean — task and shortcut losses are computed
#      from the same forward pass but backpropagated in controlled order.
#
#   3. Lower learning rate (1e-4 vs 1e-3) because ResNet-18 weights are
#      pretrained — large steps would destroy the ImageNet features we
#      rely on before the model has learned to use them for birds.
#
#   4. Worst-group accuracy tracked every epoch in addition to overall
#      accuracy.  This is the primary metric for Waterbirds — a model
#      can achieve 90%+ overall accuracy while failing catastrophically
#      on minority groups.
#
# PLACEMENT:  save to  training/waterbirds_trainer.py
# =============================================================================

import torch
import torch.optim as optim
import torch.nn.functional as F

from models.resnet import compute_gradcam_shortcut_loss
import config


# =============================================================================
# Internal helpers
# =============================================================================

def _evaluate_waterbirds(model, loader, device):
    """
    Returns (overall_acc, worst_group_acc, per_group_acc).

    worst_group_acc is the primary Waterbirds metric.
    Groups 1 and 2 are the minority groups that collapse under shortcut learning:
        Group 1: landbird on water background
        Group 2: waterbird on land background
    """
    model.eval()
    group_correct = {g: 0 for g in range(4)}
    group_total   = {g: 0 for g in range(4)}

    with torch.no_grad():
        for images, labels, groups in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1)

            for pred, label, group in zip(preds, labels, groups):
                g = int(group.item())
                group_total[g]   += 1
                if pred.item() == label.item():
                    group_correct[g] += 1

    per_group = {
        g: group_correct[g] / group_total[g]
        if group_total[g] > 0 else 0.0
        for g in range(4)
    }
    overall      = sum(group_correct.values()) / sum(group_total.values())
    worst_group  = min(per_group.values())

    model.train()
    return overall, worst_group, per_group


# =============================================================================
# Baseline training
# =============================================================================

def train_waterbirds_baseline(model, train_loader, val_loader, device):
    """
    Standard ERM fine-tuning on Waterbirds.

    The model will learn the background shortcut because landbirds appear
    on land 95% of the time and waterbirds appear on water 95% of the time.
    Overall accuracy will be high; worst-group accuracy will be low.

    Args:
        model        : ResNet18Waterbirds
        train_loader : DataLoader from get_waterbirds_loaders()
        val_loader   : validation DataLoader
        device       : torch.device

    Returns:
        history : dict with train_loss, train_acc, val_acc,
                  val_worst_group, val_per_group lists
    """
    epochs = getattr(config, 'WATERBIRDS_EPOCHS', 30)
    lr     = getattr(config, 'WATERBIRDS_LR', 1e-4)

    print("\n" + "="*64)
    print("  WATERBIRDS BASELINE (ERM fine-tuning, no suppression)")
    print("="*64)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_worst_group': [], 'val_per_group': []
    }
    
    best_wg = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()

        avg_loss  = epoch_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_wg, val_pg = _evaluate_waterbirds(model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_worst_group'].append(val_wg)
        history['val_per_group'].append(val_pg)

        print(
            f"Epoch [{epoch:3d}/{epochs}]  "
            f"Loss: {avg_loss:.4f}  "
            f"Train: {train_acc:.2%}  "
            f"Val: {val_acc:.2%}  "
            f"Worst-group: {val_wg:.2%}"
        )
        
        # Save best by worst-group accuracy, not final epoch
        if val_wg > best_wg:
            best_wg = val_wg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [✓] New best worst-group: {best_wg:.2%} at epoch {epoch}")

    # Restore best weights before returning
    model.load_state_dict(best_state)
    print(f"[Baseline] Restored best checkpoint (worst-group={best_wg:.2%})")
    return history


def train_waterbirds_suppression(model, train_loader, val_loader, device):
    """
    ERM + Grad-CAM shortcut penalty fine-tuning on Waterbirds.

    Training has two phases:
        Warmup (epochs 1 to WATERBIRDS_WARMUP):
            Task loss only — model builds basic bird representations
            before we penalize it for looking at backgrounds.
        Suppression (remaining epochs):
            Task loss + λ × Grad-CAM border penalty.

    The Grad-CAM shortcut loss is computed using activations and gradients
    already captured by the hooks registered in ResNet18Waterbirds.__init__.
    No extra forward passes are needed.

    Args:
        model        : ResNet18Waterbirds (hooks registered)
        train_loader : DataLoader from get_waterbirds_loaders()
        val_loader   : validation DataLoader
        device       : torch.device

    Returns:
        history : dict with train_loss, task_loss, shortcut_loss,
                  train_acc, val_acc, val_worst_group, val_per_group lists
    """
    epochs       = getattr(config, 'WATERBIRDS_EPOCHS', 30)
    lr           = getattr(config, 'WATERBIRDS_LR', 1e-4)
    warmup       = getattr(config, 'WATERBIRDS_WARMUP', 5)
    lambda_sc    = getattr(config, 'WATERBIRDS_LAMBDA', 1.0)
    border_frac  = getattr(config, 'WATERBIRDS_BORDER_FRACTION', 0.3)

    print("\n" + "="*64)
    print("  WATERBIRDS SUPPRESSION (Grad-CAM border penalty)")
    print(f"  Warmup epochs : {warmup}")
    print(f"  Lambda        : {lambda_sc}")
    print(f"  Border frac   : {border_frac}")
    print("="*64)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'task_loss': [], 'shortcut_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_worst_group': [], 'val_per_group': []
    }
    
    best_wg = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_task, epoch_sc = 0.0, 0.0, 0.0
        correct, total = 0, 0
        in_suppression = (epoch > warmup)

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # ── Forward pass ──────────────────────────────────────────────
            # retain_graph=True during suppression so we can backward twice:
            # once for task loss (populates Grad-CAM hooks), once for shortcut.
            logits    = model(images)
            task_loss = F.cross_entropy(logits, labels)

            if in_suppression:
                # First backward — populates layer4 gradients for Grad-CAM
                # retain_graph keeps computation graph alive for second backward
                task_loss.backward(retain_graph=True)

                # Grad-CAM shortcut loss — uses hooks populated above
                sc_loss    = compute_gradcam_shortcut_loss(
                    model, images, labels, border_fraction=border_frac
                )
                total_loss = lambda_sc * sc_loss
                total_loss.backward()

                task_val = task_loss.item()
                sc_val   = sc_loss.item()
                loss_val = task_loss.item() + lambda_sc * sc_val

            else:
                # Warmup — task loss only
                task_loss.backward()
                task_val = task_loss.item()
                sc_val   = 0.0
                loss_val = task_val

            optimizer.step()

            epoch_loss += loss_val
            epoch_task += task_val
            epoch_sc   += sc_val
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()

        avg_loss  = epoch_loss / len(train_loader)
        avg_task  = epoch_task / len(train_loader)
        avg_sc    = epoch_sc   / len(train_loader)
        train_acc = correct / total
        val_acc, val_wg, val_pg = _evaluate_waterbirds(model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['task_loss'].append(avg_task)
        history['shortcut_loss'].append(avg_sc)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_worst_group'].append(val_wg)
        history['val_per_group'].append(val_pg)

        phase = "SUPPRESS" if in_suppression else "warmup "
        print(
            f"Epoch [{epoch:3d}/{epochs}] [{phase}]  "
            f"Task: {avg_task:.4f}  SC: {avg_sc:.4f}  "
            f"Train: {train_acc:.2%}  Val: {val_acc:.2%}  "
            f"Worst-group: {val_wg:.2%}"
        )
        
        # Save best by worst-group accuracy, not final epoch
        if val_wg > best_wg:
            best_wg = val_wg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [✓] New best worst-group: {best_wg:.2%} at epoch {epoch}")
    import copy
    final_state = copy.deepcopy(model.state_dict())
    # Restore best weights before returning
    model.load_state_dict(best_state)
    print(f"[Suppression] Restored best checkpoint (worst-group={best_wg:.2%})")
    return history,final_state,best_state