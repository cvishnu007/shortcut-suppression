# =============================================================================
# training/jtt_trainer.py — Just Train Twice (JTT)
# =============================================================================
#
# PAPER: "Just Train Twice: Improving Group Robustness without Training Group
#         Information" — Liu et al., ICML 2021
#
# WHAT JTT DOES (plain English):
#   Standard ERM training on biased data causes the model to memorise
#   shortcuts because shortcut examples are easy — they produce low loss
#   quickly.  JTT exploits this: the examples a biased model gets WRONG
#   after a short training run are disproportionately the hard examples
#   that cannot be solved by the shortcut alone.  Upsampling those failures
#   forces the second model to learn the true signal.
#
#   Step 1 — Train an ERM "identifier" model for T_id epochs (short).
#             Collect the set of training examples it predicts incorrectly.
#             Call this the "failure set" F.
#
#   Step 2 — Build a new weighted dataset: every sample in F appears
#             lambda_up times, every other sample appears once.
#             Train a fresh model on this upsampled dataset with standard CE.
#
#   No new loss functions.  No attribution maps.  No group labels.
#   The only hyperparameters are T_id and lambda_up.
#
# WHY THIS IS THE RIGHT COMPARISON FOR YOUR WORK:
#   JTT is the current standard "no group labels" debiasing baseline.
#   Every workshop/conference reviewer familiar with shortcut learning
#   will ask "did you compare to JTT?"
#
#   Your method's advantage over JTT is interpretability:
#   JTT produces a robust model but gives no explanation of WHAT the
#   shortcut was or WHERE the model was looking.  Your suppression method
#   produces the attribution map as a byproduct of training, giving an
#   explicit audit trail of the unlearned shortcut.
#
#   If your shortcut score is also lower than JTT's, that is an additional
#   quantitative advantage to report.
#
# PLACEMENT:  save this file to  training/jtt_trainer.py
#
# USAGE (in main.py):
#   from training.jtt_trainer import train_jtt
#   jtt_model, jtt_history = train_jtt(
#       train_loader, test_loader, device
#   )
#
# HYPERPARAMETERS (set in config.py — see bottom of this file for guidance):
#   JTT_ID_EPOCHS  : epochs for the identifier run  (default: 1)
#   JTT_LAMBDA_UP  : upsampling multiplier for failures (default: 50)
# =============================================================================

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from training.losses import TaskLoss
from utils.logger import TrainingLogger
from models.cnn import get_model
import config


# =============================================================================
# Step 1 — Identifier training
# =============================================================================

def _train_identifier(train_loader, test_loader, device):
    """
    Trains a short ERM model to identify which training samples are hard
    (cannot be solved by the shortcut alone).

    Args:
        train_loader : standard biased training DataLoader
        test_loader  : unbiased test DataLoader
        device       : torch.device

    Returns:
        identifier   : trained model (used only for inference, then discarded)
        failure_indices : list[int] — training indices the identifier got wrong
    """
    t_id = getattr(config, 'JTT_ID_EPOCHS', 1)

    print(f"\n[JTT] Step 1 — Identifier training ({t_id} epoch(s)) ...")

    identifier = get_model().to(device)
    optimizer  = optim.Adam(identifier.parameters(), lr=config.LEARNING_RATE)
    loss_fn    = TaskLoss()

    # Short training run — intentionally brief so the model overfits shortcuts
    for epoch in range(1, t_id + 1):
        identifier.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = identifier(images)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc = correct / total
        test_acc  = _evaluate(identifier, test_loader, device)
        print(
            f"[JTT]   Identifier epoch [{epoch}/{t_id}]  "
            f"Train: {train_acc:.2%}  Test: {test_acc:.2%}"
        )

    # ── Collect failure indices ────────────────────────────────────────────
    # We need sample-level indices, so we iterate with a sequential loader
    # (shuffle=False, batch_size=1 would work but is slow — use larger batches)
    print("[JTT] Collecting failure set ...")

    identifier.eval()
    failure_indices = []
    sample_idx      = 0

    # Build a non-shuffled loader over the same dataset to get stable indices
    sequential_loader = DataLoader(
        train_loader.dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,      # CRITICAL: must be False to get stable indices
        num_workers=0,
        pin_memory=False,
    )

    with torch.no_grad():
        for images, labels, _ in sequential_loader:
            images, labels = images.to(device), labels.to(device)
            preds = identifier(images).argmax(dim=1)

            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred.item() != label.item():
                    failure_indices.append(sample_idx + i)

            sample_idx += images.size(0)

    pct = len(failure_indices) / len(train_loader.dataset) * 100
    print(
        f"[JTT] Failure set: {len(failure_indices):,} samples "
        f"({pct:.1f}% of training data)"
    )

    # Free the identifier — we only needed it to find the failure set
    del identifier
    torch.cuda.empty_cache()

    return failure_indices


# =============================================================================
# Step 2 — Upsampled retraining
# =============================================================================

def _build_upsampled_loader(train_loader, failure_indices):
    """
    Builds a DataLoader where every failure sample is upsampled by lambda_up
    and every other sample appears once.

    Uses WeightedRandomSampler so the total number of batches per epoch
    stays the same as the original loader (same len(train_loader)).
    This keeps training time comparable to baseline.

    Args:
        train_loader    : original biased DataLoader
        failure_indices : list[int] from _train_identifier()

    Returns:
        upsampled_loader : DataLoader with weighted sampling
    """
    lambda_up = getattr(config, 'JTT_LAMBDA_UP', 50)
    n         = len(train_loader.dataset)

    failure_set = set(failure_indices)

    # Weight: lambda_up for failures, 1 for all others
    weights = torch.ones(n, dtype=torch.float32)
    for idx in failure_set:
        weights[idx] = float(lambda_up)

    # num_samples = same total as original dataset so epoch length is unchanged
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n,
        replacement=True,
    )

    upsampled_loader = DataLoader(
        train_loader.dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )

    expected_failure_frac = (len(failure_indices) * lambda_up) / (
        len(failure_indices) * lambda_up + (n - len(failure_indices))
    )
    print(
        f"[JTT] Upsampled loader built  "
        f"(lambda_up={lambda_up}, "
        f"~{expected_failure_frac:.0%} of each batch from failure set)"
    )

    return upsampled_loader


# =============================================================================
# Main entry point
# =============================================================================

def train_jtt(train_loader, test_loader, device):
    """
    Full Just Train Twice pipeline.

    Args:
        train_loader : standard biased DataLoader (from get_dataloaders())
        test_loader  : unbiased test DataLoader
        device       : torch.device

    Returns:
        jtt_model   : the final retrained model
        history     : dict with 'train_loss', 'train_acc', 'test_acc' lists

    Hyperparameters (set in config.py):
        JTT_ID_EPOCHS = 1    # identifier run length — 1 epoch is standard
        JTT_LAMBDA_UP = 50   # upsampling weight for failures
    """
    print("\n" + "="*60)
    print("  JTT TRAINING (Just Train Twice)")
    print(f"  T_id      = {getattr(config, 'JTT_ID_EPOCHS', 1)} epoch(s)")
    print(f"  lambda_up = {getattr(config, 'JTT_LAMBDA_UP', 50)}")
    print("="*60)

    # ── Step 1: Identifier ─────────────────────────────────────────────────
    failure_indices = _train_identifier(train_loader, test_loader, device)

    # ── Step 2: Build upsampled loader ────────────────────────────────────
    upsampled_loader = _build_upsampled_loader(train_loader, failure_indices)

    # ── Step 3: Retrain from scratch on upsampled data ────────────────────
    print(f"\n[JTT] Step 2 — Retraining on upsampled dataset ...")

    jtt_model = get_model().to(device)
    optimizer = optim.Adam(jtt_model.parameters(), lr=config.LEARNING_RATE)
    loss_fn   = TaskLoss()
    logger    = TrainingLogger()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, config.EPOCHS + 1):
        jtt_model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for images, labels, _ in upsampled_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = jtt_model(images)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        avg_loss  = epoch_loss / len(upsampled_loader)
        train_acc = correct / total
        test_acc  = _evaluate(jtt_model, test_loader, device)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        logger.log_epoch(epoch, config.EPOCHS, avg_loss, train_acc, test_acc)

    return jtt_model, history


# =============================================================================
# Internal helpers
# =============================================================================

def _evaluate(model, data_loader, device):
    """Same as trainer.evaluate() — duplicated here to keep jtt_trainer.py
    self-contained and avoid a circular import with trainer.py."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels, _ in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits     = model(images)
            correct   += (logits.argmax(1) == labels).sum().item()
            total     += labels.size(0)

    model.train()
    return correct / total


# =============================================================================
# Config guidance
# =============================================================================
#
# Add these two lines to config.py:
#
#   JTT_ID_EPOCHS = 1    # Liu et al. use 1 epoch — model learns shortcuts fast
#   JTT_LAMBDA_UP = 50   # Liu et al. use 50 for Waterbirds; 20-50 works here
#
# Why T_id=1?
#   At bias=0.8, the model learns the color shortcut in epoch 1 (Train Acc
#   jumps to ~84% immediately — see your logs).  The failure set after 1
#   epoch is therefore the samples that resisted the shortcut, which are
#   exactly the shape-informative samples you want upsampled.
#   More identifier epochs → model memorises everything → failure set shrinks
#   → less signal for step 2.
#
# Why lambda_up=50?
#   At bias=0.8, ~20% of samples will be in the failure set (the wrong-color
#   20%).  With lambda_up=50, those samples represent ~92% of each upsampled
#   batch.  This aggressively forces the second model to solve them.
#   If JTT accuracy is too low, reduce to 20.  If shortcut score is still
#   high, increase to 100.
