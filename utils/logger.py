# =============================================================================
# utils/logger.py — Training Logger
# =============================================================================
# Prints clean, formatted training progress to the console.
# =============================================================================


class TrainingLogger:
    """Handles all training progress printing."""

    def log_epoch(self, epoch, total_epochs, loss, train_acc, test_acc):
        """Logs a single epoch for baseline training."""
        print(
            f"Epoch [{epoch:3d}/{total_epochs}] "
            f"Loss: {loss:.4f}  "
            f"Train Acc: {train_acc:.2%}  "
            f"Test Acc: {test_acc:.2%}"
        )

    def log_epoch_suppression(self, epoch, total_epochs, total_loss,
                               task_loss, sc_loss, train_acc, test_acc):
        """Logs a single epoch for suppression training."""
        print(
            f"Epoch [{epoch:3d}/{total_epochs}] "
            f"Total: {total_loss:.4f}  "
            f"Task: {task_loss:.4f}  "
            f"Shortcut: {sc_loss:.4f}  "
            f"Train: {train_acc:.2%}  "
            f"Test: {test_acc:.2%}"
        )
