# =============================================================================
# config.py — Central configuration file
# =============================================================================
# All hyperparameters and settings live here.
# This means you NEVER need to dig through code to change a setting.
# Just change it here and re-run.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/raw"          # Where MNIST will be downloaded to
BIAS_RATIO = 1.0 # How "biased" the training data is.
                                 # 0.95 = 95% of training samples have the
                                 # shortcut color. Model will definitely cheat.
                                 # Lower this later to test edge cases.

NUM_CLASSES = 10                 # MNIST has digits 0–9 = 10 classes
IMAGE_SIZE = 28                  # MNIST images are 28x28 pixels
IN_CHANNELS = 3                  # RGB (we color the grayscale MNIST images)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "SimpleCNN"         # We use a simple CNN — easy to explain

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 20
WARMUP_EPOCHS = 10                    # How many full passes over the dataset
BATCH_SIZE = 512              # How many images per gradient update step
LEARNING_RATE = 1e-3             # How fast the model updates its weights
SEED = 42                        # Random seed — keeps results reproducible

# ─────────────────────────────────────────────────────────────────────────────
# SHORTCUT SUPPRESSION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_SHORTCUT = 1.0            # Weight of the shortcut penalty in total loss
                                 # Total loss = Task loss + λ * Shortcut loss
                                 # Higher λ = stronger punishment for cheating

SHORTCUT_THRESHOLD = 0.3         # If shortcut score > this, we flag it.
                                 # Score = fraction of attribution in spurious zone
                                 # 0.3 means "30% or more attribution on background
                                 #            = shortcut detected"

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
EVAL_EVERY_N_EPOCHS = 5         # Run full evaluation every N epochs
SAVE_FIGURES = True              # Save attribution visualizations to disk
RESULTS_DIR = "./results/figures"

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "./checkpoints"
SAVE_BEST_MODEL = True           # Save only the best model by validation accuracy
