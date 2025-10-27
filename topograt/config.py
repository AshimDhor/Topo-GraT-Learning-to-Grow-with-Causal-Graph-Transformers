
DATA_DIR = "/path/to/your/wsi_datasets/"
OUTPUT_DIR = "/path/to/save/checkpoints_and_logs/"
DEVICE = "cuda"

# --- Training Parameters ---
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE_STAGE1 = 16
BATCH_SIZE_STAGE2 = 8

# --- Model Architecture Parameters ---
PATCH_SIZE = 512
SWIN_MODEL_NAME = 'swin_tiny_patch4_window7_224'
EMBED_DIM = 768 # Dimension from Swin-T

# --- Topo-GraT Specific Parameters ---
K_INITIAL_PATCHES = 50  # Number of initial patches to select
K_NEW_PATCHES = 10      # Number of new patches to add per iteration
NUM_ITERATIONS = 3      # Number of refinement iterations (T)

# --- Loss Function Weights ---
ALPHA = 0.1  # Weight for topology loss (L_topo)
BETA = 0.05  # Weight for instance selection loss (L_instance)
LAMBDA = 0.5 # Weight for uncertainty feedback in salience score
