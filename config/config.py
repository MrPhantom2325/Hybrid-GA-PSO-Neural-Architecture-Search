
"""
config.py  —  Master configuration for Hybrid GA-PSO NAS
All modules import from here. Change values here only.
"""
import torch

# ─────────────────────────────────────────────
# GENERAL
# ─────────────────────────────────────────────
SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = "/content/nas_project"
RESULTS_DIR  = f"{PROJECT_ROOT}/results"

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
DATASET         = "MNIST"       # Options: "MNIST" | "CIFAR10"
DATA_DIR        = f"{PROJECT_ROOT}/data"
NUM_CLASSES     = 10
INPUT_CHANNELS  = 1             # 1 for MNIST, 3 for CIFAR-10
IMAGE_SIZE      = 28            # 28 for MNIST, 32 for CIFAR-10
VAL_SPLIT       = 0.1           # 10% of training set for validation

# ─────────────────────────────────────────────
# CHROMOSOME / SEARCH SPACE
# ─────────────────────────────────────────────
# Each architecture = fixed-length integer vector
# Index  Name               Range          Meaning
# 0      num_conv_layers    [1, 4]         How many conv blocks
# 1      filters_1          [0, 5]         idx → [16,32,64,128,256,512]
# 2      filters_2          [0, 5]         idx → [16,32,64,128,256,512]
# 3      filters_3          [0, 5]         idx → [16,32,64,128,256,512]
# 4      filters_4          [0, 5]         idx → [16,32,64,128,256,512]
# 5      kernel_size        [0, 2]         idx → [3, 5, 7]
# 6      pool_type          [0, 2]         0=max, 1=avg, 2=none
# 7      num_dense          [1, 3]         Dense layers after conv
# 8      dense_units        [0, 4]         idx → [64,128,256,512,1024]
# 9      activation         [0, 3]         0=relu, 1=elu, 2=leaky, 3=selu
# 10     use_batchnorm      [0, 1]         0=no, 1=yes
# 11     use_skip           [0, 1]         0=no residual, 1=residual

CHROM_LENGTH = 12

SEARCH_SPACE = {
    "num_conv_layers" : (1, 4),
    "filters"         : [16, 32, 64, 128, 256, 512],
    "kernel_sizes"    : [3, 5, 7],
    "pool_types"      : ["max", "avg", "none"],
    "num_dense"       : (1, 3),
    "dense_units"     : [64, 128, 256, 512, 1024],
    "activations"     : ["relu", "elu", "leaky_relu", "selu"],
    "use_batchnorm"   : [False, True],
    "use_skip"        : [False, True],
}

# ─────────────────────────────────────────────
# GENETIC ALGORITHM
# ─────────────────────────────────────────────
GA_CONFIG = {
    "population_size"   : 30,    # Number of architectures per generation
    "num_generations"   : 20,    # Total GA generations to run
    "crossover_prob"    : 0.8,   # Probability of crossover between parents
    "mutation_prob"     : 0.2,   # Probability of mutating a chromosome
    "gene_mutation_prob": 0.1,   # Per-gene mutation probability
    "tournament_size"   : 3,     # k for tournament selection
    "elitism_k"         : 2,     # Top-k kept unchanged each generation
}

# ─────────────────────────────────────────────
# PARTICLE SWARM OPTIMIZATION
# ─────────────────────────────────────────────
PSO_CONFIG = {
    "num_particles"  : 20,     # Swarm size
    "num_iterations" : 30,     # PSO iterations per architecture
    "w"              : 0.7,    # Inertia weight
    "c1"             : 1.5,    # Cognitive coefficient (personal best)
    "c2"             : 1.5,    # Social coefficient (global best)
    "w_decay"        : 0.99,   # Inertia weight decay per iteration
    # Hyperparameter bounds [min, max]
    "bounds": {
        "log_lr"    : (-5.0, -1.0),   # log10(learning_rate): 1e-5 to 0.1
        "dropout"   : (0.0,  0.6),    # Dropout rate
        "batch_size": (4,    7),      # 2^x: 16 to 128
    }
}

# ─────────────────────────────────────────────
# SURROGATE MODEL
# ─────────────────────────────────────────────
SURROGATE_CONFIG = {
    "model_type"       : "xgboost",  # Options: "xgboost" | "random_forest" | "gbm"
    "warmup_samples"   : 10,         # Fully train this many archs before surrogate kicks in
    "top_k_ratio"      : 0.4,        # Evaluate top 40% predicted by surrogate
    "retrain_interval" : 5,          # Retrain surrogate every N generations
    "min_r2_threshold" : 0.6,        # Min R² to trust the surrogate
    "xgb_params": {
        "n_estimators"    : 100,
        "max_depth"       : 4,
        "learning_rate"   : 0.1,
        "subsample"       : 0.8,
        "random_state"    : 42,
    },
    "rf_params": {
        "n_estimators"    : 100,
        "max_depth"       : 6,
        "random_state"    : 42,
    }
}

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
TRAINING_CONFIG = {
    "full_epochs"    : 20,     # Epochs for full evaluation
    "proxy_epochs"   : 5,      # Epochs for surrogate warmup / PSO fitness
    "optimizer"      : "adam", # adam | sgd
    "weight_decay"   : 1e-4,
    "scheduler"      : "cosine",  # cosine | step | none
    "early_stop_patience": 5,
}

# ─────────────────────────────────────────────
# MULTI-OBJECTIVE
# ─────────────────────────────────────────────
MOOBJ_CONFIG = {
    "objectives"  : ["accuracy", "num_params", "train_time"],
    "weights"     : [1.0, -1.0, -1.0],  # +maximize accuracy, -minimize rest
    "param_budget": 5_000_000,          # Max params allowed (5M)
}
