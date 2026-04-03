
"""
utils/utils.py  —  Shared utilities for the NAS framework
"""
import os, json, time, logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Create a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{ts}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

# ─────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────
class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        s = self.elapsed()
        m, s = divmod(s, 60)
        return f"{int(m):02d}m {s:.1f}s"

# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    """Save a training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, device) -> dict:
    """Load a checkpoint from disk."""
    return torch.load(path, map_location=device)

# ─────────────────────────────────────────────
# RESULTS LOGGING
# ─────────────────────────────────────────────
def log_result(result: dict, path: str):
    """Append a result dict to a JSONL file (one JSON per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(result) + "
")

def load_results(path: str) -> list:
    """Load all results from a JSONL file."""
    results = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                results.append(json.loads(line.strip()))
    return results

# ─────────────────────────────────────────────
# PARAMETER COUNTER
# ─────────────────────────────────────────────
def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(n: int) -> str:
    """Format parameter count for display."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

# ─────────────────────────────────────────────
# CONVERGENCE PLOTTER
# ─────────────────────────────────────────────
def plot_convergence(history: list, title: str = "GA Convergence",
                     save_path: str = None):
    """Plot best/avg fitness over generations."""
    gens  = [h["gen"]  for h in history]
    best  = [h["best"] for h in history]
    avg   = [h["avg"]  for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gens, best, "b-o", linewidth=2, markersize=4, label="Best")
    ax.plot(gens, avg,  "r--s", linewidth=1.5, markersize=3, label="Average")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────
# PARETO FRONT PLOTTER
# ─────────────────────────────────────────────
def plot_pareto_front(results: list, save_path: str = None):
    """2D Pareto plot: accuracy vs num_params."""
    acc    = [r["accuracy"]   for r in results]
    params = [r["num_params"] for r in results]
    pareto = [r.get("is_pareto", False) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    for a, p, is_p in zip(acc, params, pareto):
        color  = "red"   if is_p else "steelblue"
        marker = "*"     if is_p else "o"
        size   = 150     if is_p else 30
        alpha  = 1.0     if is_p else 0.5
        ax.scatter(p / 1e6, a * 100, c=color, marker=marker,
                   s=size, alpha=alpha)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Pareto Front: Accuracy vs Model Size")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",   markersize=12, label="Pareto optimal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=8, label="Dominated"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# ─────────────────────────────────────────────
# SURROGATE DIAGNOSTICS
# ─────────────────────────────────────────────
def plot_surrogate_quality(y_true, y_pred, save_path: str = None):
    """Predicted vs actual accuracy scatter for surrogate."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, color="purple", s=40)
    mn, mx = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"Surrogate Quality  (R² = {r2:.3f})")
    ax.set_xlabel("True Accuracy")
    ax.set_ylabel("Predicted Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
