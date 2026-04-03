
"""
search_space/search_space_utils.py

Utilities for chromosome generation, manipulation, and feature extraction.
Used by GA (Phase 3), PSO (Phase 4), and surrogate (Phase 5).
"""

import random
import numpy as np
from typing import List
from search_space.chromosome import (
    CHROM_LENGTH, GENE_BOUNDS, FILTER_MAP, KERNEL_MAP,
    DENSE_MAP, ACTIVATION_MAP, POOL_MAP,
    G_NUM_CONV, G_FILTERS_1, G_KERNEL, G_POOL,
    G_NUM_DENSE, G_DENSE_UNITS, G_ACTIVATION, G_BATCHNORM, G_SKIP,
)


# ─────────────────────────────────────────────────────────────────────
#  CHROMOSOME GENERATION
# ─────────────────────────────────────────────────────────────────────
def random_chromosome() -> List[int]:
    """Sample a uniformly random valid chromosome."""
    return [random.randint(lo, hi) for lo, hi in GENE_BOUNDS]


def random_population(size: int) -> List[List[int]]:
    """Generate a population of `size` random chromosomes."""
    return [random_chromosome() for _ in range(size)]


def clip_chromosome(chrom: List[int]) -> List[int]:
    """
    Clip each gene to its valid range.
    Call this after mutation to avoid out-of-bounds genes.
    """
    return [int(np.clip(v, lo, hi)) for v, (lo, hi) in zip(chrom, GENE_BOUNDS)]


def is_valid_chromosome(chrom: List[int]) -> bool:
    """Return True iff all genes are within their valid bounds."""
    if len(chrom) != CHROM_LENGTH:
        return False
    return all(lo <= v <= hi for v, (lo, hi) in zip(chrom, GENE_BOUNDS))


# ─────────────────────────────────────────────────────────────────────
#  SURROGATE FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────
def chromosome_to_features(chrom: List[int]) -> np.ndarray:
    """
    Convert chromosome to a float feature vector for the surrogate model.
    We decode categorical genes into their actual numeric values so the
    surrogate sees meaningful magnitudes (e.g., 64 filters vs 512),
    not arbitrary indices.

    Feature vector (length = 13):
      [0]  num_conv_layers   (raw int)
      [1]  filters_1         (actual channels)
      [2]  filters_2
      [3]  filters_3
      [4]  filters_4
      [5]  kernel_size       (actual pixels)
      [6]  pool_type         (index: 0/1/2)
      [7]  num_dense
      [8]  dense_units       (actual units)
      [9]  activation_idx
      [10] use_batchnorm
      [11] use_skip
      [12] total_filter_capacity  (sum of active filters — a proxy for width)
    """
    n = chrom[G_NUM_CONV]
    active_filters = [FILTER_MAP[chrom[G_FILTERS_1 + i]] for i in range(n)]
    all_filters    = [FILTER_MAP[chrom[G_FILTERS_1 + i]] for i in range(4)]
    total_capacity = sum(active_filters)

    feats = [
        float(n),
        float(all_filters[0]),
        float(all_filters[1]),
        float(all_filters[2]),
        float(all_filters[3]),
        float(KERNEL_MAP[chrom[G_KERNEL]]),
        float(chrom[G_POOL]),
        float(chrom[G_NUM_DENSE]),
        float(DENSE_MAP[chrom[G_DENSE_UNITS]]),
        float(chrom[G_ACTIVATION]),
        float(chrom[G_BATCHNORM]),
        float(chrom[G_SKIP]),
        float(total_capacity),
    ]
    return np.array(feats, dtype=np.float32)


def population_to_features(population: List[List[int]]) -> np.ndarray:
    """Convert entire population to a 2D feature matrix (N x 13)."""
    return np.stack([chromosome_to_features(c) for c in population])


# ─────────────────────────────────────────────────────────────────────
#  ARCHITECTURE COMPLEXITY ESTIMATORS (used in multi-objective)
# ─────────────────────────────────────────────────────────────────────
def estimate_parameter_count(
    chrom       : List[int],
    in_channels : int = 1,
) -> int:
    """
    Fast analytical parameter count estimate.
    Avoids building the full nn.Module — useful for quick Pareto filtering.

    Note: this is an estimate. Use model.get_info()["num_params"] for exact count.
    """
    n_conv   = chrom[G_NUM_CONV]
    filters  = [FILTER_MAP[chrom[G_FILTERS_1 + i]] for i in range(n_conv)]
    kernel   = KERNEL_MAP[chrom[G_KERNEL]]
    use_bn   = bool(chrom[G_BATCHNORM])
    n_dense  = chrom[G_NUM_DENSE]
    d_units  = DENSE_MAP[chrom[G_DENSE_UNITS]]

    total = 0
    prev  = in_channels

    # Conv params
    for f in filters:
        total += f * prev * kernel * kernel  # weight
        if not use_bn:
            total += f  # bias
        if use_bn:
            total += 2 * f  # gamma + beta
        prev = f

    # After AdaptiveAvgPool2d(4,4)
    flat = prev * 4 * 4

    # Dense params
    curr = flat
    for _ in range(n_dense):
        total += curr * d_units + d_units
        curr = d_units
    total += curr * 10 + 10  # final classifier

    return total
