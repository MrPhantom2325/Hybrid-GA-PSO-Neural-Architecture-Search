
"""
search_space/architecture_validator.py

Pre-flight validation for chromosomes before decoding and training.
Filters out architectures that would crash or produce degenerate models.
"""

from typing import List, Tuple
from search_space.chromosome import (
    GENE_BOUNDS, CHROM_LENGTH,
    FILTER_MAP, KERNEL_MAP, POOL_MAP, DENSE_MAP,
    G_NUM_CONV, G_FILTERS_1, G_KERNEL, G_POOL,
    G_NUM_DENSE, G_DENSE_UNITS,
)

# Architecture constraints
MAX_PARAMS      = 10_000_000   # 10M params hard cap
MIN_SPATIAL     = 2            # spatial dims must stay >= 2 before adaptive pool
MAX_KERNEL_RATIO = 0.5         # kernel cannot be > 50% of spatial size


def validate_chromosome(
    chrom      : List[int],
    in_channels: int = 1,
    image_size : int = 28,
) -> Tuple[bool, str]:
    """
    Validate a chromosome before building the model.

    Returns:
        (True, "")          if valid
        (False, reason_str) if invalid
    """
    # 1. Length and bounds
    if len(chrom) != CHROM_LENGTH:
        return False, f"Length {len(chrom)} != {CHROM_LENGTH}"

    for i, (v, (lo, hi)) in enumerate(zip(chrom, GENE_BOUNDS)):
        if not (lo <= v <= hi):
            return False, f"Gene {i} = {v} out of bounds [{lo},{hi}]"

    # 2. Spatial dimension check
    # Simulate spatial progression through conv+pool blocks
    spatial = image_size
    n_conv  = chrom[G_NUM_CONV]
    kernel  = KERNEL_MAP[chrom[G_KERNEL]]
    pool    = POOL_MAP[chrom[G_POOL]]

    if kernel > spatial * MAX_KERNEL_RATIO * 2:
        return False, f"Kernel {kernel} too large for image size {image_size}"

    for i in range(n_conv):
        # Pool applied every 2nd block
        if i % 2 == 1 and pool != "none":
            spatial = spatial // 2
        if spatial < MIN_SPATIAL:
            return False, f"Spatial collapse after block {i}: size={spatial}"

    # 3. Parameter budget
    from search_space.search_space_utils import estimate_parameter_count
    est_params = estimate_parameter_count(chrom, in_channels)
    if est_params > MAX_PARAMS:
        return False, f"Estimated params {est_params:,} exceeds budget {MAX_PARAMS:,}"

    return True, ""


def filter_population(
    population : List[List[int]],
    in_channels: int = 1,
    image_size : int = 28,
    verbose    : bool = False,
) -> List[List[int]]:
    """
    Remove invalid chromosomes from a population.
    Used at the start of each GA generation.
    """
    valid = []
    for chrom in population:
        ok, reason = validate_chromosome(chrom, in_channels, image_size)
        if ok:
            valid.append(chrom)
        elif verbose:
            print(f"  ⚠️  Removed chromosome {chrom}: {reason}")
    return valid
