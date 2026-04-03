
"""
search_space/chromosome.py

Chromosome encoding and decoding for the Hybrid GA-PSO NAS.

Gene layout (length = 12):
  [0]  num_conv_layers   int  1–4        number of conv blocks
  [1]  filters_1         int  0–5        channel count index (see FILTER_MAP)
  [2]  filters_2         int  0–5
  [3]  filters_3         int  0–5
  [4]  filters_4         int  0–5
  [5]  kernel_size       int  0–2        index into KERNEL_MAP
  [6]  pool_type         int  0–2        0=max, 1=avg, 2=none
  [7]  num_dense         int  1–3        number of FC layers after flatten
  [8]  dense_units       int  0–4        units index (see DENSE_MAP)
  [9]  activation        int  0–3        index into ACTIVATION_MAP
  [10] use_batchnorm     int  0–1        0=no, 1=yes
  [11] use_skip          int  0–1        0=no residual, 1=residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────
#  LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────────
CHROM_LENGTH    = 12
FILTER_MAP      = [16, 32, 64, 128, 256, 512]
KERNEL_MAP      = [3, 5, 7]
DENSE_MAP       = [64, 128, 256, 512, 1024]
ACTIVATION_MAP  = ["relu", "elu", "leaky_relu", "selu"]
POOL_MAP        = ["max", "avg", "none"]

# Gene index constants — use these names everywhere, never raw indices
G_NUM_CONV   = 0
G_FILTERS_1  = 1
G_FILTERS_2  = 2
G_FILTERS_3  = 3
G_FILTERS_4  = 4
G_KERNEL     = 5
G_POOL       = 6
G_NUM_DENSE  = 7
G_DENSE_UNITS = 8
G_ACTIVATION = 9
G_BATCHNORM  = 10
G_SKIP       = 11

# Per-gene valid ranges [min, max] (inclusive)
GENE_BOUNDS = [
    (1, 4),  # 0  num_conv_layers
    (0, 5),  # 1  filters_1
    (0, 5),  # 2  filters_2
    (0, 5),  # 3  filters_3
    (0, 5),  # 4  filters_4
    (0, 2),  # 5  kernel_size
    (0, 2),  # 6  pool_type
    (1, 3),  # 7  num_dense
    (0, 4),  # 8  dense_units
    (0, 3),  # 9  activation
    (0, 1),  # 10 use_batchnorm
    (0, 1),  # 11 use_skip
]


# ─────────────────────────────────────────────────────────────────────
#  HELPER: get activation function by name
# ─────────────────────────────────────────────────────────────────────
def _get_activation(name: str) -> nn.Module:
    return {
        "relu"      : nn.ReLU(inplace=True),
        "elu"       : nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "selu"      : nn.SELU(inplace=True),
    }[name]


# ─────────────────────────────────────────────────────────────────────
#  CONV BLOCK: conv → (batchnorm) → activation → (pool)
# ─────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """
    A single convolutional block with optional BatchNorm, pooling,
    and residual skip connection.

    Skip connection: added if use_skip=True AND in_channels == out_channels.
    If the channel dimensions differ, a 1x1 conv projection is used.
    """

    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : int,
        activation   : str,
        use_batchnorm: bool,
        pool_type    : str,
        use_skip     : bool,
    ):
        super().__init__()
        padding = kernel_size // 2  # same-padding: output H,W = input H,W

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_get_activation(activation))

        self.conv_path = nn.Sequential(*layers)

        # ── Pooling ───────────────────────────────────────────────
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

        # ── Skip connection ───────────────────────────────────────
        self.use_skip = use_skip
        if use_skip:
            if in_channels != out_channels:
                # Project residual to match out_channels
                self.skip_proj = nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1, bias=False)
            else:
                self.skip_proj = nn.Identity()
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_path(x)
        if self.use_skip:
            out = out + self.skip_proj(x)
        if self.pool is not None:
            out = self.pool(out)
        return out


# ─────────────────────────────────────────────────────────────────────
#  NAS MODEL: assembled from a chromosome
# ─────────────────────────────────────────────────────────────────────
class NASModel(nn.Module):
    """
    Dynamically built CNN whose architecture is fully determined
    by a 12-gene chromosome.
    """

    def __init__(
        self,
        chromosome   : List[int],
        in_channels  : int = 1,    # 1 for MNIST, 3 for CIFAR-10
        image_size   : int = 28,   # 28 for MNIST, 32 for CIFAR-10
        num_classes  : int = 10,
    ):
        super().__init__()
        self.chromosome = list(chromosome)

        # ── Decode genes ──────────────────────────────────────────
        num_conv    = chromosome[G_NUM_CONV]
        filters     = [FILTER_MAP[chromosome[G_FILTERS_1 + i]] for i in range(4)]
        kernel_size = KERNEL_MAP[chromosome[G_KERNEL]]
        pool_type   = POOL_MAP[chromosome[G_POOL]]
        num_dense   = chromosome[G_NUM_DENSE]
        dense_units = DENSE_MAP[chromosome[G_DENSE_UNITS]]
        activation  = ACTIVATION_MAP[chromosome[G_ACTIVATION]]
        use_bn      = bool(chromosome[G_BATCHNORM])
        use_skip    = bool(chromosome[G_SKIP])

        # ── Convolutional backbone ────────────────────────────────
        conv_blocks = []
        current_channels = in_channels
        for i in range(num_conv):
            out_ch = filters[i]
            # Only add pooling after every 2nd conv block to avoid
            # spatial dimension collapsing on small images (MNIST=28px)
            block_pool = pool_type if (i % 2 == 1) else "none"
            conv_blocks.append(
                ConvBlock(
                    in_channels   = current_channels,
                    out_channels  = out_ch,
                    kernel_size   = kernel_size,
                    activation    = activation,
                    use_batchnorm = use_bn,
                    pool_type     = block_pool,
                    use_skip      = use_skip,
                )
            )
            current_channels = out_ch
        self.conv_backbone = nn.Sequential(*conv_blocks)

        # ── Adaptive pooling: always output 4x4 regardless of input size ──
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_size = current_channels * 4 * 4

        # ── Fully-connected head ──────────────────────────────────
        fc_layers = []
        current_size = flat_size
        for j in range(num_dense):
            fc_layers.append(nn.Linear(current_size, dense_units))
            fc_layers.append(_get_activation(activation))
            fc_layers.append(nn.Dropout(p=0.3))  # PSO will tune this later
            current_size = dense_units
        fc_layers.append(nn.Linear(current_size, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_backbone(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_info(self) -> dict:
        """Return a summary dict — useful for logging and the surrogate."""
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "chromosome"   : self.chromosome,
            "num_params"   : num_params,
            "num_conv"     : self.chromosome[G_NUM_CONV],
            "activation"   : ACTIVATION_MAP[self.chromosome[G_ACTIVATION]],
            "use_batchnorm": bool(self.chromosome[G_BATCHNORM]),
            "use_skip"     : bool(self.chromosome[G_SKIP]),
        }


# ─────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────
def decode_chromosome(
    chromosome  : List[int],
    in_channels : int = 1,
    image_size  : int = 28,
    num_classes : int = 10,
) -> NASModel:
    """
    Convert a chromosome vector into a runnable NASModel.

    Args:
        chromosome  : list of 12 integers (see GENE_BOUNDS for ranges)
        in_channels : 1 for MNIST/grayscale, 3 for CIFAR-10/RGB
        image_size  : 28 for MNIST, 32 for CIFAR-10
        num_classes : number of output classes (default 10)

    Returns:
        NASModel: an nn.Module ready for .to(device) and training

    Raises:
        ValueError: if chromosome length != 12 or any gene is out of bounds
    """
    if len(chromosome) != CHROM_LENGTH:
        raise ValueError(f"Chromosome must have {CHROM_LENGTH} genes, got {len(chromosome)}")
    for i, (val, (lo, hi)) in enumerate(zip(chromosome, GENE_BOUNDS)):
        if not (lo <= val <= hi):
            raise ValueError(f"Gene {i} = {val} is out of bounds [{lo}, {hi}]")
    return NASModel(chromosome, in_channels, image_size, num_classes)


def chromosome_to_description(chromosome: List[int]) -> str:
    """
    Human-readable summary of what a chromosome encodes.
    Useful for logging and paper examples.
    """
    n = chromosome[G_NUM_CONV]
    filt = [FILTER_MAP[chromosome[G_FILTERS_1 + i]] for i in range(n)]
    k    = KERNEL_MAP[chromosome[G_KERNEL]]
    pool = POOL_MAP[chromosome[G_POOL]]
    nd   = chromosome[G_NUM_DENSE]
    du   = DENSE_MAP[chromosome[G_DENSE_UNITS]]
    act  = ACTIVATION_MAP[chromosome[G_ACTIVATION]]
    bn   = "BN" if chromosome[G_BATCHNORM] else "no-BN"
    skip = "skip" if chromosome[G_SKIP] else "no-skip"
    return (
        f"{n}×Conv{k}({','.join(map(str,filt))}) "
        f"pool={pool} | "
        f"{nd}×FC({du}) "
        f"act={act} {bn} {skip}"
    )
