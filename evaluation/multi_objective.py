
"""
evaluation/multi_objective.py
Multi-objective evaluation using NSGA-II via pymoo.

Objectives (all converted to MINIMISATION):
  obj_0 = 1 - accuracy      (minimise → maximise accuracy)
  obj_1 = num_params / 1e6  (minimise model size in millions)
  obj_2 = train_time / 60   (minimise training time in minutes)
"""
import numpy as np
from evaluation.pareto import extract_pareto_front, pareto_rank_all, crowding_distance


def build_objective_matrix(architectures: list) -> np.ndarray:
    """
    Convert architecture dicts to (N, 3) objective matrix.
    All objectives are MINIMISATION.
    """
    F = np.array([
        [
            1.0 - a["accuracy"],           # obj 0: error rate
            a["num_params"] / 1_000_000,    # obj 1: params in M
            a["train_time"]  / 60.0,        # obj 2: time in minutes
        ]
        for a in architectures
    ], dtype=np.float64)
    return F


def normalise_objectives(F: np.ndarray) -> np.ndarray:
    """
    Min-max normalise each objective to [0, 1].
    Needed for fair crowding distance calculation.
    """
    F_norm = F.copy()
    for j in range(F.shape[1]):
        col = F[:, j]
        span = col.max() - col.min()
        if span > 0:
            F_norm[:, j] = (col - col.min()) / span
    return F_norm


def run_pareto_analysis(architectures: list) -> dict:
    """
    Full multi-objective analysis on a list of architecture dicts.

    Returns:
      F             : (N, 3) raw objective matrix
      F_norm        : (N, 3) normalised objectives
      pareto_indices: indices of Pareto-optimal solutions
      pareto_front  : list of Pareto-optimal architecture dicts
      ranks         : Pareto rank for every architecture
      crowding      : crowding distances for Pareto front
    """
    F       = build_objective_matrix(architectures)
    F_norm  = normalise_objectives(F)

    pareto_idx  = extract_pareto_front(F_norm)
    ranks       = pareto_rank_all(F_norm)
    crowd_dist  = crowding_distance(F_norm, pareto_idx)

    pareto_archs = []
    for i, idx in enumerate(pareto_idx):
        a = architectures[idx].copy()
        a["pareto_rank"]      = 0
        a["crowding_dist"]    = float(crowd_dist[i])
        a["obj_error"]        = float(F[idx, 0])
        a["obj_params_M"]     = float(F[idx, 1])
        a["obj_time_min"]     = float(F[idx, 2])
        pareto_archs.append(a)

    # Sort Pareto front by accuracy (descending)
    pareto_archs.sort(key=lambda x: x["accuracy"], reverse=True)

    # Annotate all architectures with their rank
    for i, a in enumerate(architectures):
        a["pareto_rank"] = int(ranks[i])

    return {
        "F"             : F,
        "F_norm"        : F_norm,
        "pareto_indices": pareto_idx,
        "pareto_front"  : pareto_archs,
        "ranks"         : ranks,
        "crowding"      : crowd_dist,
        "n_total"       : len(architectures),
        "n_pareto"      : len(pareto_idx),
    }


def select_best_balanced(pareto_front: list,
                          acc_weight: float = 0.6,
                          param_weight: float = 0.2,
                          time_weight: float = 0.2) -> dict:
    """
    From the Pareto front, select the architecture with
    best weighted combination of normalised objectives.
    Default weights favour accuracy (60%) over size/speed.
    """
    if not pareto_front:
        return None
    accs   = np.array([a["accuracy"]    for a in pareto_front])
    params = np.array([a["num_params"]  for a in pareto_front], dtype=float)
    times  = np.array([a["train_time"]  for a in pareto_front])

    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)

    score = (acc_weight   * norm(accs)           # higher = better
           - param_weight * norm(params)          # lower  = better
           - time_weight  * norm(times))          # lower  = better
    best_idx = int(np.argmax(score))
    return pareto_front[best_idx]
