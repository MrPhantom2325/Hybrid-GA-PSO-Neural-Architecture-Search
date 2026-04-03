
"""
evaluation/pareto.py
Pareto dominance, non-dominated sorting, crowding distance.
All objectives are passed as MINIMISATION problems
(negate accuracy before passing in).
"""
import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Returns True if solution a dominates solution b.
    a dominates b if:
      - a is no worse than b in all objectives, AND
      - a is strictly better in at least one objective.
    All objectives assumed to be MINIMISATION.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(F: np.ndarray) -> list:
    """
    NSGA-II fast non-dominated sorting.
    F: (N, n_obj) array of objective values (all minimisation).
    Returns list of fronts, each front is a list of indices.
    Front 0 = Pareto optimal set.
    """
    N = len(F)
    dominated_count = np.zeros(N, dtype=int)    # how many solutions dominate i
    dominates_set   = [[] for _ in range(N)]    # set of solutions i dominates

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(F[i], F[j]):
                dominates_set[i].append(j)
            elif dominates(F[j], F[i]):
                dominated_count[i] += 1

    fronts     = []
    current    = [i for i in range(N) if dominated_count[i] == 0]
    while current:
        fronts.append(current)
        next_front = []
        for i in current:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


def crowding_distance(F: np.ndarray, front: list) -> np.ndarray:
    """
    Compute crowding distance for solutions in a front.
    Higher distance = more diverse = preferred when ranks are equal.
    Returns array of distances, indexed by position in `front`.
    """
    n    = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist

    F_front = F[front]   # (n, n_obj)
    n_obj   = F_front.shape[1]

    for m in range(n_obj):
        order  = np.argsort(F_front[:, m])
        f_min  = F_front[order[0],  m]
        f_max  = F_front[order[-1], m]
        span   = f_max - f_min
        if span == 0:
            continue
        # Boundary points get infinite distance
        dist[order[0]]  = np.inf
        dist[order[-1]] = np.inf
        for i in range(1, n - 1):
            dist[order[i]] += (F_front[order[i+1], m] -
                               F_front[order[i-1], m]) / span
    return dist


def extract_pareto_front(objectives: np.ndarray):
    """
    Given (N, 3) objectives array (all minimisation),
    return indices of Pareto-optimal solutions (front 0).
    """
    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if fronts else []


def pareto_rank_all(objectives: np.ndarray) -> np.ndarray:
    """
    Assign a Pareto rank (0 = best) to every solution.
    Returns array of ranks, shape (N,).
    """
    fronts = fast_non_dominated_sort(objectives)
    ranks  = np.zeros(len(objectives), dtype=int)
    for rank, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank
    return ranks
