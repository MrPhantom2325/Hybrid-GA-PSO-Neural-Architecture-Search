
"""
ga/operators.py
GA genetic operators: selection, crossover, mutation.
All operate on plain Python lists (chromosomes).
"""
import random
import copy

# ── Gene bounds — must match chromosome.py ────────────────────────────
# Index → (min_val, max_val)  [inclusive]
GENE_BOUNDS = [
    (1, 4),   # [0]  num_conv_layers
    (0, 5),   # [1]  filters_1_idx
    (0, 5),   # [2]  filters_2_idx
    (0, 5),   # [3]  filters_3_idx
    (0, 5),   # [4]  filters_4_idx
    (0, 2),   # [5]  kernel_idx
    (0, 2),   # [6]  pool_idx
    (1, 3),   # [7]  num_dense
    (0, 4),   # [8]  dense_units_idx
    (0, 3),   # [9]  activation_idx
    (0, 1),   # [10] use_batchnorm
    (0, 1),   # [11] use_skip
]


def random_chromosome():
    """Sample one random valid chromosome."""
    return [random.randint(lo, hi) for lo, hi in GENE_BOUNDS]


def tournament_selection(population, fitnesses, k=3):
    """
    Tournament selection: pick k individuals at random,
    return the one with highest fitness.
    Population and fitnesses are parallel lists.
    """
    candidates = random.sample(range(len(population)), k)
    winner     = max(candidates, key=lambda i: fitnesses[i])
    return copy.deepcopy(population[winner])


def uniform_crossover(parent1, parent2, cx_prob=0.8):
    """
    Uniform crossover: each gene independently inherited
    from either parent with equal probability.
    Only applied with probability cx_prob.
    Returns two new children.
    """
    if random.random() > cx_prob:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    child1, child2 = [], []
    for g1, g2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(g1); child2.append(g2)
        else:
            child1.append(g2); child2.append(g1)
    return child1, child2


def integer_mutation(chromosome, mut_prob=0.2, gene_mut_prob=0.1):
    """
    Integer mutation: with probability mut_prob, attempt to
    mutate the chromosome. For each gene, with probability
    gene_mut_prob, resample it uniformly within its bounds.
    Returns a new mutated chromosome.
    """
    child = copy.deepcopy(chromosome)
    if random.random() > mut_prob:
        return child
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        if random.random() < gene_mut_prob:
            child[i] = random.randint(lo, hi)
    return child


def elitism(population, fitnesses, k=2):
    """
    Return the top-k individuals (by fitness) to carry
    unchanged into the next generation.
    """
    ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    return [copy.deepcopy(population[i]) for i in ranked[:k]]


def diversity_score(population):
    """
    Simple population diversity metric: mean pairwise
    Hamming distance between chromosomes, normalised [0,1].
    Useful for detecting premature convergence.
    """
    n    = len(population)
    L    = len(population[0])
    if n < 2:
        return 0.0
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = sum(1 for a, b in zip(population[i], population[j]) if a != b)
            total += diff
            count += 1
    return (total / count) / L   # normalise by chromosome length
