
"""
ga/population.py
Population manager: holds chromosomes, fitnesses, metadata,
and the full evaluation history across generations.
"""
import json, os, copy
import numpy as np
from ga.operators import random_chromosome


class Individual:
    """
    A single candidate architecture with its fitness and metadata.
    """
    __slots__ = [
        "chromosome", "fitness", "num_params",
        "train_time", "val_accs", "evaluated", "uid"
    ]

    def __init__(self, chromosome, uid=None):
        self.chromosome  = list(chromosome)
        self.fitness     = None   # val_accuracy after proxy training
        self.num_params  = None
        self.train_time  = None
        self.val_accs    = []     # per-epoch val accuracy during proxy train
        self.evaluated   = False
        self.uid         = uid    # unique id across all generations

    def to_dict(self):
        return {
            "uid"        : self.uid,
            "chromosome" : self.chromosome,
            "fitness"    : self.fitness,
            "num_params" : self.num_params,
            "train_time" : self.train_time,
            "val_accs"   : self.val_accs,
        }

    def __repr__(self):
        f = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Individual(uid={self.uid}, fitness={f}, chrom={self.chromosome})"


class Population:
    """
    Container for a generation's individuals.
    Provides bulk-initialisation and fitness access.
    """
    def __init__(self, size: int, start_uid: int = 0):
        self.size       = size
        self._uid_ctr   = start_uid
        self.individuals: list[Individual] = []
        self._init_random()

    def _next_uid(self):
        uid = self._uid_ctr
        self._uid_ctr += 1
        return uid

    def _init_random(self):
        self.individuals = [
            Individual(random_chromosome(), uid=self._next_uid())
            for _ in range(self.size)
        ]

    def chromosomes(self):
        return [ind.chromosome for ind in self.individuals]

    def fitnesses(self):
        return [ind.fitness if ind.fitness is not None else 0.0
                for ind in self.individuals]

    def best(self):
        evaluated = [i for i in self.individuals if i.evaluated]
        if not evaluated:
            return None
        return max(evaluated, key=lambda i: i.fitness)

    def from_chromosomes(self, chroms):
        """Replace population individuals with a new list of chromosomes."""
        self.individuals = [
            Individual(c, uid=self._next_uid()) for c in chroms
        ]

    def __len__(self):
        return len(self.individuals)


class GAHistory:
    """
    Accumulates per-generation statistics and all evaluated individuals.
    Used for convergence plots and downstream surrogate training.
    """
    def __init__(self):
        self.generations: list[dict]       = []   # per-gen stats
        self.all_individuals: list[dict]   = []   # every evaluated arch

    def record_generation(self, gen: int, population: Population):
        fits     = [i.fitness for i in population.individuals if i.evaluated]
        if not fits:
            return
        best_ind = population.best()
        stat = {
            "gen"         : gen,
            "best"        : max(fits),
            "avg"         : float(np.mean(fits)),
            "worst"       : min(fits),
            "std"         : float(np.std(fits)),
            "best_chrom"  : best_ind.chromosome if best_ind else None,
            "best_params" : best_ind.num_params  if best_ind else None,
        }
        self.generations.append(stat)
        # Also store every individual for surrogate
        for ind in population.individuals:
            if ind.evaluated:
                d = ind.to_dict()
                d["gen"] = gen
                self.all_individuals.append(d)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"generations": self.generations,
                       "individuals": self.all_individuals}, f, indent=2)

    @classmethod
    def load(cls, path: str):
        h = cls()
        with open(path) as f:
            data = json.load(f)
        h.generations      = data["generations"]
        h.all_individuals  = data["individuals"]
        return h

    def surrogate_dataset(self):
        """Return (X, y) arrays ready for training the surrogate model."""
        import numpy as np
        from search_space.chromosome import chromosome_to_features
        X, y = [], []
        for ind in self.all_individuals:
            if ind["fitness"] is not None:
                X.append(chromosome_to_features(ind["chromosome"]))
                y.append(ind["fitness"])
        return np.array(X), np.array(y)
