
import numpy as np
import time

BOUNDS = np.array([
    [-5.0, -1.0],   # log10(lr)
    [ 0.0,  0.6],   # dropout
    [ 4.0,  7.0],   # log2(batch_size)
], dtype=np.float64)   # shape (3, 2)

DIM_NAMES = ["log10_lr", "dropout", "log2_batch"]

def decode_particle(position):
    pos = np.asarray(position, dtype=np.float64)
    log_lr, dropout, log2_bs = pos[0], pos[1], pos[2]
    return {
        "lr"         : float(10 ** log_lr),
        "dropout"    : float(np.clip(dropout, 0.0, 0.6)),
        "batch_size" : int(2 ** round(float(log2_bs))),
    }

class Particle:
    def __init__(self, rng):
        # Always use module-level BOUNDS — never accept bounds as arg
        n_dims = BOUNDS.shape[0]                          # always 3
        self.position  = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])   # (3,)
        ranges         = BOUNDS[:, 1] - BOUNDS[:, 0]
        self.velocity  = rng.uniform(-ranges * 0.1, ranges * 0.1)  # (3,)
        self.pbest_position = self.position.copy()
        self.pbest_fitness  = -np.inf
        self.fitness_history = []

    def update_velocity(self, gbest_position, w, c1, c2, rng):
        gbest = np.asarray(gbest_position, dtype=np.float64)
        r1 = rng.uniform(0, 1, size=3)
        r2 = rng.uniform(0, 1, size=3)
        cognitive      = c1 * r1 * (self.pbest_position - self.position)
        social         = c2 * r2 * (gbest              - self.position)
        self.velocity  = w * self.velocity + cognitive + social

    def update_position(self):
        self.position = self.position + self.velocity
        for i in range(3):
            if self.position[i] < BOUNDS[i, 0]:
                self.position[i]  = BOUNDS[i, 0]
                self.velocity[i] *= -0.5
            elif self.position[i] > BOUNDS[i, 1]:
                self.position[i]  = BOUNDS[i, 1]
                self.velocity[i] *= -0.5

    def update_pbest(self, fitness):
        self.fitness_history.append(fitness)
        if fitness > self.pbest_fitness:
            self.pbest_fitness  = fitness
            self.pbest_position = self.position.copy()


class PSOOptimizer:
    def __init__(self, fitness_fn, config, seed=42):
        self.fitness_fn   = fitness_fn
        self.n_particles  = config["num_particles"]
        self.n_iterations = config["num_iterations"]
        self.w            = config["w"]
        self.c1           = config["c1"]
        self.c2           = config["c2"]
        self.w_decay      = config["w_decay"]
        self.verbose      = config.get("verbose", True)
        self.rng          = np.random.default_rng(seed)
        self.swarm        = [Particle(self.rng) for _ in range(self.n_particles)]
        self.gbest_position = None
        self.gbest_fitness  = -np.inf
        self.history        = []
        self.all_evals      = []

    def _update_gbest(self, particle):
        if particle.pbest_fitness > self.gbest_fitness:
            self.gbest_fitness  = particle.pbest_fitness
            self.gbest_position = particle.pbest_position.copy()

    def run(self):
        total_start = time.time()
        print("=" * 55)
        print(f"  PSO  |  Particles: {self.n_particles}  |  Iters: {self.n_iterations}")
        print("=" * 55)

        # Verify shapes on first particle before any loop
        p0 = self.swarm[0]
        assert p0.position.shape == (3,), f"Bad position shape: {p0.position.shape}"
        assert p0.velocity.shape == (3,), f"Bad velocity shape: {p0.velocity.shape}"

        w = self.w
        for iteration in range(self.n_iterations):
            iter_start     = time.time()
            iter_fitnesses = []

            for p_idx, particle in enumerate(self.swarm):
                fitness = self.fitness_fn(particle.position)
                particle.update_pbest(fitness)
                self._update_gbest(particle)
                iter_fitnesses.append(fitness)
                hp = decode_particle(particle.position)
                self.all_evals.append({
                    "iteration" : iteration,
                    "particle"  : p_idx,
                    "position"  : particle.position.tolist(),
                    "fitness"   : fitness,
                    "lr"        : hp["lr"],
                    "dropout"   : hp["dropout"],
                    "batch_size": hp["batch_size"],
                })

            for particle in self.swarm:
                particle.update_velocity(self.gbest_position, w, self.c1, self.c2, self.rng)
                particle.update_position()

            w *= self.w_decay

            best_hp = decode_particle(self.gbest_position)
            stat = {
                "iteration"    : iteration,
                "gbest_fitness": self.gbest_fitness,
                "iter_best"    : max(iter_fitnesses),
                "iter_avg"     : float(np.mean(iter_fitnesses)),
                "iter_std"     : float(np.std(iter_fitnesses)),
                "w"            : w,
                "best_lr"      : best_hp["lr"],
                "best_dropout" : best_hp["dropout"],
                "best_batch"   : best_hp["batch_size"],
            }
            self.history.append(stat)

            if self.verbose:
                elapsed = time.time() - iter_start
                print(f"  Iter {iteration+1:3d}/{self.n_iterations}  "
                      f"gbest={self.gbest_fitness:.4f}  "
                      f"iter_best={max(iter_fitnesses):.4f}  "
                      f"lr={best_hp['lr']:.2e}  "
                      f"drop={best_hp['dropout']:.2f}  "
                      f"bs={best_hp['batch_size']:3d}  "
                      f"({elapsed:.1f}s)")

        best_hp    = decode_particle(self.gbest_position)
        total_time = time.time() - total_start
        print("\n" + "=" * 55)
        print(f"  Best fitness : {self.gbest_fitness:.4f}  |  Time: {total_time/60:.1f} min")
        print(f"  lr={best_hp['lr']:.2e}  dropout={best_hp['dropout']:.3f}  batch={best_hp['batch_size']}")
        print("=" * 55)

        return {
            "best_position"   : self.gbest_position.tolist(),
            "best_fitness"    : self.gbest_fitness,
            "best_hyperparams": best_hp,
            "history"         : self.history,
            "all_evals"       : self.all_evals,
            "total_time_s"    : total_time,
        }
