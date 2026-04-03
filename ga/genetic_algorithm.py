
"""
ga/genetic_algorithm.py
Full GA engine for NAS.
"""
import time, copy
import torch
from ga.operators import (
    tournament_selection, uniform_crossover,
    integer_mutation, elitism, diversity_score
)
from ga.population import Population, GAHistory
from search_space.chromosome import decode_chromosome
from training.proxy_trainer import proxy_train


class GeneticAlgorithm:
    def __init__(
        self,
        train_loader,
        val_loader,
        device,
        ga_config      : dict,
        training_config: dict,
        in_channels    : int   = 1,
        image_size     : int   = 28,
        num_classes    : int   = 10,
        results_dir    : str   = "/content/nas_project/results",
        verbose        : bool  = True,
    ):
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.cfg            = ga_config
        self.train_cfg      = training_config
        self.in_channels    = in_channels
        self.image_size     = image_size
        self.num_classes    = num_classes
        self.results_dir    = results_dir
        self.verbose        = verbose

        self.pop_size    = ga_config["population_size"]
        self.num_gens    = ga_config["num_generations"]
        self.cx_prob     = ga_config["crossover_prob"]
        self.mut_prob    = ga_config["mutation_prob"]
        self.gene_mut_p  = ga_config["gene_mutation_prob"]
        self.tourn_size  = ga_config["tournament_size"]
        self.elitism_k   = ga_config["elitism_k"]
        self.proxy_epochs= training_config["proxy_epochs"]

        self.population  = Population(size=self.pop_size)
        self.history     = GAHistory()
        self.gen         = 0

    def _evaluate_individual(self, individual):
        model = decode_chromosome(
            individual.chromosome,
            in_channels=self.in_channels,
            image_size=self.image_size,
            num_classes=self.num_classes,
        )
        result = proxy_train(
            model,
            self.train_loader,
            self.val_loader,
            self.device,
            epochs=self.proxy_epochs,
        )
        individual.fitness    = result["val_accuracy"]
        individual.num_params = result["num_params"]
        individual.train_time = result["train_time"]
        individual.val_accs   = result["val_accs"]
        individual.evaluated  = True
        return result

    def _evaluate_population(self):
        n_eval = sum(1 for i in self.population.individuals if not i.evaluated)
        if self.verbose:
            print(f"  Evaluating {n_eval} architectures ({self.proxy_epochs} proxy epochs each)...")
        for idx, ind in enumerate(self.population.individuals):
            if ind.evaluated:
                continue
            t0 = time.time()
            self._evaluate_individual(ind)
            elapsed = time.time() - t0
            if self.verbose:
                print(f"    [{idx+1:2d}/{self.pop_size}] uid={ind.uid:3d}  val_acc={ind.fitness:.4f}  params={ind.num_params:,}  time={elapsed:.1f}s")

    def _breed_next_generation(self):
        """Create next generation: elites + offspring, preserving elite data via deepcopy."""
        current_inds = self.population.individuals
        ranked = sorted(range(len(current_inds)), key=lambda i: current_inds[i].fitness if current_inds[i].fitness else 0, reverse=True)

        next_inds = []
        for i in range(min(self.elitism_k, len(ranked))):
            elite = copy.deepcopy(current_inds[ranked[i]])
            next_inds.append(elite)

        chroms = [ind.chromosome for ind in current_inds]
        fits   = [ind.fitness if ind.fitness else 0 for ind in current_inds]

        while len(next_inds) < self.pop_size:
            p1 = tournament_selection(chroms, fits, k=self.tourn_size)
            p2 = tournament_selection(chroms, fits, k=self.tourn_size)
            c1, c2 = uniform_crossover(p1, p2, cx_prob=self.cx_prob)
            c1 = integer_mutation(c1, self.mut_prob, self.gene_mut_p)
            c2 = integer_mutation(c2, self.mut_prob, self.gene_mut_p)

            from ga.population import Individual
            next_inds.append(Individual(c1, uid=self.population._next_uid()))
            if len(next_inds) < self.pop_size:
                next_inds.append(Individual(c2, uid=self.population._next_uid()))

        self.population.individuals = next_inds

    def run(self):
        total_start = time.time()
        print("="*60)
        print("  GENETIC ALGORITHM — NAS SEARCH")
        print("="*60)

        for gen in range(self.num_gens):
            self.gen = gen
            gen_start = time.time()
            print("\n" + f"▶  Generation {gen+1}/{self.num_gens}")
            self._evaluate_population()
            self.history.record_generation(gen, self.population)
            stat = self.history.generations[-1]
            div  = diversity_score([i.chromosome for i in self.population.individuals])

            print(f"  ── Gen {gen+1} Summary ──")
            print(f"     Best  acc : {stat['best']:.4f}  Avg: {stat['avg']:.4f}  Diversity: {div:.3f}")

            if gen < self.num_gens - 1:
                self._breed_next_generation()

        best = self.history.generations[-1]["best"]
        print("\n" + "="*60)
        print(f"  🎉 GA Search Complete. Best Accuracy: {best:.4f}")
        print("="*60)
        self.history.save(f"{self.results_dir}/logs/ga_history.json")
        return self.history
