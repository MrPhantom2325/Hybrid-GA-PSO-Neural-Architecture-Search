
"""
pso/pso_fitness.py
Fitness function for PSO — wraps proxy_train.
Takes a raw PSO position vector, decodes to hyperparams,
trains the fixed architecture, returns val_accuracy.
"""
import torch
from torch.utils.data import DataLoader
from pso.pso_optimizer import decode_particle
from search_space.chromosome import decode_chromosome
from training.proxy_trainer import proxy_train


class PSOFitnessEvaluator:
    """
    Callable fitness evaluator.
    Keeps the chromosome and dataset fixed; PSO varies hyperparams.

    We rebuild the DataLoader inside each call when batch_size changes,
    because DataLoader batch_size is set at construction time.
    """
    def __init__(
        self,
        chromosome,
        train_dataset,
        val_loader,
        device,
        proxy_epochs  = 3,
        in_channels   = 1,
        image_size    = 28,
        num_classes   = 10,
    ):
        self.chromosome    = chromosome
        self.train_dataset = train_dataset   # full train Dataset (not loader)
        self.val_loader    = val_loader
        self.device        = device
        self.proxy_epochs  = proxy_epochs
        self.in_channels   = in_channels
        self.image_size    = image_size
        self.num_classes   = num_classes
        self.n_calls       = 0

    def __call__(self, position):
        """
        Decode position → hyperparams → train → return val_accuracy.
        Called once per particle per iteration.
        """
        self.n_calls += 1
        hp = decode_particle(position)
        lr         = hp["lr"]
        dropout    = hp["dropout"]
        batch_size = hp["batch_size"]

        # Build loader with PSO-selected batch size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = 2,
            pin_memory  = True,
        )

        # Fresh model for each evaluation
        model = decode_chromosome(
            self.chromosome,
            in_channels = self.in_channels,
            image_size  = self.image_size,
            num_classes = self.num_classes,
        )

        result = proxy_train(
            model,
            train_loader,
            self.val_loader,
            self.device,
            epochs       = self.proxy_epochs,
            lr           = lr,
            dropout_rate = dropout,
        )
        return result["val_accuracy"]
