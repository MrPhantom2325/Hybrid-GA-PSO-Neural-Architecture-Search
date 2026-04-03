
"""
utils/data_loader.py  —  DataLoader factory for MNIST and CIFAR-10
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms(dataset_name: str, augment: bool = False):
    """Return appropriate transforms for each dataset."""
    if dataset_name == "MNIST":
        train_t = transforms.Compose([
            transforms.RandomRotation(10) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        return train_t, test_t

    elif dataset_name == "CIFAR10":
        if augment:
            train_t = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ])
        else:
            train_t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                ),
            ])
        test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])
        return train_t, test_t
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataloaders(
    dataset_name : str,
    data_dir     : str,
    batch_size   : int  = 64,
    val_split    : float = 0.1,
    augment      : bool = False,
    num_workers  : int  = 2,
):
    """
    Returns (train_loader, val_loader, test_loader).
    Splits training set into train + val.
    """
    train_t, test_t = get_transforms(dataset_name, augment)

    if dataset_name == "MNIST":
        train_full = datasets.MNIST(data_dir, train=True,  download=True, transform=train_t)
        test_ds    = datasets.MNIST(data_dir, train=False, download=True, transform=test_t)
    elif dataset_name == "CIFAR10":
        train_full = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_t)
        test_ds    = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_t)

    val_size   = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
