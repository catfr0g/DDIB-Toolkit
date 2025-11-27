"""Module for loading various datasets including CIFAR-10."""
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import transforms


def load_cifar10_dataset(
    data_dir: Path,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
    train_val_split_ratio: float = 0.8,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the CIFAR-10 dataset with train/validation/test splits.

    Args:
        data_dir: Directory to store/load the dataset
        train_batch_size: Batch size for training data
        val_batch_size: Batch size for validation data
        test_batch_size: Batch size for test data
        train_val_split_ratio: Ratio to split training data into train/val
        num_workers: Number of workers for data loading
        download: Whether to download the dataset if not present

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Define transformations for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 means
            std=(0.2023, 0.1994, 0.2010)    # CIFAR-10 standard deviations
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 means
            std=(0.2023, 0.1994, 0.2010)    # CIFAR-10 standard deviations
        )
    ])

    # Load full training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_train
    )

    # Load validation dataset with test transforms (no augmentation)
    full_val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform_test
    )

    # Split dataset indices for train and validation
    dataset_size = len(full_train_dataset)
    train_size = int(train_val_split_ratio * dataset_size)
    val_size = dataset_size - train_size

    # Create random splits
    train_dataset, val_dataset = random_split(
        range(dataset_size), [train_size, val_size]
    )

    # Create subset datasets with appropriate transforms
    train_dataset = Subset(full_train_dataset, train_dataset.indices)
    val_dataset = Subset(full_val_dataset, val_dataset.indices)

    # Load test dataset
    full_test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        full_test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def load_cifar10_dataset_simple(
    data_dir: Path,
    train_batch_size: int = 32,
    test_batch_size: int = 32,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with train/test splits (no validation split).

    Args:
        data_dir: Directory to store/load the dataset
        train_batch_size: Batch size for training data
        test_batch_size: Batch size for test data
        num_workers: Number of workers for data loading
        download: Whether to download the dataset if not present

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10 means
            std=(0.2023, 0.1994, 0.2010)    # CIFAR-10 standard deviations
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader