"""Tests for CIFAR-10 dataset loading functionality."""
import tempfile
from pathlib import Path

import torch

from src.experiments.dataset_loading import load_cifar10_dataset, load_cifar10_dataset_simple


def test_load_cifar10_dataset():
    """Test loading CIFAR-10 dataset with train/validation/test splits."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        train_loader, val_loader, test_loader = load_cifar10_dataset(
            data_dir=data_dir,
            train_batch_size=32,
            val_batch_size=32,
            test_batch_size=32,
            download=True
        )
        
        # Check that loaders are created
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        
        # Get a batch from train loader to verify it works
        train_batch = next(iter(train_loader))
        train_images, train_labels = train_batch
        
        assert len(train_images) == 32  # batch size
        assert len(train_labels) == 32  # batch size
        assert train_images.shape[1:] == (3, 32, 32)  # CIFAR-10 image dimensions
        
        # Verify tensor types
        assert isinstance(train_images, torch.Tensor)
        assert isinstance(train_labels, torch.Tensor)
        
        # Check that labels are in the expected range (0-9 for CIFAR-10)
        assert train_labels.min() >= 0
        assert train_labels.max() <= 9


def test_load_cifar10_dataset_simple():
    """Test loading CIFAR-10 dataset with train/test splits only."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        train_loader, test_loader = load_cifar10_dataset_simple(
            data_dir=data_dir,
            train_batch_size=32,
            test_batch_size=32,
            download=True
        )
        
        # Check that loaders are created
        assert train_loader is not None
        assert test_loader is not None
        
        # Get a batch from train loader to verify it works
        train_batch = next(iter(train_loader))
        train_images, train_labels = train_batch
        
        assert len(train_images) == 32  # batch size
        assert len(train_labels) == 32  # batch size
        assert train_images.shape[1:] == (3, 32, 32)  # CIFAR-10 image dimensions
        
        # Verify tensor types
        assert isinstance(train_images, torch.Tensor)
        assert isinstance(train_labels, torch.Tensor)
        
        # Check that labels are in the expected range (0-9 for CIFAR-10)
        assert train_labels.min() >= 0
        assert train_labels.max() <= 9