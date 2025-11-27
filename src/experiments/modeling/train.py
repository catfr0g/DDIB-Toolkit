"""Training script for VGG/ResNet models with bottleneck width using DDIB."""
import os
from pathlib import Path
import random
from typing import Literal, Optional

from loguru import logger
import numpy as np
import torch
from torch import nn
import typer

from src.ddib.models import ResNetWithBottleneck, VGGWithBottleneck
from src.ddib.trainer import IBModel, train_model
from src.experiments.config import MODELS_DIR, RAW_DATA_DIR
from src.experiments.dataset_loading import load_cifar10_dataset

app = typer.Typer()


def seed_all(seed: int = 42):
    """Function to fix seed"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.multiprocessing.set_start_method("fork", force=True)


@app.command()
def main(
    model_arch: Literal["resnet18", "vgg11"] = "resnet18",
    bottleneck_width: Optional[int] = 128,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    alpha: float = 1.01,
    beta: float = 1.0,
    train_val_split_ratio: float = 0.8,
    save_model: bool = True,
    model_name: Optional[str] = None,
    log_dir: Path = MODELS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    num_workers: int = 4,
    download: bool = True,
    seed: int = 42
):
    """
    Train VGG/ResNet models with configurable bottleneck width using DDIB.

    Args:
        model_arch: Model architecture to use ('resnet18' or 'vgg11')
        bottleneck_width: Width of the bottleneck layer (None for no bottleneck)
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        test_batch_size: Batch size for testing
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        alpha: Alpha parameter for DDIB loss
        beta: Beta parameter for DDIB loss
        train_val_split_ratio: Ratio to split training data into train/val
        save_model: Whether to save the trained model
        model_name: Name for the saved model file
        log_dir: Directory to save model
        data_dir: Directory to load data from
        num_workers: Number of workers for data loading
        download: Whether to download dataset if not present
    """
    seed_all(seed)
    logger.info(f"Starting training with model: {model_arch}, bottleneck width: {bottleneck_width}")

    # Load CIFAR-10 dataset with train/validation/test splits
    logger.info("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = load_cifar10_dataset(
        data_dir=data_dir,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        train_val_split_ratio=train_val_split_ratio,
        num_workers=num_workers,
        download=download
    )
    logger.success("Dataset loaded successfully!")

    # Create model based on architecture
    if "resnet" in model_arch.lower():
        model: nn.Module = ResNetWithBottleneck(
            arch=model_arch,
            num_classes=10,  # CIFAR-10 has 10 classes
            bottleneck_width=bottleneck_width
        )
    elif "vgg" in model_arch.lower():
        model = VGGWithBottleneck(
            arch=model_arch,
            num_classes=10,  # CIFAR-10 has 10 classes
            bottleneck_width=bottleneck_width
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_arch}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Determine which layer to optimize based on bottleneck configuration
    if bottleneck_width is not None:
        # If bottleneck exists, optimize the bottleneck layer
        layer_to_optimize = "bottleneck"
    else:
        # If no bottleneck, optimize the final layer before classifier
        layer_to_optimize = "resnet" if "resnet" in model_arch.lower() else "vgg"

    # Create the DDIB model wrapper
    ddib_model = IBModel(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),  # Base loss function
        layer_to_optimize=layer_to_optimize,
        beta=beta,  # Beta parameter for DDIB regularization
        learning_rate=learning_rate,
        optimizer_class=torch.optim.Adam,
        **{"weight_decay": weight_decay}
    )

    # Train the model using the DDIB trainer
    logger.info("Starting model training...")
    train_model(
        model=ddib_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else "auto",
        log_dir="tb_logs",
        experiment_name=f"{model_arch}_bottleneck_{bottleneck_width}",
    )

    # Get the trained underlying model
    trained_model = ddib_model.model

    logger.success("Training completed successfully!")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    trained_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")

    # Save the model if requested
    if save_model:
        if model_name is None:
            model_name = f"{model_arch}_bottleneck_{bottleneck_width}_epochs_{num_epochs}.pt"

        model_path = log_dir / model_name
        torch.save(trained_model.state_dict(), model_path)
        logger.success(f"Model saved to {model_path}")

    logger.info("Training script completed!")


if __name__ == "__main__":
    app()
