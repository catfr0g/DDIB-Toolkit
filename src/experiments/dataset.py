"""Script to load dataset"""
from pathlib import Path

from loguru import logger
import typer

from experiments.config import RAW_DATA_DIR

from .dataset_loading import load_cifar10_dataset

app = typer.Typer()


@app.command()
def main(
    data_dir: Path = RAW_DATA_DIR,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
    train_val_split_ratio: float = 0.8,
    num_workers: int = 4,
    download: bool = True,
):
    """Load CIFAR-10 dataset with train/validation/test splits."""
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

    logger.success("CIFAR-10 dataset loaded successfully!")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")


if __name__ == "__main__":
    app()
