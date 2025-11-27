"""
PyTorch Lightning module for training models with flexible loss functions, train and validation data.

This module provides a flexible PyTorch Lightning module that can be used to train models
with different loss functions, train/validation splits, and other customizable parameters.
"""

from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ddib.loss import DDIB_Regularization
from ddib.models import forward_and_layer_outs


class IBModel(pl.LightningModule):
    """
    A flexible PyTorch Lightning module for training neural networks with configurable loss functions.

    Attributes:
        model: The neural network model to be trained
        loss_fn: The loss function to use during training
        val_loss_fn: The loss function to use during validation (defaults to loss_fn if None)
        optimizer: The optimizer to use for training
        learning_rate: Learning rate for the optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        layer_to_optimize: str,
        val_loss_fn: Optional[nn.Module] = None,
        optimizer_class: type = torch.optim.Adam,
        *,
        learning_rate: float = 1e-3,
        beta: float = 0.01,
        top_k: int = 10,
        **optimizer_kwargs
    ) -> None:
        """
        Initialize the FlexibleModel.

        Args:
            model: The neural network model to train
            loss_fn: The loss function to use during training
            layer_to_optimize: str
            val_loss_fn: Loss function for validation (uses loss_fn if None)
            optimizer_class: Class of optimizer to use (default: Adam)
            learning_rate: Learning rate for the optimizer
            **optimizer_kwargs: Additional arguments to pass to the optimizer
        """
        super().__init__()
        self.model = model
        self.beta = beta
        self.top_k = top_k
        self.layer_to_optimize = layer_to_optimize
        self.loss_fn = DDIB_Regularization(loss_fn, beta=beta, top_k=top_k)
        self.val_loss_fn = val_loss_fn if val_loss_fn is not None else loss_fn
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.save_hyperparameters(ignore=["model", "loss_fn", "val_loss_fn"])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:  # pylint: disable=arguments-differ
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output
        """
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> Dict[str, Any]:  # pylint: disable=arguments-differ
        """
        Training step for one batch.

        Args:
            batch: Batch of training data (x, y)
            batch_idx: Index of the batch

        Returns:
            Dict with 'loss' key containing the training loss
        """
        x, y = batch
        y_hat, outs = forward_and_layer_outs(self, x, [self.layer_to_optimize])
        loss = self.loss_fn(y_hat, y, x, outs[self.layer_to_optimize])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_scalar("Loss/Train", loss, self.global_step)
        return {"loss": loss}

    def validation_step(self, batch: tuple, batch_idx: int, *args, **kwargs) -> Dict[str, Any]:  # pylint: disable=arguments-differ
        """
        Validation step for one batch.

        Args:
            batch: Batch of validation data (x, y)
            batch_idx: Index of the batch

        Returns:
            Dict with 'val_loss' key containing the validation loss
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.val_loss_fn(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.logger and hasattr(self.logger, "experiment"):
            self.logger.experiment.add_scalar("Loss/Validation", val_loss, self.global_step)

        return {"val_loss": val_loss}

    def configure_optimizers(self) -> Any:
        """
        Configure the optimizer for training.

        Returns:
            Configured optimizer
        """
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        return optimizer


def prepare_dataloader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from input tensors.

    Args:
        X: Input features tensor
        y: Target labels tensor
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def train_model(
    model: pl.LightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_epochs: int = 10,
    accelerator: str = "auto",
    devices: Union[int, str] = "auto",
    log_dir: str = "tb_logs",
    experiment_name: str = "ddib_experiment",
    **trainer_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train a PyTorch Lightning model with train and validation data.

    Args:
        model: PyTorch Lightning model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        max_epochs: Maximum number of epochs to train
        accelerator: Accelerator to use for training ('auto', 'cpu', 'gpu', 'tpu', 'mps')
        devices: Devices to use for training
        log_dir: Directory to save tensorboard logs
        experiment_name: Name of the experiment for tensorboard
        **trainer_kwargs: Additional arguments to pass to the trainer

    Returns:
        Dictionary containing training results/metrics
    """
    tb_logger = TensorBoardLogger(log_dir, name=experiment_name)
    if "logger" not in trainer_kwargs:
        trainer_kwargs["logger"] = tb_logger
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator=accelerator, devices=devices, **trainer_kwargs
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    results = {
        "final_train_loss": trainer.logged_metrics.get(
            "train_loss_epoch", torch.tensor(float("inf"))
        ).item(),
        "final_val_loss": trainer.logged_metrics.get(
            "val_loss", torch.tensor(float("inf"))
        ).item(),
        "num_epochs": max_epochs,
    }
    return results
