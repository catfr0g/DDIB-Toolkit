"""
PyTorch Lightning module for training models with flexible loss
functions, train and validation data.

This module provides a flexible PyTorch Lightning module that can
be used to train models with different loss functions, train/validation
splits, and other customizable parameters.
"""

from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ddib.loss import DDIB_Regularization, calculate_kernel_width, calculate_MI, rbf_kernel
from ddib.models import forward_and_layer_outs


class EpochProgressCallback(Callback):
	"""Callback that logs epoch progress instead of using progress bars."""

	def __init__(self, experiment_id: Optional[str] = None):
		"""
		Initialize the callback.

		Args:
		    experiment_id: Optional identifier to prefix log messages
		"""
		self.experiment_id = experiment_id
		self._setup_logger()

	def _setup_logger(self):
		"""Setup loguru logger with optional experiment prefix."""
		import sys

		from loguru import logger

		self.logger = logger
		self.logger.remove()

		if self.experiment_id:
			format_str = (
				'<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
				f'<level>{{level}}</level> | <cyan>[{self.experiment_id}]</cyan> | '
				'<level>{message}</level>'
			)
		else:
			format_str = (
				'<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
				'<level>{level}</level> | <level>{message}</level>'
			)

		self.logger.add(sys.stderr, format=format_str, level='INFO')

	def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
		"""Log at the start of each epoch."""
		current_epoch = trainer.current_epoch
		total_epochs = trainer.max_epochs
		self.logger.info(f'Epoch {current_epoch + 1}/{total_epochs} - Training started')

	def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
		"""Log metrics at the end of each epoch."""
		current_epoch = trainer.current_epoch
		total_epochs = trainer.max_epochs

		# Get metrics from logged values
		metrics = {}
		for key, value in trainer.callback_metrics.items():
			if hasattr(value, 'item'):
				metrics[key] = value.item()

		# Format metrics for logging
		metric_str = ' | '.join(
			f'{k}: {v:.4f}' for k, v in metrics.items() if not k.startswith('val_')
		)
		if metric_str:
			self.logger.info(
				f'Epoch {current_epoch + 1}/{total_epochs} - Training completed | {metric_str}'
			)

	def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
		"""Log validation metrics at the end of each validation epoch."""
		current_epoch = trainer.current_epoch
		total_epochs = trainer.max_epochs

		# Get validation metrics
		metrics = {}
		for key, value in trainer.callback_metrics.items():
			if hasattr(value, 'item') and key.startswith('val_'):
				metrics[key] = value.item()

		if metrics:
			metric_str = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
			self.logger.info(
				f'Epoch {current_epoch + 1}/{total_epochs} - Validation | {metric_str}'
			)


class IBModel(pl.LightningModule):
	"""
	A flexible PyTorch Lightning module for training neural
	networks with configurable loss functions.

	Attributes:
	    model: The neural network model to be trained
	    loss_fn: The loss function to use during training
	    val_loss_fn: The loss function to use during validation
	                 (defaults to loss_fn if None)
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
		use_scheduler: bool = True,
		scheduler_type: str = 'reduceonplateau',
		# Options: "reduceonplateau", "step", "cosine"
		**optimizer_kwargs,
	) -> None:
		"""
		Initialize the FlexibleModel.

		Args:
		    model: The neural network model to train
		    loss_fn: The loss function to use during training
		    layer_to_optimize: str
		    val_loss_fn: Loss function for validation
		                 (uses loss_fn if None)
		    optimizer_class: Class of optimizer to use (default: Adam)
		    learning_rate: Learning rate for the optimizer
		    use_scheduler: Whether to use a learning rate scheduler
		    scheduler_type: Type of scheduler to use
		                    ("reduceonplateau", "step", "cosine")
		    **optimizer_kwargs: Additional arguments to pass to
		                        the optimizer
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
		self.use_scheduler = use_scheduler
		self.scheduler_type = scheduler_type
		self.optimizer_kwargs = optimizer_kwargs
		self.save_hyperparameters(ignore=['model', 'loss_fn', 'val_loss_fn'])

	def forward(  # pylint: disable=arguments-differ
		self, x: torch.Tensor, *args, **kwargs
	) -> Any:
		"""
		Forward pass through the model.

		Args:
		    x: Input tensor

		Returns:
		    Model output
		"""
		return self.model(x)

	def training_step(  # pylint: disable=arguments-differ
		self, batch: tuple, batch_idx: int, *args, **kwargs
	) -> Dict[str, Any]:
		"""
		Training step for one batch.

		Args:
		    batch: Batch of training data (x, y)
		    batch_idx: Index of the batch

		Returns:
		    Dict with 'loss' key containing the training loss
		"""
		x, y = batch
		# Ensure inputs are on the correct device
		device = self.device
		x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

		y_hat, outs = forward_and_layer_outs(self.model, x, [self.layer_to_optimize])
		loss = self.loss_fn(y_hat, y, x, outs[self.layer_to_optimize])

		# Calculate classification accuracy
		acc = (y_hat.argmax(dim=1) == y).float().mean()

		# Calculate empirical compression (mutual info) for logging.
		# Wrap in no_grad() to avoid creating a separate graph that
		# would conflict with loss.backward()
		with torch.no_grad():
			z = outs[self.layer_to_optimize]
			if x.dim() > 2:
				x_flat = x.view(x.size(0), -1)
			else:
				x_flat = x
			if z.dim() > 2:
				z_flat = z.view(z.size(0), -1)
			else:
				z_flat = z

			x_sigma = calculate_kernel_width(x_flat, top_k=self.top_k)
			z_sigma = calculate_kernel_width(z_flat, top_k=self.top_k)
			x_gram = rbf_kernel(x_flat, x_flat, sigma=x_sigma)
			z_gram = rbf_kernel(z_flat, z_flat, sigma=z_sigma)
			mutual_info = calculate_MI(x_gram, z_gram)

		# Calculate effective capacity utilization (ratio of original
		# loss to log2(W)) where W is the number of parameters
		total_params = sum(p.numel() for p in self.parameters())
		log_w = torch.log2(torch.tensor(float(total_params)))
		effective_capacity_utilization = loss.detach() / log_w if log_w > 0 else torch.tensor(0.0)

		# Log metrics
		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
		self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=False)
		self.log(
			'train_empirical_compression',
			mutual_info,
			on_step=True,
			on_epoch=True,
		)
		self.log(
			'train_effective_capacity_utilization',
			effective_capacity_utilization,
			on_step=True,
			on_epoch=True,
		)

		# Log to TensorBoard if available
		if self.logger is not None:
			experiment = getattr(self.logger, 'experiment', None)
			if experiment is not None and hasattr(experiment, 'add_scalar'):
				experiment.add_scalar('Loss/Train', loss, self.global_step)
				experiment.add_scalar('Accuracy/Train', acc, self.global_step)
				experiment.add_scalar(
					'Metrics/TrainEmpiricalCompression',
					mutual_info,
					self.global_step,
				)
				experiment.add_scalar(
					'Metrics/TrainEffectiveCapacityUtilization',
					effective_capacity_utilization,
					self.global_step,
				)

		return {
			'loss': loss,
			'train_acc': acc,
			'train_empirical_compression': mutual_info,
			'train_effective_capacity_utilization': effective_capacity_utilization,
		}

	def validation_step(  # pylint: disable=arguments-differ
		self, batch: tuple, batch_idx: int, *args, **kwargs
	) -> Dict[str, Any]:
		"""
		Validation step for one batch.

		Args:
		    batch: Batch of validation data (x, y)
		    batch_idx: Index of the batch

		Returns:
		    Dict with 'val_loss' key containing the validation loss
		"""
		x, y = batch
		# Ensure inputs are on the correct device
		device = self.device
		x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

		# Get predictions and layer outputs for metrics calculation
		y_hat, outs = forward_and_layer_outs(self.model, x, [self.layer_to_optimize])
		val_loss = self.val_loss_fn(y_hat, y)

		# Calculate classification accuracy
		acc = (y_hat.argmax(dim=1) == y).float().mean()

		# Calculate empirical compression (mutual info) for logging.
		# Wrap in no_grad() to avoid graph conflicts.
		with torch.no_grad():
			z = outs[self.layer_to_optimize]
			x_flat = x if x.dim() <= 2 else x.view(x.size(0), -1)
			z_flat = z if z.dim() <= 2 else z.view(z.size(0), -1)
			x_gram = rbf_kernel(
				x_flat, x_flat, sigma=calculate_kernel_width(x_flat, top_k=self.top_k)
			)
			z_gram = rbf_kernel(
				z_flat, z_flat, sigma=calculate_kernel_width(z_flat, top_k=self.top_k)
			)
			mutual_info = calculate_MI(x_gram, z_gram)

		# Calculate effective capacity utilization (ratio of original
		# loss to log2(W)) where W is the number of parameters
		total_params = sum(p.numel() for p in self.parameters())
		log_w = torch.log2(torch.tensor(float(total_params)))
		effective_capacity_utilization = (
			val_loss.detach() / log_w if log_w > 0 else torch.tensor(0.0)
		)

		# Log metrics
		self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
		self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
		self.log('empirical_compression', mutual_info, on_step=False, on_epoch=True)
		self.log(
			'effective_capacity_utilization',
			effective_capacity_utilization,
			on_step=False,
			on_epoch=True,
		)

		# Log to TensorBoard if available
		if self.logger is not None:
			experiment = getattr(self.logger, 'experiment', None)
			if experiment is not None and hasattr(experiment, 'add_scalar'):
				experiment.add_scalar('Loss/Validation', val_loss, self.global_step)
				experiment.add_scalar('Accuracy/Validation', acc, self.global_step)
				experiment.add_scalar(
					'Metrics/EmpiricalCompression',
					mutual_info,
					self.global_step,
				)
				experiment.add_scalar(
					'Metrics/EffectiveCapacityUtilization',
					effective_capacity_utilization,
					self.global_step,
				)

		return {
			'val_loss': val_loss,
			'val_acc': acc,
			'empirical_compression': mutual_info,
			'effective_capacity_utilization': effective_capacity_utilization,
		}

	def configure_optimizers(self) -> Any:
		"""
		Configure the optimizer for training.

		Returns:
		    Configured optimizer and scheduler
		"""
		optimizer = self.optimizer_class(
			self.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
		)

		if self.use_scheduler:
			if self.scheduler_type == 'reduceonplateau':
				scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
					optimizer,
					mode='min',  # Reduce LR when loss stops improving
					factor=0.5,  # Reduce LR by half
					patience=10,  # Wait 10 epochs before reducing
				)

				return {
					'optimizer': optimizer,
					'lr_scheduler': {
						'scheduler': scheduler,
						'monitor': 'val_loss',  # Monitor validation loss
						'interval': 'epoch',
						'frequency': 1,
					},
				}
			elif self.scheduler_type == 'step':
				scheduler = torch.optim.lr_scheduler.StepLR(
					optimizer,
					step_size=30,  # Reduce LR every 30 epochs
					gamma=0.1,  # Multiply LR by 0.1
				)

				return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'interval': 'epoch'}
			elif self.scheduler_type == 'cosine':
				scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
					optimizer,
					T_max=200,  # Maximum number of epochs
					eta_min=1e-6,  # Minimum learning rate
				)

				return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'interval': 'epoch'}

		# If no scheduler is used, return just the optimizer
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
	accelerator: str = 'auto',
	devices: Union[int, str] = 'auto',
	log_dir: str = 'tb_logs',
	experiment_name: str = 'ddib_experiment',
	precision: str = '32-true',  # Default to 32-bit precision
	log_every_n_steps: Optional[int] = None,
	enable_progress_bar: bool = False,
	experiment_id: Optional[str] = None,
	hyperparams: Optional[Dict[str, Any]] = None,
	**trainer_kwargs: Any,
) -> Dict[str, Any]:
	"""
	Train a PyTorch Lightning model with train and validation data.

	Args:
	    model: PyTorch Lightning model to train
	    train_dataloader: DataLoader for training data
	    val_dataloader: DataLoader for validation data
	    max_epochs: Maximum number of epochs to train
	    accelerator: Accelerator to use for training
	                 ('auto', 'cpu', 'gpu', 'tpu', 'mps')
	    devices: Devices to use for training
	    log_dir: Directory to save tensorboard logs
	    experiment_name: Name of the experiment for tensorboard
	    precision: Precision for training ('32-true', '16-mixed', etc.)
	    log_every_n_steps: Log every n steps
	                       (defaults to 1 epoch length if None)
	    enable_progress_bar: Whether to enable progress bars
	                         (default: False)
	    experiment_id: Optional identifier for logging
	                   (used in callback)
	    hyperparams: Optional dict of hyperparameters to log
	                 to TensorBoard
	    **trainer_kwargs: Additional arguments to pass to the trainer

	Returns:
	    Dictionary containing training results/metrics
	"""
	tb_logger = TensorBoardLogger(log_dir, name=experiment_name)
	if 'logger' not in trainer_kwargs:
		trainer_kwargs['logger'] = tb_logger
	if 'precision' not in trainer_kwargs:
		trainer_kwargs['precision'] = precision
	if log_every_n_steps is None:
		num_train_batches = len(train_dataloader)
		log_every_n_steps = max(1, num_train_batches)
	if 'callbacks' not in trainer_kwargs:
		trainer_kwargs['callbacks'] = []
	if not enable_progress_bar:
		trainer_kwargs['callbacks'].append(EpochProgressCallback(experiment_id=experiment_id))
	trainer = pl.Trainer(
		max_epochs=max_epochs,
		accelerator=accelerator,
		devices=devices,
		log_every_n_steps=log_every_n_steps,
		enable_progress_bar=enable_progress_bar,
		**trainer_kwargs,
	)
	trainer.fit(
		model,
		train_dataloaders=train_dataloader,
		val_dataloaders=val_dataloader,
	)
	logged_metrics = trainer.logged_metrics
	results = {
		'final_train_loss': logged_metrics.get(
			'train_loss_epoch', torch.tensor(float('inf'))
		).item(),
		'final_val_loss': logged_metrics.get('val_loss', torch.tensor(float('inf'))).item(),
		'final_train_acc': logged_metrics.get('train_acc', torch.tensor(0.0)).item(),
		'final_val_acc': logged_metrics.get('val_acc', torch.tensor(0.0)).item(),
		'final_empirical_compression': logged_metrics.get(
			'empirical_compression', torch.tensor(0.0)
		).item(),
		'final_train_empirical_compression': logged_metrics.get(
			'train_empirical_compression', torch.tensor(0.0)
		).item(),
		'final_effective_capacity_utilization': logged_metrics.get(
			'effective_capacity_utilization', torch.tensor(0.0)
		).item(),
		'final_train_effective_capacity_utilization': logged_metrics.get(
			'train_effective_capacity_utilization', torch.tensor(0.0)
		).item(),
		'num_epochs': max_epochs,
	}
	if hyperparams and hasattr(tb_logger, 'log_hyperparams'):
		tb_logger.log_hyperparams(hyperparams, metrics=results)
	return results
