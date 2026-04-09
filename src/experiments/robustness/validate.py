"""
Robustness validation script for DDIB models.

This script evaluates trained models on:
1. CIFAR-10-C: 15 corruption types at 5 severity levels
2. Adversarial examples: PGD attack with ε=8/255, α=2/255, 10 iterations

The script supports two modes:
- Single model evaluation: load an existing model checkpoint
- Train and evaluate: train a model from scratch using config and evaluate

Usage:
    # Evaluate existing model
    python -m src.experiments.robustness.validate \
        --model-path models/best_model.pt \
        --config config/robustness_config.yaml \
        --data-dir data/processed \
        --output-dir results/robustness

    # Train and evaluate all models from config
    python -m src.experiments.robustness.validate \
        --config config/robustness_config.yaml \
        --data-dir data/processed \
        --output-dir results/robustness
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import typer
import yaml

from src.ddib.models import EfficientNetWithBottleneck, ResNetWithBottleneck, VGGWithBottleneck
from src.experiments.dataset_loading import load_cifar10_dataset
from src.experiments.robustness.imagenet_c import (
	CORRUPTION_TYPES,
	SEVERITY_LEVELS,
	ImageNetCCIFAR10Dataset,
	create_cifar10_c_dataloader,
)
from src.experiments.robustness.metrics import (
	calculate_robustness_metrics,
	compare_robustness,
	print_robustness_report,
	save_metrics_to_json,
)
from src.experiments.robustness.pgd_attack import PGDAttack, evaluate_adversarial_robustness

app = typer.Typer()


def load_config(config_path: Path) -> Dict[str, Any]:
	"""
	Load configuration from YAML file.

	Args:
	    config_path: Path to YAML configuration file

	Returns:
	    Configuration dictionary
	"""
	if not config_path.exists():
		raise FileNotFoundError(f'Configuration file not found: {config_path}')

	with open(config_path, 'r', encoding='utf-8') as f:
		config = yaml.safe_load(f)

	logger.info(f'Loaded configuration from {config_path}')
	return config


def get_training_params(model_config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Merge model-specific training parameters with defaults.

	Args:
	    model_config: Model-specific configuration
	    defaults: Default training parameters

	Returns:
	    Merged training parameters
	"""
	params = defaults.copy()
	if 'training' in model_config:
		params.update(model_config['training'])
	return params


def load_model(
	model_arch: str,
	bottleneck_width: Optional[int],
	model_path: Path,
	device: torch.device,
) -> nn.Module:
	"""
	Load a trained model from checkpoint.

	Args:
	    model_arch: Model architecture name
	    bottleneck_width: Width of bottleneck layer
	    model_path: Path to model checkpoint
	    device: Device to load model on

	Returns:
	    Loaded model with weights
	"""
	logger.info(f'Loading model: {model_arch} with bottleneck_width={bottleneck_width}')

	# Create model
	if 'resnet' in model_arch.lower():
		model = ResNetWithBottleneck(
			arch=model_arch,
			num_classes=10,
			bottleneck_width=bottleneck_width,
		)
	elif 'vgg' in model_arch.lower():
		model = VGGWithBottleneck(
			arch=model_arch,
			num_classes=10,
			bottleneck_width=bottleneck_width,
		)
	elif 'efficientnet' in model_arch.lower():
		model = EfficientNetWithBottleneck(
			arch=model_arch,
			num_classes=10,
			bottleneck_width=bottleneck_width,
		)
	else:
		raise ValueError(f'Unsupported model architecture: {model_arch}')

	# Load weights
	if model_path.exists():
		state_dict = torch.load(model_path, map_location=device, weights_only=True)
		model.load_state_dict(state_dict)
		logger.success(f'Model loaded from {model_path}')
	else:
		logger.warning(f'Model path does not exist: {model_path}')

	model = model.to(device)
	model.eval()

	return model


def train_model_from_config(
	model_config: Dict[str, Any],
	training_defaults: Dict[str, Any],
	data_dir: Path,
	device: torch.device,
) -> Tuple[nn.Module, DataLoader, str]:
	"""
	Train a model from scratch using configuration and return it with test loader.

	Args:
	    model_config: Model configuration from YAML
	    training_defaults: Default training parameters
	    data_dir: Directory containing data
	    device: Device for training

	Returns:
	    Tuple of (trained model, test loader, model name)
	"""
	import os
	import random

	from src.ddib.trainer import IBModel, train_model
	from src.experiments.config import MODELS_DIR

	model_name = model_config['name']
	model_arch = model_config['model_arch']
	bottleneck_width = model_config.get('bottleneck_width')
	beta = model_config.get('beta', 1.0)
	seed = model_config.get('seed', 42)

	# Get training parameters
	training_params = get_training_params(model_config, training_defaults)
	num_epochs = training_params.get('num_epochs', 100)

	# Set seeds
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	logger.info(f'Training model: {model_name}')
	logger.info(f'  Architecture: {model_arch}')
	logger.info(f'  Bottleneck width: {bottleneck_width}')
	logger.info(f'  Beta: {beta}')
	logger.info(f'  Seed: {seed}')
	logger.info(f'  Epochs: {num_epochs}')

	# Load data
	logger.info('Loading CIFAR-10 dataset...')
	train_loader, val_loader, test_loader = load_cifar10_dataset(
		data_dir=data_dir,
		train_batch_size=training_params.get('train_batch_size', 128),
		val_batch_size=training_params.get('val_batch_size', 128),
		test_batch_size=training_params.get('test_batch_size', 128),
		train_val_split_ratio=training_params.get('train_val_split_ratio', 0.8),
		num_workers=training_params.get('num_workers', 4),
		download=True,
	)

	# Create model
	if 'resnet' in model_arch.lower():
		base_model = ResNetWithBottleneck(
			arch=model_arch,
			num_classes=10,
			bottleneck_width=bottleneck_width,
		)
		layer_to_optimize = 'bottleneck' if bottleneck_width else 'resnet'
	elif 'vgg' in model_arch.lower():
		base_model = VGGWithBottleneck(
			arch=model_arch,
			num_classes=10,
			bottleneck_width=bottleneck_width,
		)
		layer_to_optimize = 'bottleneck' if bottleneck_width else 'vgg'
	elif 'efficientnet' in model_arch.lower():
		base_model = EfficientNetWithBottleneck(
			arch=model_arch,
			num_classes=10,
			bottleneck_width=bottleneck_width,
		)
		layer_to_optimize = 'bottleneck' if bottleneck_width else 'efficient_net'
	else:
		raise ValueError(f'Unsupported architecture: {model_arch}')

	# Create DDIB model
	ddib_model = IBModel(
		model=base_model,
		loss_fn=nn.CrossEntropyLoss(),
		layer_to_optimize=layer_to_optimize,
		beta=beta,
		learning_rate=training_params.get('learning_rate', 1e-3),
		optimizer_class=torch.optim.Adam,
		weight_decay=training_params.get('weight_decay', 1e-4),
	)

	# Train
	logger.info('Starting training...')
	train_model(
		model=ddib_model,
		train_dataloader=train_loader,
		val_dataloader=val_loader,
		max_epochs=num_epochs,
		accelerator='auto',
		devices=1,
		log_dir='tb_logs',
		experiment_name=f'robustness_{model_name}',
		experiment_id=model_name,
	)

	# Save model
	model_path = MODELS_DIR / f'{model_name}_epochs_{num_epochs}.pt'
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	torch.save(ddib_model.model.state_dict(), model_path)
	logger.success(f'Model saved to {model_path}')

	return ddib_model.model, test_loader, model_name


def evaluate_on_corruptions(
	model: nn.Module,
	data_dir: Path,
	corruption_types: Optional[List[str]] = None,
	severity_levels: Optional[List[int]] = None,
	batch_size: int = 64,
	num_workers: int = 4,
	device: Optional[torch.device] = None,
) -> Dict[str, Dict[int, float]]:
	"""
	Evaluate model on CIFAR-10-C corruptions.

	Args:
	    model: Model to evaluate
	    data_dir: Directory containing CIFAR-10-C data
	    corruption_types: List of corruption types to evaluate
	    severity_levels: List of severity levels to evaluate
	    batch_size: Batch size for evaluation
	    num_workers: Number of data loading workers
	    device: Device for evaluation

	Returns:
	    Dict mapping corruption_type -> severity -> accuracy
	"""
	if device is None:
		device = next(model.parameters()).device

	corruption_types = corruption_types or CORRUPTION_TYPES
	severity_levels = severity_levels or SEVERITY_LEVELS

	results: Dict[str, Dict[int, float]] = {}

	logger.info(
		f'Evaluating on {len(corruption_types)} corruption types × {len(severity_levels)} severity levels'
	)

	for corruption_type in corruption_types:
		results[corruption_type] = {}
		logger.info(f' Evaluating corruption: {corruption_type}')

		for severity in severity_levels:
			# Create dataloader for this corruption/severity
			try:
				dataloader = create_cifar10_c_dataloader(
					data_dir=data_dir / 'CIFAR-10-C',
					corruption_types=[corruption_type],
					severity_levels=[severity],
					batch_size=batch_size,
					num_workers=num_workers,
				)

				# Evaluate
				correct = 0
				total = 0

				with torch.no_grad():
					for images, labels, _, _ in dataloader:
						images = images.to(device)
						labels = labels.to(device)

						outputs = model(images)
						preds = outputs.argmax(dim=1)

						correct += (preds == labels).sum().item()
						total += labels.size(0)

				accuracy = correct / total if total > 0 else 0.0
				results[corruption_type][severity] = accuracy

				logger.debug(
					f'  Severity {severity}: Accuracy = {accuracy:.4f} ({correct}/{total})'
				)

			except Exception as e:
				logger.warning(f'  Failed to evaluate severity {severity}: {e}')
				results[corruption_type][severity] = 0.0

	return results


def evaluate_on_pgd(
	model: nn.Module,
	test_loader: DataLoader,
	epsilon: float = 8 / 255,
	alpha: float = 2 / 255,
	iterations: int = 10,
	max_samples: int = 1000,
	device: Optional[torch.device] = None,
) -> Tuple[float, Dict[str, float]]:
	"""
	Evaluate model on PGD adversarial examples.

	Args:
	    model: Model to evaluate
	    test_loader: DataLoader with clean test data
	    epsilon: PGD epsilon parameter
	    alpha: PGD alpha parameter
	    iterations: Number of PGD iterations
	    max_samples: Maximum number of samples to evaluate
	    device: Device for evaluation

	Returns:
	    Tuple of (adversarial_accuracy, metrics_dict)
	"""
	if device is None:
		device = next(model.parameters()).device

	# Collect a subset of test data for adversarial evaluation
	# (adversarial example generation is computationally expensive)
	all_images = []
	all_labels = []

	for images, labels in test_loader:
		all_images.append(images)
		all_labels.append(labels)

		if sum(t.size(0) for t in all_images) >= max_samples:
			break

	images = torch.cat(all_images, dim=0)[:max_samples]
	labels = torch.cat(all_labels, dim=0)[:max_samples]

	logger.info(f'Evaluating PGD attack on {len(images)} samples')

	# Create attack and evaluate
	attack = PGDAttack(
		model=model,
		epsilon=epsilon,
		alpha=alpha,
		iterations=iterations,
	)

	adv_accuracy, metrics = attack.attack_accuracy(
		images=images,
		labels=labels,
		batch_size=32,
		device=device,
	)

	return adv_accuracy, metrics


def evaluate_clean_accuracy(
	model: nn.Module,
	test_loader: DataLoader,
	device: Optional[torch.device] = None,
) -> float:
	"""
	Evaluate model on clean test data.

	Args:
	    model: Model to evaluate
	    test_loader: DataLoader with clean test data
	    device: Device for evaluation

	Returns:
	    Clean test accuracy
	"""
	if device is None:
		device = next(model.parameters()).device

	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in test_loader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			preds = outputs.argmax(dim=1)

			correct += (preds == labels).sum().item()
			total += labels.size(0)

	accuracy = correct / total if total > 0 else 0.0
	logger.info(f'Clean test accuracy: {accuracy:.4f} ({correct}/{total})')

	return accuracy


def evaluate_model_robustness(
	model: nn.Module,
	model_name: str,
	data_dir: Path,
	output_dir: Path,
	test_loader: Optional[DataLoader] = None,
	eval_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	"""
	Comprehensive robustness evaluation for a single model.

	Args:
	    model: Model to evaluate
	    model_name: Name identifier for the model
	    data_dir: Data directory
	    output_dir: Output directory for results
	    test_loader: Optional test loader for clean/PGD evaluation
	    eval_config: Evaluation configuration

	Returns:
	    Dictionary with all robustness metrics
	"""
	if eval_config is None:
		eval_config = {}

	device = next(model.parameters()).device

	results = {
		'model_name': model_name,
		'timestamp': torch.__version__,
	}

	# Get evaluation settings
	corruption_types = eval_config.get('corruption_types', CORRUPTION_TYPES)
	severity_levels = eval_config.get('severity_levels', SEVERITY_LEVELS)
	skip_corruptions = len(corruption_types) == 0
	skip_pgd = eval_config.get('skip_pgd', False)

	# 1. Evaluate on clean data
	if test_loader is not None:
		clean_acc = evaluate_clean_accuracy(model, test_loader, device)
		results['clean_accuracy'] = clean_acc
	else:
		results['clean_accuracy'] = 0.0

	# 2. Evaluate on CIFAR-10-C corruptions
	if not skip_corruptions:
		corruption_results = evaluate_on_corruptions(
			model=model,
			data_dir=data_dir,
			corruption_types=corruption_types,
			severity_levels=severity_levels,
			batch_size=eval_config.get('eval_batch_size', 64),
			num_workers=eval_config.get('eval_num_workers', 4),
			device=device,
		)
		results['corruption_accuracies'] = corruption_results
	else:
		results['corruption_accuracies'] = {}
		logger.info('Skipping corruption evaluation')

	# 3. Evaluate on PGD adversarial examples
	if test_loader is not None and not skip_pgd:
		adv_acc, pgd_metrics = evaluate_on_pgd(
			model=model,
			test_loader=test_loader,
			epsilon=eval_config.get('pgd_epsilon', 8 / 255),
			alpha=eval_config.get('pgd_alpha', 2 / 255),
			iterations=eval_config.get('pgd_iterations', 10),
			max_samples=eval_config.get('pgd_max_samples', 1000),
			device=device,
		)
		results['adversarial_accuracy'] = adv_acc
		results.update(pgd_metrics)
	else:
		results['adversarial_accuracy'] = 0.0
		if skip_pgd:
			logger.info('Skipping PGD adversarial evaluation')

	# 4. Calculate comprehensive metrics
	adversarial_accuracies = [results.get('adversarial_accuracy', 0.0)]

	robustness_metrics = calculate_robustness_metrics(
		corruption_accuracies=corruption_results if not skip_corruptions else {},
		clean_accuracy=results.get('clean_accuracy'),
		adversarial_accuracies=adversarial_accuracies,
	)

	results.update(robustness_metrics)

	# 5. Save results
	model_output_dir = output_dir / model_name
	model_output_dir.mkdir(parents=True, exist_ok=True)

	# Save full results
	results_path = model_output_dir / 'robustness_results.json'
	save_metrics_to_json(results, str(results_path))

	# Save summary metrics
	summary_metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
	summary_path = model_output_dir / 'robustness_summary.json'
	save_metrics_to_json(summary_metrics, str(summary_path))

	# Print report
	print_robustness_report(robustness_metrics)

	return results


@app.command()
def main(
	config: Path = typer.Option(
		Path('config/robustness_config.yaml'),
		'--config',
		'-c',
		help='Path to YAML configuration file',
	),
	model_path: Optional[Path] = typer.Option(
		None,
		'--model-path',
		'-m',
		help='Path to trained model checkpoint (for single model evaluation)',
	),
	model_arch: Optional[str] = typer.Option(
		None,
		'--model-arch',
		'-a',
		help='Model architecture (required with --model-path if not in config)',
	),
	bottleneck_width: Optional[int] = typer.Option(
		None,
		'--bottleneck-width',
		'-b',
		help='Bottleneck width (required with --model-path if not in config)',
	),
	data_dir: Path = typer.Option(
		Path('data/processed'),
		'--data-dir',
		'-d',
		help='Directory containing data',
	),
	output_dir: Path = typer.Option(
		Path('results/robustness'),
		'--output-dir',
		'-o',
		help='Directory to save results',
	),
	model_name: Optional[str] = typer.Option(
		None,
		'--model-name',
		'-n',
		help='Model name for results (default: filename stem)',
	),
	skip_corruptions: bool = typer.Option(
		False,
		'--skip-corruptions',
		help='Skip CIFAR-10-C evaluation',
	),
	skip_pgd: bool = typer.Option(
		False,
		'--skip-pgd',
		help='Skip PGD adversarial evaluation',
	),
):
	"""
	Evaluate model robustness on corruptions and adversarial examples.

	This command supports two modes:
	1. Single model evaluation: provide --model-path with optional --model-arch and --bottleneck-width
	2. Train and evaluate all models: provide --config with model definitions

	If --model-path is provided, evaluates a single existing model.
	Otherwise, trains and evaluates all models defined in the config file.
	"""
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logger.info(f'Using device: {device}')

	# Ensure output directory exists
	output_dir.mkdir(parents=True, exist_ok=True)

	# Load configuration
	try:
		full_config = load_config(config)
	except FileNotFoundError as e:
		logger.error(str(e))
		raise typer.Exit(1)

	models_config = full_config.get('models', [])
	training_defaults = full_config.get('training_defaults', {})
	eval_settings = full_config.get('evaluation', {})

	# Build evaluation config
	eval_config = {
		'eval_batch_size': eval_settings.get('batch_size', 64),
		'eval_num_workers': eval_settings.get('num_workers', 4),
	}

	# PGD settings
	pgd_config = eval_settings.get('pgd', {})
	if pgd_config:
		eval_config['pgd_epsilon'] = pgd_config.get('epsilon', 8 / 255)
		eval_config['pgd_alpha'] = pgd_config.get('alpha', 2 / 255)
		eval_config['pgd_iterations'] = pgd_config.get('iterations', 10)
		eval_config['pgd_max_samples'] = pgd_config.get('max_samples', 1000)

	# Corruption settings
	corruption_config = eval_settings.get('corruptions', {})
	if corruption_config:
		eval_config['corruption_types'] = corruption_config.get('types', CORRUPTION_TYPES)
		eval_config['severity_levels'] = corruption_config.get('severity_levels', SEVERITY_LEVELS)

	# Override with CLI flags
	if skip_corruptions:
		eval_config['corruption_types'] = []
	if skip_pgd:
		eval_config['skip_pgd'] = True

	all_results = []

	if model_path:
		# Mode 1: Evaluate single existing model
		if model_arch is None or bottleneck_width is None:
			# Try to find matching config in YAML
			found_config = None
			if model_name:
				for m in models_config:
					if m['name'] == model_name:
						found_config = m
						break
			if found_config is None:
				logger.error(
					'When using --model-path, either provide --model-arch and --bottleneck-width, '
					'or ensure --model-name matches a config entry'
				)
				raise typer.Exit(1)

			model_arch = found_config['model_arch']
			bottleneck_width = found_config.get('bottleneck_width')

		logger.info(f'Evaluating single model: {model_path}')
		model = load_model(
			model_arch=model_arch,
			bottleneck_width=bottleneck_width,
			model_path=model_path,
			device=device,
		)

		# Load test data
		logger.info('Loading CIFAR-10 test data...')
		_, _, test_loader = load_cifar10_dataset(
			data_dir=data_dir,
			train_batch_size=training_defaults.get('train_batch_size', 128),
			val_batch_size=training_defaults.get('val_batch_size', 128),
			test_batch_size=training_defaults.get('test_batch_size', 128),
			train_val_split_ratio=training_defaults.get('train_val_split_ratio', 0.8),
			num_workers=training_defaults.get('num_workers', 4),
			download=True,
		)

		name = model_name or model_path.stem
		results = evaluate_model_robustness(
			model=model,
			model_name=name,
			data_dir=data_dir,
			output_dir=output_dir,
			test_loader=test_loader,
			eval_config=eval_config,
		)

		all_results.append(results)

	else:
		# Mode 2: Train and evaluate all models from config
		if not models_config:
			logger.error(
				'No models defined in config and --model-path not provided. '
				'Either add models to config or use --model-path for single model evaluation.'
			)
			raise typer.Exit(1)

		logger.info(f'Training and evaluating {len(models_config)} models...')

		for model_config in models_config:
			model_name = model_config['name']
			logger.info(f'\n{"=" * 60}')
			logger.info(f'Model: {model_name}')
			logger.info(f'Description: {model_config.get("description", "N/A")}')
			logger.info(f'{"=" * 60}')

			try:
				# Train model
				model, test_loader, name = train_model_from_config(
					model_config=model_config,
					training_defaults=training_defaults,
					data_dir=data_dir,
					device=device,
				)

				# Evaluate robustness
				results = evaluate_model_robustness(
					model=model,
					model_name=name,
					data_dir=data_dir,
					output_dir=output_dir,
					test_loader=test_loader,
					eval_config=eval_config,
				)

				all_results.append(results)

			except Exception as e:
				logger.error(f'Failed to train/evaluate {model_name}: {e}')
				continue

	# Compare results if multiple models evaluated
	if len(all_results) > 1:
		model_names = [r['model_name'] for r in all_results]
		compare_robustness(all_results, model_names)

	# Save combined results
	if len(all_results) > 1:
		combined_path = output_dir / 'all_robustness_results.json'
		combined = {
			'models': all_results,
			'comparison': {
				r['model_name']: {k: v for k, v in r.items() if isinstance(v, (int, float))}
				for r in all_results
			},
		}
		with open(combined_path, 'w', encoding='utf-8') as f:
			json.dump(combined, f, indent=2)
		logger.info(f'Combined results saved to {combined_path}')

	logger.success('Robustness evaluation completed!')

	return all_results


if __name__ == '__main__':
	app()
