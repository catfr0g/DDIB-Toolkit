"""
Configuration file with best model configurations from grid search experiments.

These configurations are based on the analysis of grid search results
from results/grid_search_*/grid_search_results_final.json files.
"""

from typing import Any, Dict, List

# Best configurations found from grid search experiments
BEST_CONFIGURATIONS: List[Dict[str, Any]] = [
	# VGG11 - Best overall performance
	{
		'name': 'vgg11_best',
		'model_arch': 'vgg11',
		'bottleneck_width': 2048,
		'beta': 1e-08,
		'seed': 1984,
		'test_accuracy': 85.92,
		'description': 'VGG11 with 2048 bottleneck - best overall accuracy',
	},
	# VGG11 - Alternative good configuration
	{
		'name': 'vgg11_alt',
		'model_arch': 'vgg11',
		'bottleneck_width': 2048,
		'beta': 1e-06,
		'seed': 12,
		'test_accuracy': 85.75,
		'description': 'VGG11 with 2048 bottleneck - alternative config',
	},
	# ResNet18 - Best configuration
	{
		'name': 'resnet18_best',
		'model_arch': 'resnet18',
		'bottleneck_width': 2048,
		'beta': 1e-06,
		'seed': 42,
		'test_accuracy': 84.65,
		'description': 'ResNet18 with 2048 bottleneck - best accuracy',
	},
	# ResNet18 - Alternative good configuration
	{
		'name': 'resnet18_alt',
		'model_arch': 'resnet18',
		'bottleneck_width': 2048,
		'beta': 1e-07,
		'seed': 42,
		'test_accuracy': 84.0,
		'description': 'ResNet18 with 2048 bottleneck - alternative config',
	},
	# EfficientNet-B0 - Representative configuration
	{
		'name': 'efficientnet_b0_best',
		'model_arch': 'efficientnet_b0',
		'bottleneck_width': 16,
		'beta': 1e-06,
		'seed': 42,
		'test_accuracy': 78.31,
		'description': 'EfficientNet-B0 with 16 bottleneck - best found',
	},
]

# Default training hyperparameters for robustness evaluation
DEFAULT_TRAINING_HYPERPARAMS: Dict[str, Any] = {
	'num_epochs': 100,
	'train_batch_size': 128,
	'val_batch_size': 128,
	'test_batch_size': 128,
	'learning_rate': 1e-3,
	'weight_decay': 1e-4,
	'alpha': 1.01,  # DDIB alpha parameter
	'beta': 1.0,  # DDIB beta parameter (will be overridden by config)
	'train_val_split_ratio': 0.8,
	'num_workers': 4,
}

# Robustness evaluation settings
ROBUSTNESS_EVAL_CONFIG: Dict[str, Any] = {
	# PGD Attack settings
	'pgd_epsilon': 8 / 255,
	'pgd_alpha': 2 / 255,
	'pgd_iterations': 10,
	'pgd_norm': 'inf',
	# CIFAR-10-C settings
	'corruption_types': [
		'gaussian_noise',
		'shot_noise',
		'impulse_noise',
		'defocus_blur',
		'glass_blur',
		'motion_blur',
		'zoom_blur',
		'snow',
		'frost',
		'fog',
		'brightness',
		'contrast',
		'elastic_transform',
		'pixelate',
		'jpeg_compression',
	],
	'severity_levels': [1, 2, 3, 4, 5],
	# Evaluation batch size
	'eval_batch_size': 64,
	'eval_num_workers': 4,
}


def get_config_by_name(name: str) -> Dict[str, Any]:
	"""
	Get a model configuration by name.

	Args:
	    name: Configuration name (e.g., 'vgg11_best')

	Returns:
	    Configuration dictionary

	Raises:
	    ValueError: If configuration name not found
	"""
	for config in BEST_CONFIGURATIONS:
		if config['name'] == name:
			return config.copy()

	raise ValueError(
		f"Configuration '{name}' not found. Available: {[c['name'] for c in BEST_CONFIGURATIONS]}"
	)


def get_configs_by_arch(arch: str) -> List[Dict[str, Any]]:
	"""
	Get all configurations for a specific model architecture.

	Args:
	    arch: Model architecture name (e.g., 'vgg11', 'resnet18')

	Returns:
	    List of configuration dictionaries
	"""
	return [c.copy() for c in BEST_CONFIGURATIONS if c['model_arch'] == arch]


def get_best_config() -> Dict[str, Any]:
	"""
	Get the best overall configuration.

	Returns:
	    Configuration dictionary for the best model
	"""
	# Configurations are sorted by accuracy, best first
	return BEST_CONFIGURATIONS[0].copy()


def get_all_configs() -> List[Dict[str, Any]]:
	"""
	Get all available configurations.

	Returns:
	    List of all configuration dictionaries
	"""
	return [c.copy() for c in BEST_CONFIGURATIONS]
