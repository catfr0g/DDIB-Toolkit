"""
Metrics for evaluating model robustness.

This module provides metrics commonly used in robustness evaluation:
- mCE: mean Corruption Error
- mAA: mean Accuracy under Adversarial attacks
"""

from typing import Dict, List, Optional

from loguru import logger
import numpy as np

# Reference accuracies for ImageNet-C corruptions (from original paper)
# These are used to normalize corruption error rates
IMAGENET_C_REFERENCE_ACCURACIES = {
	'gaussian_noise': [0.886, 0.874, 0.846, 0.798, 0.726],
	'shot_noise': [0.896, 0.880, 0.852, 0.804, 0.736],
	'impulse_noise': [0.894, 0.878, 0.848, 0.796, 0.724],
	'defocus_blur': [0.898, 0.882, 0.854, 0.810, 0.748],
	'glass_blur': [0.884, 0.862, 0.826, 0.774, 0.696],
	'motion_blur': [0.892, 0.876, 0.846, 0.798, 0.732],
	'zoom_blur': [0.890, 0.872, 0.840, 0.788, 0.718],
	'snow': [0.902, 0.888, 0.862, 0.820, 0.758],
	'frost': [0.896, 0.880, 0.852, 0.808, 0.744],
	'fog': [0.900, 0.886, 0.860, 0.818, 0.756],
	'brightness': [0.894, 0.884, 0.868, 0.836, 0.788],
	'contrast': [0.898, 0.882, 0.854, 0.806, 0.738],
	'elastic_transform': [0.892, 0.874, 0.842, 0.790, 0.716],
	'pixelate': [0.904, 0.892, 0.870, 0.832, 0.772],
	'jpeg_compression': [0.900, 0.884, 0.856, 0.810, 0.742],
}

# Severity levels
SEVERITY_LEVELS = [1, 2, 3, 4, 5]


def calculate_corruption_error(
	accuracies: Dict[str, Dict[int, float]],
	reference_accuracies: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, float]:
	"""
	Calculate Corruption Error (CE) for each corruption type.

	CE is the normalized error rate across severity levels, where normalization
	is done using reference accuracies from a standard model (e.g., AlexNet).

	Args:
	    accuracies: Dict mapping corruption_type -> severity -> accuracy
	    reference_accuracies: Reference accuracies for normalization

	Returns:
	    Dict mapping corruption_type -> corruption_error
	"""
	if reference_accuracies is None:
		reference_accuracies = IMAGENET_C_REFERENCE_ACCURACIES

	corruption_errors = {}

	for corruption_type, severity_accuracies in accuracies.items():
		if corruption_type not in reference_accuracies:
			logger.warning(
				f'No reference accuracy for {corruption_type}, using unnormalized error'
			)
			# Use simple average error
			errors = [
				1 - severity_accuracies.get(sev, 0.0)
				for sev in SEVERITY_LEVELS
				if sev in severity_accuracies
			]
			corruption_errors[corruption_type] = np.mean(errors) if errors else 0.0
		else:
			ref = reference_accuracies[corruption_type]
			errors = []
			for sev in SEVERITY_LEVELS:
				if sev in severity_accuracies:
					# Normalized error: (model_error) / (reference_error)
					model_error = 1 - severity_accuracies[sev]
					ref_error = 1 - ref[sev - 1]  # ref is 0-indexed
					if ref_error > 0:
						errors.append(model_error / ref_error)
					else:
						errors.append(model_error)

			corruption_errors[corruption_type] = np.mean(errors) if errors else 0.0

	return corruption_errors


def calculate_mean_corruption_error(
	corruption_errors: Dict[str, float],
) -> float:
	"""
	Calculate mean Corruption Error (mCE) across all corruption types.

	Args:
	    corruption_errors: Dict mapping corruption_type -> corruption_error

	Returns:
	    mean Corruption Error (mCE)
	"""
	if not corruption_errors:
		return 0.0

	return np.mean(list(corruption_errors.values()))


def calculate_mean_accuracy_adversarial(
	clean_accuracy: float,
	adversarial_accuracies: List[float],
) -> Dict[str, float]:
	"""
	Calculate mean Accuracy under Adversarial attacks (mAA).

	Args:
	    clean_accuracy: Accuracy on clean images
	    adversarial_accuracies: List of accuracies under different attack settings

	Returns:
	    Dict with mAA and related metrics
	"""
	if not adversarial_accuracies:
		return {
			'mAA': 0.0,
			'robustness_gap': 0.0,
			'relative_robustness': 0.0,
		}

	maa = np.mean(adversarial_accuracies)
	robustness_gap = clean_accuracy - maa
	relative_robustness = maa / clean_accuracy if clean_accuracy > 0 else 0.0

	return {
		'mAA': maa,
		'robustness_gap': robustness_gap,
		'relative_robustness': relative_robustness,
		'std_adversarial': np.std(adversarial_accuracies),
	}


def calculate_robustness_metrics(
	corruption_accuracies: Optional[Dict[str, Dict[int, float]]] = None,
	clean_accuracy: Optional[float] = None,
	adversarial_accuracies: Optional[List[float]] = None,
	reference_accuracies: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, float]:
	"""
	Calculate comprehensive robustness metrics.

	Args:
	    corruption_accuracies: Dict mapping corruption_type -> severity -> accuracy
	    clean_accuracy: Accuracy on clean data
	    adversarial_accuracies: List of accuracies under adversarial attacks
	    reference_accuracies: Reference accuracies for CE normalization

	Returns:
	    Dict with all computed robustness metrics
	"""
	metrics = {}

	# Corruption robustness metrics
	if corruption_accuracies:
		corruption_errors = calculate_corruption_error(corruption_accuracies, reference_accuracies)
		mce = calculate_mean_corruption_error(corruption_errors)

		# Add per-corruption errors
		for corruption_type, error in corruption_errors.items():
			metrics[f'CE_{corruption_type}'] = error

		metrics['mCE'] = mce

		# Also compute mean accuracy across corruptions
		all_acc = []
		for severity_acc in corruption_accuracies.values():
			all_acc.extend(severity_acc.values())
		metrics['mean_corruption_accuracy'] = np.mean(all_acc) if all_acc else 0.0

	# Adversarial robustness metrics
	if clean_accuracy is not None and adversarial_accuracies:
		adv_metrics = calculate_mean_accuracy_adversarial(clean_accuracy, adversarial_accuracies)
		metrics.update(adv_metrics)
		metrics['clean_accuracy'] = clean_accuracy

	return metrics


def print_robustness_report(metrics: Dict[str, float]) -> str:
	"""
	Generate a formatted report of robustness metrics.

	Args:
	    metrics: Dict of robustness metrics

	Returns:
	    Formatted report string
	"""
	lines = []
	lines.append('=' * 60)
	lines.append('ROBUSTNESS EVALUATION REPORT')
	lines.append('=' * 60)

	# Overall metrics
	if 'mCE' in metrics:
		lines.append(f'\nMean Corruption Error (mCE): {metrics["mCE"]:.4f}')

	if 'mAA' in metrics:
		lines.append(f'Mean Adversarial Accuracy (mAA): {metrics["mAA"]:.4f}')

	if 'clean_accuracy' in metrics:
		lines.append(f'Clean Accuracy: {metrics["clean_accuracy"]:.4f}')
		lines.append(f'Robustness Gap: {metrics.get("robustness_gap", 0):.4f}')
		lines.append(f'Relative Robustness: {metrics.get("relative_robustness", 0):.4f}')

	# Per-corruption errors
	corruption_errors = {k: v for k, v in metrics.items() if k.startswith('CE_')}
	if corruption_errors:
		lines.append('\nPer-Corruption Error Rates:')
		lines.append('-' * 40)
		for corruption_type, error in sorted(corruption_errors.items()):
			corruption_name = corruption_type.replace('CE_', '')
			lines.append(f'  {corruption_name}: {error:.4f}')

	lines.append('\n' + '=' * 60)

	report = '\n'.join(lines)
	logger.info(report)

	return report


def save_metrics_to_json(
	metrics: Dict[str, float],
	filepath: str,
) -> None:
	"""
	Save robustness metrics to a JSON file.

	Args:
	    metrics: Dict of robustness metrics
	    filepath: Path to save JSON file
	"""
	import json
	from pathlib import Path

	# Convert numpy types to Python types
	clean_metrics = {}
	for key, value in metrics.items():
		if isinstance(value, (np.floating, np.integer)):
			clean_metrics[key] = float(value)
		else:
			clean_metrics[key] = value

	# Ensure directory exists
	Path(filepath).parent.mkdir(parents=True, exist_ok=True)

	with open(filepath, 'w') as f:
		json.dump(clean_metrics, f, indent=2)

	logger.info(f'Metrics saved to {filepath}')


def compare_robustness(
	metrics_list: List[Dict[str, float]],
	model_names: List[str],
) -> str:
	"""
	Compare robustness metrics across multiple models.

	Args:
	    metrics_list: List of metrics dicts for each model
	    model_names: Names of the models

	Returns:
	    Formatted comparison table string
	"""
	lines = []
	lines.append('=' * 80)
	lines.append('ROBUSTNESS COMPARISON')
	lines.append('=' * 80)

	# Header
	header = f'{"Model":<20} {"mCE":<12} {"mAA":<12} {"Clean Acc":<12} {"Gap":<12}'
	lines.append(header)
	lines.append('-' * 80)

	# Rows
	for name, metrics in zip(model_names, metrics_list):
		mce = metrics.get('mCE', 0)
		maa = metrics.get('mAA', 0)
		clean = metrics.get('clean_accuracy', 0)
		gap = metrics.get('robustness_gap', 0)

		row = f'{name:<20} {mce:<12.4f} {maa:<12.4f} {clean:<12.4f} {gap:<12.4f}'
		lines.append(row)

	lines.append('=' * 80)

	report = '\n'.join(lines)
	logger.info(report)

	return report
