"""
Robustness evaluation module for DDIB models.

This module provides tools for evaluating model robustness against:
- ImageNet-C / CIFAR-10-C corruptions
- Adversarial examples (PGD attack)
"""

from .imagenet_c import (
	CORRUPTION_TYPES,
	SEVERITY_LEVELS,
	ImageNetCCIFAR10Dataset,
	ImageNetCDataset,
	create_cifar10_c_dataloader,
	create_imagenet_c_dataloader,
	get_default_corruptions,
	get_default_severities,
)
from .metrics import (
	IMAGENET_C_REFERENCE_ACCURACIES,
	calculate_corruption_error,
	calculate_mean_accuracy_adversarial,
	calculate_mean_corruption_error,
	calculate_robustness_metrics,
	compare_robustness,
	print_robustness_report,
	save_metrics_to_json,
)
from .pgd_attack import (
	PGDAttack,
	create_pgd_attack,
	evaluate_adversarial_robustness,
)
from .prepare_data import (
	download_cifar10c,
	organize_cifar10c,
	prepare_cifar10c,
	verify_cifar10c,
)

__all__ = [
	# Data preparation
	'prepare_cifar10c',
	'download_cifar10c',
	'organize_cifar10c',
	'verify_cifar10c',
	# Dataset
	'CORRUPTION_TYPES',
	'SEVERITY_LEVELS',
	'ImageNetCDataset',
	'ImageNetCCIFAR10Dataset',
	'create_imagenet_c_dataloader',
	'create_cifar10_c_dataloader',
	'get_default_corruptions',
	'get_default_severities',
	# PGD Attack
	'PGDAttack',
	'create_pgd_attack',
	'evaluate_adversarial_robustness',
	# Metrics
	'IMAGENET_C_REFERENCE_ACCURACIES',
	'calculate_corruption_error',
	'calculate_mean_corruption_error',
	'calculate_mean_accuracy_adversarial',
	'calculate_robustness_metrics',
	'print_robustness_report',
	'save_metrics_to_json',
	'compare_robustness',
]
