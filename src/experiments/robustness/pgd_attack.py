"""
PGD (Projected Gradient Descent) attack implementation for adversarial robustness evaluation.

PGD is an iterative adversarial attack that maximizes the loss while keeping
perturbations within an epsilon-ball around the original image.
"""

from typing import Dict, Optional, Tuple

from loguru import logger
import torch
from torch import nn


class PGDAttack:
	"""
	Projected Gradient Descent (PGD) adversarial attack.

	PGD iteratively applies gradient updates to create adversarial examples,
	projecting the result back into the epsilon-ball after each step.

	Args:
	    model: The target model to attack
	    epsilon: Maximum perturbation size (e.g., 8/255)
	    alpha: Step size for each iteration (e.g., 2/255)
	    iterations: Number of iterations to run
	    norm: Norm to use for projection ('inf' or '2')
	    loss_fn: Loss function to maximize (default: CrossEntropyLoss)
	    targeted: Whether this is a targeted attack (default: False)
	"""

	def __init__(
		self,
		model: nn.Module,
		epsilon: float = 8 / 255,
		alpha: float = 2 / 255,
		iterations: int = 10,
		norm: str = 'inf',
		loss_fn: Optional[nn.Module] = None,
		targeted: bool = False,
	):
		self.model = model
		self.epsilon = epsilon
		self.alpha = alpha
		self.iterations = iterations
		self.norm = norm
		self.loss_fn = loss_fn or nn.CrossEntropyLoss()
		self.targeted = targeted

		logger.info(
			f'PGD Attack initialized: epsilon={epsilon:.4f}, alpha={alpha:.4f}, '
			f'iterations={iterations}, norm=L{norm}'
		)

	def perturb(
		self,
		images: torch.Tensor,
		labels: torch.Tensor,
		random_start: bool = True,
	) -> torch.Tensor:
		"""
		Generate adversarial examples using PGD.

		Args:
		    images: Input images tensor of shape (N, C, H, W)
		    labels: Target labels tensor of shape (N,)
		    random_start: Whether to start from a random point in epsilon-ball

		Returns:
		    Adversarial images tensor of same shape as input
		"""
		self.model.eval()

		# Ensure inputs are in [0, 1] range for proper clipping
		images = images.clamp(0, 1)

		# Initialize perturbation
		if random_start:
			# Start from random point within epsilon-ball
			delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
			delta = self._project(delta)
		else:
			# Start from zero perturbation
			delta = torch.zeros_like(images)

		delta.requires_grad = True

		# Create optimizer for the perturbation
		optimizer = torch.optim.Adam([delta], lr=self.alpha)

		# Iterative attack
		for i in range(self.iterations):
			optimizer.zero_grad()

			# Forward pass with perturbed images
			adv_images = images + delta
			adv_images = adv_images.clamp(0, 1)

			# Compute loss
			outputs = self.model(adv_images)

			if self.targeted:
				# For targeted attack, minimize loss to target
				loss = self.loss_fn(outputs, labels)
			else:
				# For untargeted attack, maximize loss (minimize negative loss)
				loss = -self.loss_fn(outputs, labels)

			# Backward pass
			loss.backward()

			# Update perturbation
			optimizer.step()

			# Project back to epsilon-ball
			delta = self._project(delta)

		# Return final adversarial examples
		adv_images = images + delta
		return adv_images.clamp(0, 1)

	def _project(self, delta: torch.Tensor) -> torch.Tensor:
		"""Project delta to the epsilon-ball using the specified norm."""
		if self.norm == 'inf':
			# L-infinity projection
			return torch.clamp(delta, -self.epsilon, self.epsilon)
		elif self.norm == '2':
			# L2 projection
			norm = delta.view(delta.size(0), -1).norm(2, dim=1, keepdim=True)
			# Add small epsilon to avoid division by zero
			norm = torch.clamp(norm, min=1e-10)
			# Scale delta to have norm <= epsilon
			scale = self.epsilon / norm
			scale = torch.clamp(scale, max=1.0)  # Don't scale if already within ball
			delta_flat = delta.view(delta.size(0), -1)
			delta_flat = delta_flat * scale
			return delta_flat.view_as(delta)
		else:
			raise ValueError(f'Unsupported norm: {self.norm}')

	def attack_accuracy(
		self,
		images: torch.Tensor,
		labels: torch.Tensor,
		batch_size: int = 32,
		device: Optional[torch.device] = None,
	) -> Tuple[float, Dict[str, float]]:
		"""
		Evaluate model accuracy under PGD attack.

		Args:
		    images: Input images tensor
		    labels: True labels tensor
		    batch_size: Batch size for processing
		    device: Device to run computation on

		Returns:
		    Tuple of (adversarial_accuracy, metrics_dict)
		"""
		if device is None:
			device = next(self.model.parameters()).device

		self.model.eval()

		total_samples = images.size(0)
		correct = 0
		all_adv_images = []
		all_preds = []

		# Process in batches to avoid OOM
		with torch.no_grad():
			for i in range(0, total_samples, batch_size):
				batch_images = images[i : i + batch_size].to(device)
				batch_labels = labels[i : i + batch_size].to(device)

				# Generate adversarial examples
				adv_images = self.perturb(batch_images, batch_labels)

				# Evaluate on adversarial examples
				outputs = self.model(adv_images)
				preds = outputs.argmax(dim=1)

				correct += (preds == batch_labels).sum().item()

				all_adv_images.append(adv_images.cpu())
				all_preds.append(preds.cpu())

		adversarial_accuracy = correct / total_samples

		metrics = {
			'adversarial_accuracy': adversarial_accuracy,
			'adversarial_error_rate': 1 - adversarial_accuracy,
			'robust_accuracy': adversarial_accuracy,
		}

		logger.info(
			f'PGD Attack completed: '
			f'Adversarial Accuracy = {adversarial_accuracy:.4f} '
			f'({correct}/{total_samples})'
		)

		return adversarial_accuracy, metrics


def create_pgd_attack(
	model: nn.Module,
	epsilon: float = 8 / 255,
	alpha: float = 2 / 255,
	iterations: int = 10,
	**kwargs,
) -> PGDAttack:
	"""
	Factory function to create a PGD attack with standard parameters.

	Args:
	    model: Target model
	    epsilon: Maximum perturbation (default: 8/255)
	    alpha: Step size (default: 2/255)
	    iterations: Number of iterations (default: 10)
	    **kwargs: Additional arguments to PGDAttack

	Returns:
	    Configured PGDAttack instance
	"""
	return PGDAttack(
		model=model,
		epsilon=epsilon,
		alpha=alpha,
		iterations=iterations,
		**kwargs,
	)


def evaluate_adversarial_robustness(
	model: nn.Module,
	images: torch.Tensor,
	labels: torch.Tensor,
	epsilon: float = 8 / 255,
	alpha: float = 2 / 255,
	iterations: int = 10,
	batch_size: int = 32,
	device: Optional[torch.device] = None,
) -> Dict[str, float]:
	"""
	Evaluate model robustness against PGD adversarial attacks.

	Args:
	    model: Model to evaluate
	    images: Clean images tensor
	    labels: True labels tensor
	    epsilon: Maximum perturbation
	    alpha: Step size
	    iterations: Number of PGD iterations
	    batch_size: Batch size for evaluation
	    device: Device to use

	Returns:
	    Dictionary with robustness metrics
	"""
	if device is None:
		device = next(model.parameters()).device

	# First evaluate on clean images
	model.eval()
	with torch.no_grad():
		clean_correct = 0
		for i in range(0, len(images), batch_size):
			batch_images = images[i : i + batch_size].to(device)
			batch_labels = labels[i : i + batch_size].to(device)
			outputs = model(batch_images)
			preds = outputs.argmax(dim=1)
			clean_correct += (preds == batch_labels).sum().item()

	clean_accuracy = clean_correct / len(images)

	# Then evaluate under PGD attack
	attack = PGDAttack(
		model=model,
		epsilon=epsilon,
		alpha=alpha,
		iterations=iterations,
	)

	adv_accuracy, adv_metrics = attack.attack_accuracy(
		images=images,
		labels=labels,
		batch_size=batch_size,
		device=device,
	)

	# Compile results
	results = {
		'clean_accuracy': clean_accuracy,
		'adversarial_accuracy': adv_accuracy,
		'robustness_gap': clean_accuracy - adv_accuracy,
		'relative_robustness': adv_accuracy / clean_accuracy if clean_accuracy > 0 else 0.0,
	}
	results.update(adv_metrics)

	logger.info(
		f'Robustness Evaluation Results:\n'
		f'  Clean Accuracy: {clean_accuracy:.4f}\n'
		f'  Adversarial Accuracy: {adv_accuracy:.4f}\n'
		f'  Robustness Gap: {clean_accuracy - adv_accuracy:.4f}\n'
		f'  Relative Robustness: {results["relative_robustness"]:.4f}'
	)

	return results
