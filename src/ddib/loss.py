# -*- coding: utf-8 -*-
"""
Method to transform general loss of the model to its Deep Determenistic Information Bottelneck version version
"""

import torch
from torch import nn


def rbf_kernel(X: torch.Tensor, Y: torch.Tensor, *, sigma: float = 1.0) -> torch.Tensor:
	"""
	Compute the radial basis function (RBF / Gaussian) kernel between two sets of vectors.

	Args:
	    X: Tensor of shape (n_samples_X, n_features)
	    Y: Tensor of shape (n_samples_Y, n_features)
	    sigma: Bandwidth parameter (float or tensor)

	Returns:
	    K: Kernel matrix of shape (n_samples_X, n_samples_Y)
	"""
	X_norm = (X**2).sum(dim=1, keepdim=True)
	Y_norm = (Y**2).sum(dim=1, keepdim=True)
	XY = torch.mm(X, Y.t())
	dist_sq = X_norm + Y_norm.t() - 2 * XY
	return torch.exp(-dist_sq / (2 * sigma**2))


def reyi_entropy(X: torch.Tensor, *, alpha: float = 1.01) -> torch.Tensor:
	"""
	Calculate Renyi entropy for the given tensor x and sigma parameter. Input tensor should be Gramm matrix.

	Args:
	    x: Input tensor
	    sigma: Sigma parameter for the Gaussian kernel
	    alpha: Alpha parameter for Renyi entropy (default 1.01)

	Returns:
	    Calculated Renyi entropy value
	"""
	# Normalize the matrix to have trace = 1 (creating a density matrix)
	trace_X = torch.trace(X)
	if trace_X < 1e-10:
		return torch.tensor(0.0, dtype=X.dtype, device=X.device)

	A = X / trace_X

	# Add small regularization to improve conditioning
	eps = torch.finfo(A.dtype).eps * 1e4
	A_reg = A + eps * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

	try:
		# Compute eigenvalues using eigh (for symmetric matrices)
		eigv = torch.linalg.eigh(A_reg)[0]  # pylint: disable=not-callable
	except torch._C._LinAlgError:
		# Fallback: use diagonal elements if eigendecomposition fails
		eigv = torch.diag(A_reg)

	# Clamp eigenvalues to be positive (numerical stability)
	min_eigv = max(eps, 1e-10)
	eigv = torch.clamp(eigv, min=min_eigv)

	# Normalize eigenvalues to sum to 1 (proper probability distribution)
	eigv = eigv / eigv.sum()

	# Compute Renyi entropy using numerically stable formula
	# For alpha close to 1, this approaches Shannon entropy
	log_sum = torch.log2((eigv**alpha).sum())
	entropy_value = log_sum / (1 - alpha)

	# Renyi entropy should be non-negative for valid probability distributions
	# but can be negative for continuous distributions (which is OK)
	# However, for MI calculation we need consistent scaling

	# Handle potential NaN or Inf values
	if torch.isnan(entropy_value) or torch.isinf(entropy_value):
		return torch.tensor(0.0, dtype=X.dtype, device=X.device)

	return entropy_value


def joint_entropy(x: torch.Tensor, y: torch.Tensor, *, alpha: float = 1.01) -> torch.Tensor:
	"""
	Calculate joint entropy for tensors x and y. Input tensors should be Gramm matricies.

	Args:
	    x: First input tensor
	    y: Second input tensor
	    s_x: Sigma parameter for x
	    s_y: Sigma parameter for y
	    alpha: Alpha parameter for Renyi entropy (default 1.01)

	Returns:
	    Calculated joint entropy value
	"""
	k = torch.mul(x, y)

	# Check if trace is zero or very small to avoid division by zero
	trace_k = torch.trace(k)
	if torch.abs(trace_k) < 1e-12:
		# Return a small constant value for near-zero case
		return torch.tensor(0.0, dtype=x.dtype, device=x.device)

	k = k / trace_k
	return reyi_entropy(k, alpha=alpha)


def calculate_MI(x: torch.Tensor, y: torch.Tensor, *, alpha=1.01) -> torch.Tensor:
	"""
	Calculate mutual information between tensors x and y. X and Y should be in Gramm matrix form.

	Args:
	    x: First input tensor
	    y: Second input tensor
	    alpha: Alpha parameter for Renyi entropy (default 1.01)

	Returns:
	    Calculated mutual information value
	"""
	Hx = reyi_entropy(x, alpha=alpha)
	Hy = reyi_entropy(y, alpha=alpha)
	Hxy = joint_entropy(x, y, alpha=alpha)

	# Calculate MI but handle potential numerical issues
	mi = Hx + Hy - Hxy

	# Ensure MI is non-negative (due to numerical errors it might be slightly negative)
	mi = torch.clamp(mi, min=0.0)

	# Handle potential NaN or Inf values
	if torch.isnan(mi) or torch.isinf(mi):
		return torch.tensor(0.0, dtype=x.dtype, device=x.device)

	return mi


def calculate_kernel_width(x: torch.Tensor, top_k=10) -> float:
	"""Function to calculate kernel width for Gramm Matrix transformation"""
	x_detached = x.detach()
	with torch.no_grad():
		# Use more efficient distance calculation that's GPU-friendly
		x_norm = (x_detached**2).sum(dim=1, keepdim=True)
		dist_matrix = x_norm + x_norm.t() - 2 * torch.mm(x_detached, x_detached.t())
		# Take square root to get actual distances
		dist_matrix = torch.sqrt(torch.clamp(dist_matrix, min=0.0))

		# Zero out diagonal elements to exclude self-distances
		dist_matrix.fill_diagonal_(float('inf'))

		sorted_dists, _ = torch.sort(dist_matrix, dim=1)
		k_closest = sorted_dists[:, :top_k]  # Take top_k closest (excluding self)
		mean_of_k_closest_per_point = torch.mean(k_closest, dim=1)
		sigma_z = torch.mean(mean_of_k_closest_per_point)
		assert sigma_z.shape == torch.Size([]), (
			f'Expected sigma_z to be a scalar tensor, but got shape {sigma_z.shape}'
		)
		return sigma_z.item()


class DDIB_Regularization(nn.Module):
	"""Loss wraper for making any training powerd by information bottelneck!"""

	def __init__(self, original_loss: nn.Module, beta: float = 0.01, top_k: int = 10):
		super(DDIB_Regularization, self).__init__()
		self.original_loss = original_loss
		self.beta = beta
		self.top_k = top_k

	def forward(
		self, y_pred: torch.Tensor, y_true: torch.Tensor, X: torch.Tensor, Z: torch.Tensor
	) -> torch.Tensor:
		"""
		Args:
		    y_true: torch.Tensor - labels
		    y_pred: torch.Tensor - predicted results
		    X: torch.Tensor - input data
		    Z: torch.Tensor - output of the layer to optimize
		"""
		# Flatten both X and Z if they are not 2D (for cases where bottleneck input is conv features)
		if X.dim() > 2:
			X_flat = X.view(X.size(0), -1)
		else:
			X_flat = X

		if Z.dim() > 2:
			Z_flat = Z.view(Z.size(0), -1)
		else:
			Z_flat = Z

		try:
			X_gram = rbf_kernel(
				X_flat, X_flat, sigma=calculate_kernel_width(X_flat, top_k=self.top_k)
			)
			Z_gram = rbf_kernel(
				Z_flat, Z_flat, sigma=calculate_kernel_width(Z_flat, top_k=self.top_k)
			)  # Fixed: use Z_flat for Z_gram
			mutual_info = calculate_MI(X_gram, Z_gram)
		except Exception:
			# If there's an error calculating mutual information, return just the original loss
			mutual_info = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)

		original_loss = self.original_loss(y_pred, y_true)

		# Handle potential numerical issues in the final result
		result = original_loss + self.beta * mutual_info

		# Check for NaN or Inf values and handle them
		if torch.isnan(result) or torch.isinf(result):
			# Return just the original loss if the combined result is invalid
			return original_loss

		return result
