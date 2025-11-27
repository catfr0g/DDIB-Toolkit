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
    A = X / torch.trace(X)
    eigv = torch.abs(torch.linalg.eigh(A)[0])  # pylint: disable=not-callable
    eig_pow = eigv**alpha
    return torch.log2(torch.sum(eig_pow)) / (1 - alpha)


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
    k = k / torch.trace(k)
    return reyi_entropy(k, alpha=alpha)


def calculate_MI(x: torch.Tensor, y: torch.Tensor, *, alpha=1.01) -> torch.Tensor:
    """
    Calculate mutual information between tensors x and y. X and Y should be in Gramm matrix form.

    Args:
        x: First input tensor
        y: Second input tensor
        s_x: Sigma parameter for x
        s_y: Sigma parameter for y

    Returns:
        Calculated mutual information value
    """
    Hx = reyi_entropy(x, alpha=alpha)
    Hy = reyi_entropy(y, alpha=alpha)
    Hxy = joint_entropy(x, y, alpha=alpha)
    return Hx + Hy - Hxy


def calculate_kernel_width(x: torch.Tensor, top_k=10) -> float:
    """Function to calculate kernel width for Gramm Matrix transformation"""
    x_detached = x.detach()
    with torch.no_grad():
        dist_matrix = torch.cdist(x_detached, x_detached, p=2)
        sorted_dists, _ = torch.sort(dist_matrix, dim=1)
        k_closest = sorted_dists[:, 1 : (top_k + 1)]
        mean_of_10_closest_per_point = torch.mean(k_closest, dim=1)
        sigma_z = torch.mean(mean_of_10_closest_per_point)
        assert sigma_z.shape == torch.Size([]), (
            f"Expected sigma_z to be a scalar tensor, but got shape {sigma_z.shape}"
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
        X_gram = rbf_kernel(X, X, sigma=calculate_kernel_width(X, top_k=self.top_k))
        Z_gram = rbf_kernel(Z, Z, sigma=calculate_kernel_width(X, top_k=self.top_k))
        mutual_info = calculate_MI(X_gram, Z_gram)
        original_loss = self.original_loss(y_pred, y_true)
        return original_loss + self.beta * mutual_info
