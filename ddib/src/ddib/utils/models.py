import torch.nn as nn


def create_simple_ffn(
    input_size: int, hidden_sizes: list, output_size: int, dropout_rate: float = 0.1
) -> nn.Module:
    """
    Create a simple feedforward neural network.

    Args:
        input_size: Size of the input features
        hidden_sizes: List of sizes for hidden layers
        output_size: Size of the output
        dropout_rate: Dropout rate to use in hidden layers

    Returns:
        Sequential neural network module
    """
    layers = []
    prev_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)
