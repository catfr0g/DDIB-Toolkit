"""
Contains some usefull utils for pytorch models creation
"""

from typing import List, Optional

import torch
from torch import nn
import torchvision


def create_simple_ffn(
    input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.1
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
    layers: List[nn.Module] = []
    prev_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(prev_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        prev_size = hidden_size
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


def forward_and_layer_outs(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_names: list[str]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return results of forward pass and layer outputs of layers from specified names"""
    outputs = {}

    def register_hook(name):
        def hook(module, input, output): # pylint: disable=redefined-builtin
            outputs[name] = input[0] if isinstance(input, tuple) else input
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            module.register_forward_hook(register_hook(name))
    y = model(input_tensor)
    return y, outputs


class ResNetWithBottleneck(nn.Module):
    """
    ResNet model with configurable bottleneck layer between features and classifier.

    This is a wrapper around torchvision's ResNet that adds an optional bottleneck
    layer between the feature extractor and classifier.

    Args:
        arch: Name of the ResNet architecture (e.g., 'resnet18', 'resnet34', etc.)
        num_classes: Number of output classes
        bottleneck_width: Width of the layer between features and classifier
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 10,
        bottleneck_width: Optional[int] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        # Load the pretrained ResNet model
        self.resnet = getattr(torchvision.models, arch)(pretrained=pretrained)
        # Replace the final classifier to match the number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the original classifier
        # Add optional bottleneck layer between features and classifier
        self.bottleneck = None
        classifier_input_size = num_features
        if bottleneck_width is not None:
            # Group bottleneck components in a sequential for easier access via hooks
            self.bottleneck = nn.Sequential(
                nn.Linear(num_features, bottleneck_width),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            classifier_input_size = bottleneck_width
        # Final classifier layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResNet model with optional bottleneck."""
        # Extract features using the ResNet backbone
        x = self.resnet(x)
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        x = self.classifier(x)
        return x


class VGGWithBottleneck(nn.Module):
    """
    VGG model with configurable bottleneck layer between features and classifier.

    This is a wrapper around torchvision's VGG that adds an optional bottleneck
    layer between the feature extractor and classifier.

    Args:
        arch: Name of the VGG architecture (e.g., 'vgg11', 'vgg16', etc.)
        num_classes: Number of output classes
        bottleneck_width: Width of the layer between features and classifier
        pretrained: Whether to use pretrained weights
    """

    def __init__(
        self,
        arch: str = "vgg11",
        num_classes: int = 10,
        bottleneck_width: Optional[int] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        # Load the pretrained VGG model
        self.vgg = getattr(torchvision.models, arch)(pretrained=pretrained)
        num_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Identity()
        # Add optional bottleneck layer between features and classifier
        self.bottleneck = None
        classifier_input_size = num_features
        if bottleneck_width is not None:
            # Group bottleneck components in a sequential for easier access via hooks
            self.bottleneck = nn.Sequential(
                nn.Linear(num_features, bottleneck_width),
                nn.ReLU(True),
                nn.Dropout(),
            )
            classifier_input_size = bottleneck_width
        # Final classifier layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VGG model with optional bottleneck."""
        # Extract features using the VGG backbone
        x = self.vgg(x)

        # Apply bottleneck if specified
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        # Apply final classifier
        x = self.classifier(x)

        return x
