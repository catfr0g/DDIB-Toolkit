"""Tests for ResNet and VGG models with configurable bottleneck."""
import torch

from src.ddib.models import ResNetWithBottleneck, VGGWithBottleneck


def test_resnet_with_bottleneck():
    """Test ResNet with configurable bottleneck."""
    # Test ResNet-18 with bottleneck width of 128
    model = ResNetWithBottleneck(
        arch="resnet18",
        num_classes=10,
        bottleneck_width=128  # Bottleneck between features and classifier
    )

    # Test with CIFAR-10 input
    x = torch.randn(4, 3, 32, 32)
    output = model(x)

    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print(f"ResNet with bottleneck test passed. Output shape: {output.shape}")


def test_resnet_without_bottleneck():
    """Test ResNet without bottleneck."""
    # Test ResNet-18 without bottleneck
    model = ResNetWithBottleneck(
        arch="resnet18",
        num_classes=10,
        bottleneck_width=None  # No bottleneck
    )

    # Test with CIFAR-10 input
    x = torch.randn(4, 3, 32, 32)
    output = model(x)

    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print(f"ResNet without bottleneck test passed. Output shape: {output.shape}")


def test_vgg_with_bottleneck():
    """Test VGG with configurable bottleneck."""
    # Create VGG-11 with bottleneck
    model = VGGWithBottleneck(
        arch="vgg11",
        num_classes=10,
        bottleneck_width=256  # Bottleneck between features and classifier
    )

    # Test with CIFAR-10 input
    x = torch.randn(4, 3, 32, 32)
    output = model(x)

    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print(f"VGG with bottleneck test passed. Output shape: {output.shape}")


def test_vgg_without_bottleneck():
    """Test VGG without bottleneck."""
    # Create VGG-11 without bottleneck
    model = VGGWithBottleneck(
        arch="vgg11",
        num_classes=10,
        bottleneck_width=None  # No bottleneck
    )

    # Test with CIFAR-10 input
    x = torch.randn(4, 3, 32, 32)
    output = model(x)

    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print(f"VGG without bottleneck test passed. Output shape: {output.shape}")


def test_resnet_bottleneck_none_vs_int():
    """Test that ResNet behaves differently with and without bottleneck."""
    # Create models with and without bottleneck
    model_with_bottleneck = ResNetWithBottleneck(
        arch="resnet18",
        num_classes=10,
        bottleneck_width=128
    )

    model_without_bottleneck = ResNetWithBottleneck(
        arch="resnet18",
        num_classes=10,
        bottleneck_width=None
    )

    assert model_with_bottleneck.bottleneck is not None, "Bottleneck should not be None"
    assert model_without_bottleneck.bottleneck is None, "Bottleneck should be None"

    # Use a larger batch to avoid batch norm issues
    x = torch.randn(4, 3, 32, 32)

    # Both should produce outputs of the same shape
    out1 = model_with_bottleneck(x)
    out2 = model_without_bottleneck(x)

    assert out1.shape == out2.shape == (4, 10), f"Output shapes should match: {out1.shape}, {out2.shape}"
    print("ResNet bottleneck comparison test passed.")


def test_vgg_bottleneck_none_vs_int():
    """Test that VGG behaves differently with and without bottleneck."""
    # Create models with and without bottleneck
    model_with_bottleneck = VGGWithBottleneck(
        arch="vgg11",
        num_classes=10,
        bottleneck_width=128
    )

    model_without_bottleneck = VGGWithBottleneck(
        arch="vgg11",
        num_classes=10,
        bottleneck_width=None
    )

    assert model_with_bottleneck.bottleneck is not None, "Bottleneck should not be None"
    assert model_without_bottleneck.bottleneck is None, "Bottleneck should be None"

    # Use a larger batch to avoid batch norm issues
    x = torch.randn(4, 3, 32, 32)

    # Both should produce outputs of the same shape
    out1 = model_with_bottleneck(x)
    out2 = model_without_bottleneck(x)

    assert out1.shape == out2.shape == (4, 10), f"Output shapes should match: {out1.shape}, {out2.shape}"
    print("VGG bottleneck comparison test passed.")


if __name__ == "__main__":
    test_resnet_with_bottleneck()
    test_resnet_without_bottleneck()
    test_vgg_with_bottleneck()
    test_vgg_without_bottleneck()
    test_resnet_bottleneck_none_vs_int()
    test_vgg_bottleneck_none_vs_int()
    print("All tests passed!")