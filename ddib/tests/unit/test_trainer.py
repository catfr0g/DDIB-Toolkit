"""
Unit tests for the PyTorch Lightning trainer module.

Each test is atomic and focuses on testing a single function or class method.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ddib.trainer import FlexibleModel, create_simple_ffn, prepare_dataloader, train_model


def test_flexible_model_initialization():
    """Test initializing the FlexibleModel with different parameters."""
    # Create a simple model
    model = nn.Linear(10, 1)
    loss_fn = nn.MSELoss()
    
    # Test basic initialization
    flex_model = FlexibleModel(
        model=model,
        loss_fn=loss_fn
    )
    
    assert flex_model.model == model
    assert flex_model.loss_fn == loss_fn
    assert flex_model.val_loss_fn == loss_fn
    assert flex_model.learning_rate == 1e-3


def test_flexible_model_forward():
    """Test the forward pass of the FlexibleModel."""
    model = nn.Linear(10, 1)
    loss_fn = nn.MSELoss()
    flex_model = FlexibleModel(model, loss_fn)
    
    # Create dummy input
    x = torch.randn(5, 10)
    output = flex_model(x)
    
    assert output.shape == (5, 1)
    assert isinstance(output, torch.Tensor)


def test_flexible_model_training_step():
    """Test the training step of the FlexibleModel."""
    model = nn.Linear(10, 1)
    loss_fn = nn.MSELoss()
    flex_model = FlexibleModel(model, loss_fn)
    
    # Create dummy batch
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    batch = (x, y)
    
    # Run training step
    result = flex_model.training_step(batch, 0)
    
    assert 'loss' in result
    assert isinstance(result['loss'], torch.Tensor)
    assert result['loss'].shape == torch.Size([])  # scalar


def test_flexible_model_validation_step():
    """Test the validation step of the FlexibleModel."""
    model = nn.Linear(10, 1)
    loss_fn = nn.MSELoss()
    flex_model = FlexibleModel(model, loss_fn)
    
    # Create dummy batch
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    batch = (x, y)
    
    # Run validation step
    result = flex_model.validation_step(batch, 0)
    
    assert 'val_loss' in result
    assert isinstance(result['val_loss'], torch.Tensor)
    assert result['val_loss'].shape == torch.Size([])  # scalar


def test_flexible_model_different_loss_functions():
    """Test FlexibleModel with different train and validation loss functions."""
    model = nn.Linear(10, 2)  # 2 outputs for classification
    train_loss = nn.CrossEntropyLoss()
    val_loss = nn.MSELoss()
    
    flex_model = FlexibleModel(
        model=model,
        loss_fn=train_loss,
        val_loss_fn=val_loss
    )
    
    assert flex_model.loss_fn == train_loss
    assert flex_model.val_loss_fn == val_loss


def test_prepare_dataloader_basic():
    """Test basic functionality of prepare_dataloader function."""
    # Create dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Create dataloader
    dataloader = prepare_dataloader(X, y, batch_size=32)
    
    # Check that it's a DataLoader
    assert isinstance(dataloader, DataLoader)
    
    # Check that we can get a batch
    batch = next(iter(dataloader))
    x_batch, y_batch = batch
    assert x_batch.shape[1:] == X.shape[1:]  # Feature dimensions match
    assert y_batch.shape[1:] == y.shape[1:]   # Target dimensions match
    assert len(x_batch) <= 32  # Batch size is respected


def test_prepare_dataloader_shuffle():
    """Test that the shuffle parameter works in prepare_dataloader."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Create dataloaders with and without shuffling
    dataloader_shuffled = prepare_dataloader(X, y, batch_size=32, shuffle=True)
    dataloader_unshuffled = prepare_dataloader(X, y, batch_size=32, shuffle=False)
    
    assert isinstance(dataloader_shuffled, DataLoader)
    assert isinstance(dataloader_unshuffled, DataLoader)


def test_prepare_dataloader_batch_size():
    """Test that different batch sizes work in prepare_dataloader."""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # Test different batch sizes
    for batch_size in [16, 32, 64]:
        dataloader = prepare_dataloader(X, y, batch_size=batch_size)
        # Get one batch to verify
        batch = next(iter(dataloader))
        x_batch, y_batch = batch
        assert len(x_batch) <= batch_size


def test_create_simple_ffn_basic():
    """Test creating a simple feedforward network."""
    model = create_simple_ffn(
        input_size=10,
        hidden_sizes=[20, 15],
        output_size=1,
        dropout_rate=0.1
    )
    
    # Check that it's a Sequential model
    assert isinstance(model, nn.Sequential)
    
    # Check that the model has the expected number of layers
    # Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear (output)
    expected_layers = 3 * 2 + 1  # 3 hidden layers: Linear+ReLU+Dropout, 1 output layer
    assert len(model) == expected_layers
    
    # Test forward pass
    x = torch.randn(5, 10)
    output = model(x)
    assert output.shape == (5, 1)


def test_create_simple_ffn_empty_hidden():
    """Test creating a feedforward network with no hidden layers (linear model)."""
    model = create_simple_ffn(
        input_size=10,
        hidden_sizes=[],
        output_size=1
    )
    
    # Should just be a single linear layer
    assert isinstance(model, nn.Sequential)
    assert len(model) == 1  # Just the output layer
    assert isinstance(model[0], nn.Linear)
    
    x = torch.randn(5, 10)
    output = model(x)
    assert output.shape == (5, 1)


def test_create_simple_ffn_different_configurations():
    """Test creating different FFN configurations."""
    configs = [
        {"input_size": 5, "hidden_sizes": [10], "output_size": 2},
        {"input_size": 20, "hidden_sizes": [32, 16, 8], "output_size": 1},
        {"input_size": 7, "hidden_sizes": [14, 7, 5, 3], "output_size": 4}
    ]
    
    for config in configs:
        model = create_simple_ffn(**config)
        x = torch.randn(3, config["input_size"])
        output = model(x)
        assert output.shape == (3, config["output_size"])


class TestTrainModel:
    """Test the train_model function."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Create a simple model for testing
        self.model = nn.Linear(10, 1)
        self.train_loss = nn.MSELoss()
        self.flex_model = FlexibleModel(self.model, self.train_loss)
        
        # Create dummy data
        X_train = torch.randn(100, 10)
        y_train = torch.randn(100, 1)
        X_val = torch.randn(20, 10)
        y_val = torch.randn(20, 1)
        
        self.train_loader = prepare_dataloader(X_train, y_train, batch_size=16)
        self.val_loader = prepare_dataloader(X_val, y_val, batch_size=16)
    
    def test_train_model_basic(self):
        """Test basic functionality of train_model."""
        results = train_model(
            model=self.flex_model,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            max_epochs=2,  # Small number for testing
            accelerator='cpu'  # Use CPU to avoid device issues
        )
        
        # Check that results dictionary has the expected keys
        assert 'final_train_loss' in results
        assert 'final_val_loss' in results
        assert 'num_epochs' in results
        
        # Check that losses are numbers
        assert isinstance(results['final_train_loss'], (int, float))
        assert isinstance(results['final_val_loss'], (int, float))
        assert results['num_epochs'] == 2
    
    def test_train_model_different_epochs(self):
        """Test train_model with different epoch counts."""
        for epochs in [1, 3]:
            results = train_model(
                model=self.flex_model,
                train_dataloader=self.train_loader,
                val_dataloader=self.val_loader,
                max_epochs=epochs,
                accelerator='cpu'
            )
            
            assert results['num_epochs'] == epochs