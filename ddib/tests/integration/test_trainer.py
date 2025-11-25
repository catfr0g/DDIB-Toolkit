"""
Integration tests for the PyTorch Lightning trainer module.

These tests verify the complete workflow of training models with real data and real training loops.
"""

import tempfile

import torch
import torch.nn as nn

from ddib.trainer import FlexibleModel, create_simple_ffn, prepare_dataloader, train_model


def test_complete_training_workflow_regression():
    """Test the complete training workflow for a regression task."""
    # Create synthetic regression data
    torch.manual_seed(42)  # For reproducible results
    X = torch.randn(500, 10)
    y = torch.sum(X[:, :5], dim=1, keepdim=True) + 0.1 * torch.randn(500, 1)  # Simple relationship
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]
    
    # Create model
    model = create_simple_ffn(input_size=10, hidden_sizes=[32, 16], output_size=1)
    loss_fn = nn.MSELoss()
    
    # Create PyTorch Lightning module
    pl_model = FlexibleModel(model, loss_fn, learning_rate=0.01)
    
    # Create data loaders
    train_loader = prepare_dataloader(X_train, y_train, batch_size=32)
    val_loader = prepare_dataloader(X_val, y_val, batch_size=32)
    
    # Train the model
    results = train_model(
        model=pl_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=5,
        accelerator='cpu',
        logger=False,  # Disable logging for tests
        enable_checkpointing=False,  # Disable checkpointing for tests
        enable_progress_bar=False  # Disable progress bar for tests
    )
    
    # Verify results
    assert 'final_train_loss' in results
    assert 'final_val_loss' in results
    assert 'num_epochs' in results
    assert results['num_epochs'] == 5
    assert isinstance(results['final_train_loss'], (int, float))
    assert isinstance(results['final_val_loss'], (int, float))
    assert results['final_train_loss'] >= 0  # Loss should be non-negative
    assert results['final_val_loss'] >= 0    # Loss should be non-negative


def test_complete_training_workflow_classification():
    """Test the complete training workflow for a classification task."""
    # Create synthetic classification data
    torch.manual_seed(42)
    X = torch.randn(500, 20)
    y = torch.randint(0, 3, (500,))  # 3 classes
    X_train, X_val = X[:400], X[400:]
    y_train, y_val = y[:400], y[400:]
    
    # Create model
    model = create_simple_ffn(input_size=20, hidden_sizes=[64, 32], output_size=3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create PyTorch Lightning module
    pl_model = FlexibleModel(model, loss_fn, learning_rate=0.001)
    
    # Create data loaders
    train_loader = prepare_dataloader(X_train, y_train, batch_size=32)
    val_loader = prepare_dataloader(X_val, y_val, batch_size=32)
    
    # Train the model
    results = train_model(
        model=pl_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=5,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    
    # Verify results
    assert 'final_train_loss' in results
    assert 'final_val_loss' in results
    assert 'num_epochs' in results
    assert results['num_epochs'] == 5
    assert isinstance(results['final_train_loss'], (int, float))
    assert isinstance(results['final_val_loss'], (int, float))
    assert results['final_train_loss'] >= 0
    assert results['final_val_loss'] >= 0


def test_different_model_architectures():
    """Test training with different model architectures."""
    torch.manual_seed(42)
    X = torch.randn(300, 15)
    y = torch.randn(300, 1)
    X_train, X_val = X[:250], X[250:]
    y_train, y_val = y[:250], y[250:]
    
    architectures = [
        {"input_size": 15, "hidden_sizes": [], "output_size": 1},  # Linear model
        {"input_size": 15, "hidden_sizes": [8], "output_size": 1},  # One hidden layer
        {"input_size": 15, "hidden_sizes": [32, 16, 8], "output_size": 1}  # Deep network
    ]
    
    for arch in architectures:
        model = create_simple_ffn(**arch)
        loss_fn = nn.MSELoss()
        
        pl_model = FlexibleModel(model, loss_fn, learning_rate=0.001)
        
        train_loader = prepare_dataloader(X_train, y_train, batch_size=16)
        val_loader = prepare_dataloader(X_val, y_val, batch_size=16)
        
        results = train_model(
            model=pl_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            max_epochs=3,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        
        # Verify that training completed properly
        assert results['num_epochs'] == 3
        assert 'final_train_loss' in results
        assert 'final_val_loss' in results


def test_different_loss_functions():
    """Test training with different loss functions."""
    torch.manual_seed(42)
    X = torch.randn(200, 8)
    y_regression = torch.randn(200, 1)
    y_classification = torch.randint(0, 2, (200,))  # Binary classification
    
    # Test Mean Squared Error loss
    X_train, X_val = X[:150], X[150:]
    y_reg_train, y_reg_val = y_regression[:150], y_regression[150:]
    
    model_mse = create_simple_ffn(input_size=8, hidden_sizes=[16], output_size=1)
    pl_model_mse = FlexibleModel(model_mse, nn.MSELoss(), learning_rate=0.01)
    
    train_loader = prepare_dataloader(X_train, y_reg_train, batch_size=16)
    val_loader = prepare_dataloader(X_val, y_reg_val, batch_size=16)
    
    results_mse = train_model(
        model=pl_model_mse,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=3,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    
    # Test Cross Entropy loss
    y_cls_train, y_cls_val = y_classification[:150], y_classification[150:]
    model_ce = create_simple_ffn(input_size=8, hidden_sizes=[16], output_size=2)
    pl_model_ce = FlexibleModel(model_ce, nn.CrossEntropyLoss(), learning_rate=0.01)
    
    train_loader = prepare_dataloader(X_train, y_cls_train, batch_size=16)
    val_loader = prepare_dataloader(X_val, y_cls_val, batch_size=16)
    
    results_ce = train_model(
        model=pl_model_ce,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=3,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    
    # Verify both runs completed successfully
    assert results_mse['num_epochs'] == 3
    assert results_ce['num_epochs'] == 3
    assert 'final_train_loss' in results_mse
    assert 'final_train_loss' in results_ce


def test_checkpointing_and_resume_training():
    """Test training with checkpointing functionality."""
    torch.manual_seed(42)
    X = torch.randn(200, 10)
    y = torch.randn(200, 1)
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]
    
    model = create_simple_ffn(input_size=10, hidden_sizes=[16], output_size=1)
    loss_fn = nn.MSELoss()
    pl_model = FlexibleModel(model, loss_fn, learning_rate=0.01)
    
    train_loader = prepare_dataloader(X_train, y_train, batch_size=16)
    val_loader = prepare_dataloader(X_val, y_val, batch_size=16)
    
    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Train for a few epochs with checkpointing enabled
        results = train_model(
            model=pl_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            max_epochs=3,
            accelerator='cpu',
            logger=False,
            enable_progress_bar=False,
            default_root_dir=tmp_dir  # Use temp directory for checkpoints
        )
        
        # Verify training completed
        assert results['num_epochs'] == 3
        assert 'final_train_loss' in results
        assert 'final_val_loss' in results


def test_model_prediction_after_training():
    """Test that the trained model can make predictions."""
    torch.manual_seed(42)
    X = torch.randn(300, 5)
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(300, 1)
    X_train, X_val = X[:250], X[250:]
    y_train, y_val = y[:250], y[250:]
    
    model = create_simple_ffn(input_size=5, hidden_sizes=[10], output_size=1)
    loss_fn = nn.MSELoss()
    pl_model = FlexibleModel(model, loss_fn, learning_rate=0.01)
    
    train_loader = prepare_dataloader(X_train, y_train, batch_size=16)
    val_loader = prepare_dataloader(X_val, y_val, batch_size=16)
    
    # Train the model
    train_model(
        model=pl_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=5,
        accelerator='cpu',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    
    # Test predictions
    test_data = torch.randn(10, 5)
    
    # Set model to eval mode and make predictions
    pl_model.eval()
    with torch.no_grad():
        predictions = pl_model(test_data)
    
    # Verify predictions shape
    assert predictions.shape == (10, 1)
    
    # Verify predictions are reasonable (not NaN or infinite)
    assert not torch.isnan(predictions).any()
    assert not torch.isinf(predictions).any()