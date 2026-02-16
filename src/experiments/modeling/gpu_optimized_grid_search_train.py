"""GPU-Optimized Grid search training script for VGG/ResNet models with bottleneck width using DDIB.

This version focuses on maximizing GPU utilization by optimizing data loading,
model placement, and computation distribution.
"""

import os
from pathlib import Path
import random
import json
from typing import Literal, Optional
import yaml
from itertools import product
import gc
import psutil
import math
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue

from loguru import logger
import numpy as np
import torch
from torch import nn
import typer

from src.ddib.models import ResNetWithBottleneck, VGGWithBottleneck
from src.ddib.trainer import IBModel, train_model
from src.experiments.config import MODELS_DIR, RAW_DATA_DIR
from src.experiments.dataset_loading import load_cifar10_dataset


app = typer.Typer()


def get_available_memory_gb():
    """Get available system memory in GB"""
    memory = psutil.virtual_memory()
    return memory.available / (1024**3)


def seed_all(seed: int = 42):
    """Function to fix seed"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Use warn_only=True to allow non-deterministic operations while still warning about them
    torch.backends.cudnn.deterministic = False  # Set to False for better performance
    torch.backends.cudnn.benchmark = True  # Enable benchmarking for better performance
    # Don't use deterministic algorithms globally as it breaks some CUDA operations
    # torch.use_deterministic_algorithms(True, warn_only=True)


def run_single_training_gpu_optimized(
    model_arch: str,
    bottleneck_width: int,
    beta: float,
    seed: int,
    config: dict
):
    """Run a single training experiment with GPU-optimized approach."""
    # Ensure CUDA is available and set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for better performance
    else:
        device = torch.device("cpu")
    
    # Fix seeds for reproducibility
    seed_all(seed)

    logger.info(
        f"Starting training with model: {model_arch}, bottleneck width: {bottleneck_width}, "
        f"beta: {beta}, seed: {seed}"
    )

    try:
        # Load CIFAR-10 dataset with train/validation/test splits
        logger.info("Loading CIFAR-10 dataset...")
        train_loader, val_loader, test_loader = load_cifar10_dataset(
            data_dir=Path(config.get('data_dir', str(RAW_DATA_DIR))),
            train_batch_size=config['training_params']['train_batch_size'],
            val_batch_size=config['training_params']['val_batch_size'],
            test_batch_size=config['training_params']['test_batch_size'],
            train_val_split_ratio=config['training_params']['train_val_split_ratio'],
            num_workers=config['training_params'].get('num_workers', 4),  # Increase for better data loading
            download=config['training_params'].get('download', True),
        )
        logger.success("Dataset loaded successfully!")

        # Create model based on architecture
        if "resnet" in model_arch.lower():
            model: nn.Module = ResNetWithBottleneck(
                arch=model_arch,
                num_classes=10,  # CIFAR-10 has 10 classes
                bottleneck_width=bottleneck_width,
            )
        elif "vgg" in model_arch.lower():
            model = VGGWithBottleneck(
                arch=model_arch,
                num_classes=10,  # CIFAR-10 has 10 classes
                bottleneck_width=bottleneck_width,
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_arch}")

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Move model to device immediately
        model = model.to(device)

        # Create the DDIB model wrapper
        ddib_model = IBModel(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),  # Base loss function
            layer_to_optimize="bottleneck",  # Always optimize the bottleneck layer
            beta=beta,  # Beta parameter for DDIB regularization
            learning_rate=config['training_params']['learning_rate'],
            optimizer_class=torch.optim.Adam,
            weight_decay=config['training_params']['weight_decay'],
            use_scheduler=config['training_params'].get('use_scheduler', True),
            scheduler_type=config['training_params'].get('scheduler_type', 'reduceonplateau'),
        )

        # Move the entire model to GPU if available
        if torch.cuda.is_available():
            ddib_model = ddib_model.to(device)

        # Train the model using the DDIB trainer with GPU optimizations
        logger.info("Starting model training...")
        train_results = train_model(
            model=ddib_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            max_epochs=config['training_params']['num_epochs'],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            log_dir="tb_logs",
            experiment_name=f"{model_arch}_bottleneck_{bottleneck_width}_beta_{beta}_seed_{seed}",
            precision=config['training_params'].get('precision', '16-mixed'),  # Use mixed precision
            strategy="auto",  # Use appropriate strategy for single GPU
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # Get the trained underlying model
        trained_model = ddib_model.model

        logger.success("Training completed successfully!")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        trained_model.eval()
        correct = 0
        total = 0
        
        # Move to eval mode and disable gradient computation for memory efficiency
        with torch.no_grad():
            for data in test_loader:
                # Move data to GPU immediately
                images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                # Ensure the model is on the same device as the data
                trained_model = trained_model.to(device)
                outputs = trained_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f"Test Accuracy: {accuracy:.2f}%")

        # Clean up GPU memory
        del ddib_model, model, trained_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

        # Return results for this experiment
        return {
            'model_arch': model_arch,
            'bottleneck_width': bottleneck_width,
            'beta': beta,
            'seed': seed,
            'test_accuracy': accuracy,
            'final_train_loss': train_results.get('final_train_loss', float('inf')),
            'final_val_loss': train_results.get('final_val_loss', float('inf')),
            'final_train_acc': train_results.get('final_train_acc', 0.0),
            'final_val_acc': train_results.get('final_val_acc', 0.0),
            'final_empirical_compression': train_results.get('final_empirical_compression', 0.0),
            'final_train_empirical_compression': train_results.get('final_train_empirical_compression', 0.0),
            'final_effective_capacity_utilization': train_results.get('final_effective_capacity_utilization', 0.0),
            'final_train_effective_capacity_utilization': train_results.get('final_train_effective_capacity_utilization', 0.0)
        }
    
    except Exception as e:
        logger.error(f"Experiment failed: model={model_arch}, bottleneck_width={bottleneck_width}, "
                    f"beta={beta}, seed={seed}, error: {str(e)}")
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        import traceback
        return {
            'model_arch': model_arch,
            'bottleneck_width': bottleneck_width,
            'beta': beta,
            'seed': seed,
            'error': str(e),
            'error_traceback': traceback.format_exc(),
            'test_accuracy': 0.0,
            'final_train_loss': float('inf'),
            'final_val_loss': float('inf'),
            'final_train_acc': 0.0,
            'final_val_acc': 0.0,
            'final_empirical_compression': 0.0,
            'final_train_empirical_compression': 0.0,
            'final_effective_capacity_utilization': 0.0,
            'final_train_effective_capacity_utilization': 0.0
        }


def run_single_training_gpu_optimized_wrapper(args):
    """Wrapper function for multiprocessing to unpack arguments"""
    return run_single_training_gpu_optimized(*args)


def estimate_gpu_memory_usage(model_arch: str, bottleneck_width: int) -> float:
    """Estimate GPU memory usage for a given model configuration"""
    # Rough estimation based on model parameters and bottleneck width
    if "resnet" in model_arch.lower():
        base_params = 11000000  # ~11M params for ResNet18
    elif "vgg" in model_arch.lower():
        base_params = 9000000   # ~9M params for VGG11
    else:
        base_params = 10000000  # Default estimate
    
    # Add bottleneck parameters
    bottleneck_params = 512 * bottleneck_width  # Approximation for feature map dimensions
    total_params = base_params + bottleneck_params
    
    # Estimate memory usage (parameters + activations + optimizer states)
    # Typically 4 bytes per parameter for FP32, plus activations and optimizer overhead
    estimated_gb = (total_params * 4 * 3) / (1024**3)  # 3x for parameters + gradients + optimizer
    return estimated_gb


@app.command()
def main(
    config_path: Path = typer.Option(
        "config/grid_search_config.yaml",
        "--config", "-c",
        help="Path to the grid search configuration file"
    ),
    results_dir: Path = typer.Option(
        "results/grid_search",
        "--results-dir", "-r",
        help="Directory to save grid search results"
    ),
    max_concurrent: int = typer.Option(
        1,  # Start with 1 to ensure GPU is properly utilized
        "--max-concurrent", "-m",
        help="Maximum number of concurrent experiments (set to 1 for best GPU utilization)"
    )
):
    """
    Perform GPU-optimized grid search over hyperparameters using DDIB.
    Focuses on maximizing GPU utilization while managing memory efficiently.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Extract hyperparameter ranges
    model_archs = config['model_archs']
    bottleneck_widths = config['bottleneck_widths']
    betas = config['betas']
    seeds = config['seeds']

    logger.info(f"Starting grid search with:")
    logger.info(f"  Models: {model_archs}")
    logger.info(f"  Bottleneck widths: {bottleneck_widths}")
    logger.info(f"  Betas: {betas}")
    logger.info(f"  Seeds: {seeds}")

    # Calculate total experiments
    total_experiments = len(model_archs) * len(bottleneck_widths) * len(betas) * len(seeds)
    logger.info(f"Total experiments to run: {total_experiments}")

    # For GPU optimization, it's often better to run experiments sequentially
    # to maximize GPU utilization per experiment rather than running many in parallel
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU memory available: {gpu_memory_gb:.2f} GB")
        
        # For GPU optimization, run one experiment at a time to maximize GPU usage per experiment
        if max_concurrent != 1:
            logger.info("For optimal GPU utilization, consider running with --max-concurrent 1")
    else:
        logger.info("CUDA not available, running on CPU")

    # Generate all experiment combinations
    experiment_combinations = list(product(model_archs, bottleneck_widths, betas, seeds))
    
    # Store results
    results = []
    completed_experiments = 0

    # For GPU optimization, run experiments sequentially to maximize GPU utilization per experiment
    for i, (model_arch, bottleneck_width, beta, seed) in enumerate(experiment_combinations):
        logger.info(f"Running experiment {i+1}/{total_experiments}: "
                   f"model={model_arch}, bottleneck_width={bottleneck_width}, "
                   f"beta={beta}, seed={seed}")

        try:
            result = run_single_training_gpu_optimized(
                model_arch=model_arch,
                bottleneck_width=bottleneck_width,
                beta=beta,
                seed=seed,
                config=config
            )
            results.append(result)
            completed_experiments += 1
            
            logger.info(f"Completed experiment {completed_experiments}/{total_experiments}: "
                       f"model={result['model_arch']}, bottleneck_width={result['bottleneck_width']}, "
                       f"beta={result['beta']}, seed={result['seed']}, "
                       f"accuracy={result.get('test_accuracy', 0):.2f}%")
            
            # Save intermediate results periodically
            if completed_experiments % max(1, total_experiments // 10) == 0:  # Every 10%
                results_file = results_dir / f"grid_search_results_intermediate_{completed_experiments}.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
            # Memory cleanup after each experiment
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Experiment failed: model={model_arch}, bottleneck_width={bottleneck_width}, "
                        f"beta={beta}, seed={seed}, error: {str(e)}")
            import traceback
            error_details = {
                'model_arch': model_arch,
                'bottleneck_width': bottleneck_width,
                'beta': beta,
                'seed': seed,
                'error': str(e),
                'error_traceback': traceback.format_exc(),
                'test_accuracy': 0.0,
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'final_train_acc': 0.0,
                'final_val_acc': 0.0,
                'final_empirical_compression': 0.0,
                'final_train_empirical_compression': 0.0,
                'final_effective_capacity_utilization': 0.0,
                'final_train_effective_capacity_utilization': 0.0
            }
            results.append(error_details)
            completed_experiments += 1

    # Save final results
    results_file = results_dir / "grid_search_results_final.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.success(f"Grid search completed! Results saved to {results_file}")

    # Print summary
    successful_results = [r for r in results if 'error' not in r]
    if successful_results:
        best_result = max(successful_results, key=lambda x: x.get('test_accuracy', 0), default=None)
        if best_result:
            logger.info(f"Best result: model={best_result['model_arch']}, "
                       f"bottleneck_width={best_result['bottleneck_width']}, "
                       f"beta={best_result['beta']}, "
                       f"seed={best_result['seed']}, "
                       f"accuracy={best_result['test_accuracy']:.2f}%")

    return results


if __name__ == "__main__":
    app()