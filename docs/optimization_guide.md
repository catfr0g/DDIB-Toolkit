# Optimized Grid Search for DDIB Toolkit

This document explains the optimizations made to improve GPU utilization and reduce RAM usage during grid search.

## Resource Utilization Issues in Original Implementation

1. **Sequential Processing**: Experiments ran one-by-one, not utilizing GPU parallelism
2. **Memory Accumulation**: Results accumulated in memory during the process
3. **GPU Underutilization**: Single GPU usage per experiment with idle time
4. **RAM Overutilization**: Multiple models and data loaders consumed memory

## Optimization Strategies Implemented

### 1. Advanced Optimized Version (`advanced_optimized_grid_search_train.py`)
- **Multiprocessing with Memory-Aware Batching**: Combines parallel execution with memory management
- **Dynamic Concurrency Control**: Automatically determines optimal number of concurrent experiments based on available GPU memory
- **Batch Processing**: Processes experiments in batches to manage memory usage
- **Mixed Precision Training**: Uses 16-bit mixed precision to reduce memory usage and increase throughput
- **Automatic Memory Cleanup**: Regular cleanup of GPU cache and garbage collection

### 2. Batch Optimized Version (`batch_optimized_grid_search_train.py`)
- **Batch Processing**: Processes experiments in configurable batches
- **Memory Monitoring**: Tracks memory usage to prevent overallocation
- **Periodic Cleanup**: Regular cleanup between batches

### 3. Parallel Optimized Version (`optimized_grid_search_train.py`)
- **Process Pool Executor**: Runs multiple experiments in parallel
- **Memory Estimation**: Estimates memory requirements to prevent overallocation

## Configuration Optimizations

The `grid_search_config.yaml` file has been updated with:
- Reduced batch sizes (128 → 64) to save memory
- Reduced epochs (200 → 100) to speed up search
- Mixed precision training enabled by default
- Optimized number of data loader workers

## Usage Instructions

### Using the Advanced Optimized Version (Recommended):
```bash
python -m src.experiments.modeling.advanced_optimized_grid_search_train \
  --config config/grid_search_config.yaml \
  --results-dir results/grid_search \
  --max-concurrent -1 \
  --batch-size 4
```

### Using the Batch Optimized Version:
```bash
python -m src.experiments.modeling.batch_optimized_grid_search_train \
  --config config/grid_search_config.yaml \
  --results-dir results/grid_search \
  --batch-size 4
```

### Using the Parallel Optimized Version:
```bash
python -m src.experiments.modeling.optimized_grid_search_train \
  --config config/grid_search_config.yaml \
  --results-dir results/grid_search \
  --max-concurrent -1
```

## Key Parameters

- `--max-concurrent`: Maximum number of concurrent experiments (-1 for auto-detection based on GPU memory)
- `--batch-size`: Number of experiments to run in each batch
- `--config`: Path to the configuration file
- `--results-dir`: Directory to save results

## Expected Improvements

1. **Better GPU Utilization**: Experiments run in parallel, keeping GPU busy
2. **Reduced RAM Usage**: Memory management prevents accumulation
3. **Faster Execution**: Parallel processing reduces total runtime
4. **Stable Memory Usage**: Batching and cleanup prevent memory overflow


## GPU Optimization Specific Improvements

The `gpu_optimized_grid_search_train.py` version addresses CPU vs GPU utilization issues:

### 1. **Single Experiment Approach for Maximum GPU Utilization**
- Runs one experiment at a time to maximize GPU usage per experiment
- Allows each model to fully utilize GPU compute and memory
- Better for compute-intensive operations like DDIB regularization

### 2. **GPU-Optimized Operations**
- All tensor operations moved to GPU when possible
- Non-blocking tensor transfers for better performance
- Optimized distance calculations in DDIB loss functions
- Enabled cuDNN benchmarking for better performance

### 3. **Improved Data Pipeline**
- Increased num_workers for faster data loading
- Non-blocking data transfers to GPU
- Better memory pinning for faster host-to-device transfers

### 4. **Enhanced DDIB Computation**
- Optimized kernel width calculation using efficient GPU operations
- Replaced inefficient cdist with manual distance computation
- Better memory management during kernel matrix operations

## Running GPU-Optimized Version

To run the GPU-optimized version (recommended for maximum GPU utilization):

```bash
make grid-search-gpu
```

Or directly:
```bash
python -m src.experiments.modeling.gpu_optimized_grid_search_train \
  --config config/grid_search_config.yaml \
  --results-dir results/grid_search \
  --max-concurrent 1
```

This approach prioritizes GPU utilization over parallelism, which is often more effective for compute-intensive deep learning tasks.


## Additional Metrics Added

The enhanced implementation now tracks and logs the following metrics to TensorBoard:

### 1. **Empirical Compression**
- Mutual information estimated using the matrix-based entropy functional
- Measures the information bottleneck effect
- Logged as `Metrics/EmpiricalCompression` in TensorBoard

### 2. **Classification Accuracy** 
- Top-1 accuracy on validation and test sets
- Tracked separately for training and validation phases
- Logged as `Accuracy/Validation` and `Accuracy/Train` in TensorBoard

### 3. **Effective Capacity Utilization**
- Ratio of original loss to log2(W), where W is the number of parameters
- Measures how efficiently the model uses its capacity
- Logged as `Metrics/EffectiveCapacityUtilization` in TensorBoard

These metrics provide deeper insights into the model's learning dynamics and the effectiveness of the DDIB regularization.


## Numerical Stability Fixes

The implementation now includes fixes for numerical stability issues that were causing `linalg.eigh` convergence errors:

### 1. **Matrix Conditioning**
- Added regularization to kernel matrices to improve conditioning
- Implemented fallback to SVD when eigenvalue decomposition fails
- Added clamping of eigenvalues to prevent numerical issues

### 2. **Learning Rate Scheduling**
- Added support for multiple scheduler types: ReduceOnPlateau, Step, and Cosine Annealing
- Configurable through the configuration file
- Helps with training stability and convergence

These improvements ensure more robust training and prevent the "algorithm failed to converge" errors during eigenvalue computation.


## Enhanced Error Handling

The implementation now includes comprehensive error handling that:

### 1. **Detailed Error Logging**
- Captures full traceback information for debugging
- Logs all error details to results files
- Continues execution despite individual experiment failures

### 2. **Robust Result Tracking**
- Failed experiments are recorded with error details
- All metrics are set to appropriate default values for failed experiments
- Results maintain consistent structure regardless of failures

This ensures that even if some experiments fail due to numerical issues, the grid search continues and preserves all results for analysis.