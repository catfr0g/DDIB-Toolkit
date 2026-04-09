"""Shared analysis utilities for experiment results."""

from experiments.analysis.data_loader import (
	load_grid_search_results,
	load_with_baseline,
	prepare_regression_data,
)
from experiments.analysis.plotting import (
	BOTTLENECK_CMAP,
	make_heatmap,
	plot_accuracy_vs_beta_with_gradient,
	plot_bootstrap_ci_vs_beta,
	plot_bootstrap_ci_vs_beta_with_baseline,
	plot_compression_vs_beta_with_gradient,
	plot_metric_vs_beta_error_bars,
	save_and_close,
	scatter_with_colorbar,
)
from experiments.analysis.statistics import (
	add_engineered_features,
	analyze_beta_metric_relationship,
	bootstrap_ci,
	compute_beta_group_statistics,
	compute_correlation_matrix,
	find_significant_beta_regions,
	gradient_boosting_analysis,
	linear_regression_analysis,
	pairwise_beta_significance,
	random_forest_analysis,
)

__all__ = [
	# Data loading
	'load_grid_search_results',
	'load_with_baseline',
	'prepare_regression_data',
	# Plotting
	'BOTTLENECK_CMAP',
	'make_heatmap',
	'plot_accuracy_vs_beta_with_gradient',
	'plot_bootstrap_ci_vs_beta',
	'plot_bootstrap_ci_vs_beta_with_baseline',
	'plot_compression_vs_beta_with_gradient',
	'plot_metric_vs_beta_error_bars',
	'save_and_close',
	'scatter_with_colorbar',
	# Statistics
	'add_engineered_features',
	'analyze_beta_metric_relationship',
	'bootstrap_ci',
	'compute_beta_group_statistics',
	'compute_correlation_matrix',
	'find_significant_beta_regions',
	'gradient_boosting_analysis',
	'linear_regression_analysis',
	'pairwise_beta_significance',
	'random_forest_analysis',
]
