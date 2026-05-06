"""Plotting utilities for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Default colormap
BOTTLENECK_CMAP = 'plasma'


def plot_accuracy_vs_beta_with_gradient(
	ax: plt.Axes,
	data: pd.DataFrame,
	title: str,
	show_legend: bool = True,
	cmap: str = BOTTLENECK_CMAP,
	baseline_data: pd.DataFrame | None = None,
) -> None:
	"""Plot accuracy vs beta with lines colored by bottleneck width.

	Parameters
	----------
	ax :
	    Matplotlib axis to plot on.
	data :
	    DataFrame containing ``beta``, ``test_accuracy``, and ``bottleneck_width``.
	title :
	    Plot title.
	show_legend :
	    Whether to show the legend.
	cmap :
	    Colormap name for bottleneck width gradient.
	baseline_data :
	    Optional DataFrame with beta=0 baseline results. If provided,
	    baseline accuracy for each bottleneck width is shown as a
	    horizontal dashed line across the plot.
	"""
	widths = sorted(data['bottleneck_width'].unique()) if len(data) > 0 else []
	if not widths:
		if baseline_data is not None and len(baseline_data) > 0:
			bw = sorted(baseline_data['bottleneck_width'].unique())
			norm = LogNorm(min(bw), max(bw))
			cmap_obj = plt.get_cmap(cmap)
			for width in bw:
				width_baseline = baseline_data[baseline_data['bottleneck_width'] == width]
				if len(width_baseline) > 0:
					mean_acc = width_baseline['test_accuracy'].mean()
					color = cmap_obj(norm(width))
					ax.axhline(
						y=mean_acc,
						color=color,
						linestyle='--',
						linewidth=2,
						alpha=0.7,
						zorder=3,
					)
			if show_legend:
				baseline_handle = plt.Line2D(
					[0],
					[0],
					color='black',
					linestyle='--',
					linewidth=2,
					alpha=0.7,
					label='Baseline (β=0)',
				)
				ax.legend(handles=[baseline_handle], loc='best')
			ax.set_xscale('log')
			ax.set_xlabel('Beta (log scale)')
			ax.set_ylabel('Test Accuracy (%)')
			ax.set_title(title)
			ax.grid(True, alpha=0.3)
		return
	norm = LogNorm(min(widths), max(widths))
	cmap_obj = plt.get_cmap(cmap)

	all_betas: list[float] = []
	all_accuracies: list[float] = []
	line_handles: list[plt.Line2D] = []

	for width in widths:
		subset = data[data['bottleneck_width'] == width].groupby('beta')['test_accuracy'].mean()
		points = np.array([subset.index.values, subset.values]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		lc = LineCollection(segments, cmap=cmap_obj, norm=norm, linewidth=2.5)
		lc.set_array(np.full(len(segments), width))
		ax.add_collection(lc)

		all_betas.extend(subset.index.values)
		all_accuracies.extend(subset.values)

	for width in widths:
		color = cmap_obj(norm(width))
		line_handles.append(
			plt.Line2D([0], [0], color=color, linewidth=2.5, label=f'Width: {width}')
		)

	# Add baseline (beta=0) as horizontal dashed lines
	baseline_handle = None
	if baseline_data is not None and len(baseline_data) > 0:
		for width in widths:
			width_baseline = baseline_data[baseline_data['bottleneck_width'] == width]
			if len(width_baseline) > 0:
				mean_acc = width_baseline['test_accuracy'].mean()
				color = cmap_obj(norm(width))
				ax.axhline(
					y=mean_acc,
					color=color,
					linestyle='--',
					linewidth=2,
					alpha=0.7,
					zorder=3,
				)
		# Single legend entry for all baseline lines
		baseline_handle = plt.Line2D(
			[0],
			[0],
			color='black',
			linestyle='--',
			linewidth=2,
			alpha=0.7,
			label='Baseline (β=0)',
		)

	if show_legend:
		all_handles = list(line_handles)
		if baseline_handle is not None:
			all_handles.append(baseline_handle)
		if all_handles:
			ax.legend(handles=all_handles, loc='best')

	ax.set_xscale('log')
	ax.set_xlabel('Beta (log scale)')
	ax.set_ylabel('Test Accuracy (%)')
	ax.set_title(title)
	ax.grid(True, alpha=0.3)

	if all_betas and all_accuracies:
		ax.set_xlim(min(all_betas) * 0.4, max(all_betas) * 1.1)
		ax.set_ylim(min(all_accuracies) - 1, max(all_accuracies) + 1)


def plot_compression_vs_beta_with_gradient(
	ax: plt.Axes,
	data: pd.DataFrame,
	title: str,
	show_legend: bool = True,
	cmap: str = BOTTLENECK_CMAP,
	baseline_data: pd.DataFrame | None = None,
) -> None:
	"""Plot empirical compression vs beta with lines colored by bottleneck width.

	Parameters
	----------
	ax :
	    Matplotlib axis to plot on.
	data :
	    DataFrame containing ``beta``, ``final_empirical_compression``, and ``bottleneck_width``.
	title :
	    Plot title.
	show_legend :
	    Whether to show the legend.
	cmap :
	    Colormap name for bottleneck width gradient.
	baseline_data :
	    Optional DataFrame with beta=0 baseline results. If provided,
	    baseline compression for each bottleneck width is shown as a
	    horizontal dashed line across the plot.
	"""
	widths = sorted(data['bottleneck_width'].unique()) if len(data) > 0 else []
	if not widths:
		if baseline_data is not None and len(baseline_data) > 0:
			bw = sorted(baseline_data['bottleneck_width'].unique())
			norm = LogNorm(min(bw), max(bw))
			cmap_obj = plt.get_cmap(cmap)
			for width in bw:
				width_baseline = baseline_data[baseline_data['bottleneck_width'] == width]
				if len(width_baseline) > 0:
					mean_comp = width_baseline['final_empirical_compression'].mean()
					color = cmap_obj(norm(width))
					ax.axhline(
						y=mean_comp,
						color=color,
						linestyle='--',
						linewidth=2,
						alpha=0.7,
						zorder=3,
					)
			if show_legend:
				baseline_handle = plt.Line2D(
					[0],
					[0],
					color='black',
					linestyle='--',
					linewidth=2,
					alpha=0.7,
					label='Baseline (β=0)',
				)
				ax.legend(handles=[baseline_handle], loc='best')
			ax.set_xscale('log')
			ax.set_xlabel('Beta (log scale)')
			ax.set_ylabel('Empirical Compression')
			ax.set_title(title)
			ax.grid(True, alpha=0.3)
		return
	norm = LogNorm(min(widths), max(widths))
	cmap_obj = plt.get_cmap(cmap)

	all_betas: list[float] = []
	all_compressions: list[float] = []
	line_handles: list[plt.Line2D] = []

	for width in widths:
		subset = (
			data[data['bottleneck_width'] == width]
			.groupby('beta')['final_empirical_compression']
			.mean()
		)
		points = np.array([subset.index.values, subset.values]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		lc = LineCollection(segments, cmap=cmap_obj, norm=norm, linewidth=2.5)
		lc.set_array(np.full(len(segments), width))
		ax.add_collection(lc)

		all_betas.extend(subset.index.values)
		all_compressions.extend(subset.values)

	for width in widths:
		color = cmap_obj(norm(width))
		line_handles.append(
			plt.Line2D([0], [0], color=color, linewidth=2.5, label=f'Width: {width}')
		)

	# Add baseline (beta=0) as horizontal dashed lines
	baseline_handle = None
	if baseline_data is not None and len(baseline_data) > 0:
		for width in widths:
			width_baseline = baseline_data[baseline_data['bottleneck_width'] == width]
			if len(width_baseline) > 0:
				mean_comp = width_baseline['final_empirical_compression'].mean()
				color = cmap_obj(norm(width))
				ax.axhline(
					y=mean_comp,
					color=color,
					linestyle='--',
					linewidth=2,
					alpha=0.7,
					zorder=3,
				)
		# Single legend entry for all baseline lines
		baseline_handle = plt.Line2D(
			[0],
			[0],
			color='black',
			linestyle='--',
			linewidth=2,
			alpha=0.7,
			label='Baseline (β=0)',
		)

	if show_legend:
		all_handles = list(line_handles)
		if baseline_handle is not None:
			all_handles.append(baseline_handle)
		if all_handles:
			ax.legend(handles=all_handles, loc='best')

	ax.set_xscale('log')
	ax.set_xlabel('Beta (log scale)')
	ax.set_ylabel('Empirical Compression')
	ax.set_title(title)
	ax.grid(True, alpha=0.3)

	if all_betas and all_compressions:
		ax.set_xlim(min(all_betas) * 0.4, max(all_betas) * 1.1)
		ax.set_ylim(min(all_compressions) - 0.5, max(all_compressions) + 0.5)


def plot_metric_vs_beta_error_bars(
	ax: plt.Axes,
	data: pd.DataFrame,
	metric_col: str,
	title: str,
	ylabel: str,
	baseline_data: pd.DataFrame | None = None,
) -> None:
	"""Plot metric vs beta with error bars (±1 std across seeds and widths).

	Aggregates across all bottleneck widths. Shows mean ± std for each beta value.

	Parameters
	----------
	ax :
	    Matplotlib axis to plot on.
	data :
	    DataFrame containing ``beta`` and the metric column.
	metric_col :
	    Column name for the metric (e.g. ``test_accuracy``).
	title :
	    Plot title.
	ylabel :
	    Y-axis label.
	baseline_data :
	    Optional DataFrame with beta=0 baseline results. If provided,
	    baseline mean ± std is shown as a horizontal dashed line with shaded region.
	"""
	# Aggregate across all widths and seeds: group by beta
	grouped = data.groupby('beta')[metric_col]
	stats = grouped.agg(['mean', 'std', 'count']).reset_index()
	# Filter out beta=0 from the main plot (it's shown as baseline)
	stats_plot = stats[stats['beta'] > 0].copy()

	# Plot error bars
	if len(stats_plot) == 0:
		ax.text(0.5, 0.5, 'No data (beta > 0)', ha='center', va='center', fontsize=14)
		ax.set_xlabel('Beta (log scale)')
		ax.set_ylabel(ylabel)
		ax.set_title(title)
		return

	ax.errorbar(
		stats_plot['beta'],
		stats_plot['mean'],
		yerr=stats_plot['std'],
		fmt='o-',
		color='#1f77b4',
		capsize=5,
		capthick=1.5,
		linewidth=2,
		markersize=6,
		alpha=0.8,
		zorder=3,
	)

	# Add baseline if provided
	if baseline_data is not None and len(baseline_data) > 0:
		bl_mean = baseline_data[metric_col].mean()
		bl_std = baseline_data[metric_col].std()
		# Horizontal dashed line at baseline mean
		ax.axhline(y=bl_mean, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
		# Shaded region for baseline ±1 std
		ax.axhspan(bl_mean - bl_std, bl_mean + bl_std, color='red', alpha=0.1, zorder=1)
		ax.text(
			stats_plot['beta'].min() * 1.1,
			bl_mean + bl_std * 0.3,
			f'Baseline: {bl_mean:.2f} ± {bl_std:.2f}',
			color='red',
			fontsize=9,
			fontweight='bold',
			bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
			zorder=4,
		)

	ax.set_xscale('log')
	ax.set_xlabel('Beta (log scale)')
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(True, alpha=0.3)

	if len(stats_plot) > 1:
		ax.set_xlim(stats_plot['beta'].min() * 0.7, stats_plot['beta'].max() * 1.3)
	elif len(stats_plot) == 1:
		beta_val = stats_plot['beta'].values[0]
		ax.set_xlim(beta_val * 0.5, beta_val * 2.0)


def save_and_close(fig: plt.Figure, output_dir: Path, filename: str, dpi: int = 150) -> None:
	"""Save figure and close all figures to avoid blocking."""
	fig.savefig(output_dir / filename, dpi=dpi)
	print(f'Saved: {filename}')
	plt.close('all')


def plot_bootstrap_ci_vs_beta(
	ax: plt.Axes,
	group_stats: pd.DataFrame,
	metric_col: str,
	title: str,
	ylabel: str,
	significant_regions: pd.DataFrame | None = None,
) -> None:
	"""Plot bootstrap confidence intervals for each beta value.

	Parameters
	----------
	ax :
	    Matplotlib axis to plot on.
	group_stats :
	    DataFrame from ``compute_beta_group_statistics`` with columns:
	    beta, mean, ci_lower, ci_upper, std, n.
	metric_col :
	    Name of the metric column (for labeling).
	title :
	    Plot title.
	ylabel :
	    Y-axis label.
	significant_regions :
	    Optional DataFrame from ``find_significant_beta_regions`` to
	    highlight significant transitions.
	"""
	# Filter out beta=0 for log scale
	gs = group_stats[group_stats['beta'] > 0].copy()

	ax.errorbar(
		gs['beta'],
		gs['mean'],
		yerr=[gs['mean'] - gs['ci_lower'], gs['ci_upper'] - gs['mean']],
		fmt='o-',
		color='#1f77b4',
		capsize=5,
		capthick=1.5,
		linewidth=2,
		markersize=6,
		alpha=0.8,
		zorder=3,
	)
	ax.fill_between(
		gs['beta'],
		gs['ci_lower'],
		gs['ci_upper'],
		alpha=0.15,
		color='#1f77b4',
		zorder=2,
	)

	# Highlight significant transitions
	if significant_regions is not None and len(significant_regions) > 0:
		sig = significant_regions[significant_regions['significant']]
		for _, row in sig.iterrows():
			ax.axvline(
				x=row['beta_to'],
				color='red',
				linestyle='--',
				alpha=0.5,
				linewidth=1,
				zorder=1,
			)
		# Add legend entry for significant transitions
		ax.axvline(
			x=float('nan'),
			color='red',
			linestyle='--',
			alpha=0.5,
			linewidth=1,
			label='Significant transition (p<0.05)',
		)

	ax.set_xscale('log')
	ax.set_xlabel('Beta (log scale)')
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best', fontsize=9)

	if len(gs) > 1:
		ax.set_xlim(gs['beta'].min() * 0.7, gs['beta'].max() * 1.3)


def plot_bootstrap_ci_vs_beta_with_baseline(
	ax: plt.Axes,
	group_stats: pd.DataFrame,
	baseline_stats: pd.DataFrame,
	metric_col: str,
	title: str,
	ylabel: str,
	alpha: float = 0.05,
) -> None:
	"""Plot bootstrap CIs for each beta value INCLUDING baseline, with significance markers.

	Shows which beta regions are significantly different from baseline.

	Parameters
	----------
	ax :
	    Matplotlib axis to plot on.
	group_stats :
	    DataFrame from ``compute_beta_group_statistics`` with columns:
	    beta, mean, ci_lower, ci_upper, std, n (beta > 0 only).
	baseline_stats :
	    DataFrame with baseline (beta=0) statistics: mean, ci_lower, ci_upper, std.
	metric_col :
	    Name of the metric column (for labeling).
	title :
	    Plot title.
	ylabel :
	    Y-axis label.
	alpha :
	    Significance level for pairwise tests.
	"""
	# Plot baseline as a horizontal band
	bl_mean = baseline_stats['mean']
	bl_ci_lower = baseline_stats['ci_lower']
	bl_ci_upper = baseline_stats['ci_upper']

	ax.axhspan(bl_ci_lower, bl_ci_upper, color='gray', alpha=0.15, zorder=1)
	ax.axhline(
		y=bl_mean,
		color='gray',
		linestyle='--',
		linewidth=2,
		alpha=0.7,
		zorder=2,
		label=f'Baseline (beta=0): {bl_mean:.2f}',
	)

	# Plot each beta region with CI
	gs = group_stats[group_stats['beta'] > 0].copy()

	# Determine which beta regions are significantly different from baseline
	# by checking if CI excludes baseline CI
	sig_mask = (gs['ci_lower'] > bl_ci_upper) | (gs['ci_upper'] < bl_ci_lower)

	# Plot non-significant points
	if (~sig_mask).any():
		gs_ns = gs[~sig_mask]
		ax.errorbar(
			gs_ns['beta'],
			gs_ns['mean'],
			yerr=[gs_ns['mean'] - gs_ns['ci_lower'], gs_ns['ci_upper'] - gs_ns['mean']],
			fmt='o-',
			color='#1f77b4',
			capsize=5,
			capthick=1.5,
			linewidth=2,
			markersize=6,
			alpha=0.8,
			zorder=3,
			label='Not sig. different from baseline',
		)
		ax.fill_between(
			gs_ns['beta'],
			gs_ns['ci_lower'],
			gs_ns['ci_upper'],
			alpha=0.15,
			color='#1f77b4',
			zorder=2,
		)

	# Plot significant points
	if sig_mask.any():
		gs_sig = gs[sig_mask]
		ax.errorbar(
			gs_sig['beta'],
			gs_sig['mean'],
			yerr=[gs_sig['mean'] - gs_sig['ci_lower'], gs_sig['ci_upper'] - gs_sig['mean']],
			fmt='s-',
			color='#d62728',
			capsize=5,
			capthick=1.5,
			linewidth=2,
			markersize=7,
			alpha=0.9,
			zorder=4,
			label='Sig. different from baseline (p<0.05)',
		)
		ax.fill_between(
			gs_sig['beta'],
			gs_sig['ci_lower'],
			gs_sig['ci_upper'],
			alpha=0.2,
			color='#d62728',
			zorder=2,
		)

	ax.set_xscale('log')
	ax.set_xlabel('Beta (log scale)')
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best', fontsize=9)

	if len(gs) > 1:
		ax.set_xlim(gs['beta'].min() * 0.7, gs['beta'].max() * 1.3)


def make_heatmap(
	df: pd.DataFrame,
	values: str,
	output_dir: Path,
	filename: str,
	cmap: str = BOTTLENECK_CMAP,
	fmt: str = '.1f',
	norm_type: str | None = None,
	title_suffix: str = '',
) -> None:
	"""Create a heatmap of ``values`` pivoted by bottleneck_width x beta.

	Parameters
	----------
	df :
	    DataFrame with ``bottleneck_width``, ``beta``, and ``values`` columns.
	values :
	    Column name to use for heatmap values.
	output_dir :
	    Directory to save the plot.
	filename :
	    Output filename.
	cmap :
	    Colormap.
	fmt :
	    Format string for annotations.
	norm_type :
	    One of ``'log'``, ``'power'``, or ``None``.
	title_suffix :
	    Extra text to append to the title.
	"""
	pivot = df.pivot_table(
		values=values,
		index='bottleneck_width',
		columns='beta',
		aggfunc='mean',
	)
	finite = pivot.values[np.isfinite(pivot.values)]
	vmin = np.min(finite) if len(finite) > 0 else 0
	vmax = np.max(finite) if len(finite) > 0 else 1

	# Ensure vmin is positive for LogNorm/PowerNorm
	if vmin <= 0:
		vmin = vmin * 0.9 if vmin < 0 else vmax * 0.01

	norm = None
	if norm_type == 'log':
		norm = LogNorm(vmin=vmin, vmax=vmax)
	elif norm_type == 'power':
		norm = PowerNorm(gamma=1.5, vmin=vmin, vmax=vmax)

	fig, ax = plt.subplots(figsize=(10, 6))
	sns.heatmap(
		pivot,
		annot=True,
		fmt=fmt,
		cmap=cmap,
		ax=ax,
		norm=norm,
		cbar_kws={'label': values.replace('_', ' ').title()},
	)
	ax.set_title(
		f'Average {values.replace("_", " ").title()}: Bottleneck Width x Beta{title_suffix}'
	)
	ax.set_xlabel('Beta')
	ax.set_ylabel('Bottleneck Width')
	plt.tight_layout()
	save_and_close(fig, output_dir, filename)


def scatter_with_colorbar(
	df: pd.DataFrame,
	x: str,
	y: str,
	c: str,
	output_dir: Path,
	filename: str,
	title: str = '',
	cmap: str = BOTTLENECK_CMAP,
	norm_type: str | None = None,
	xscale: str | None = None,
	yscale: str | None = None,
	sizes: np.ndarray | None = None,
	size_legend: bool = False,
	size_labels: Sequence[str] | None = None,
) -> None:
	"""Create a scatter plot with a colorbar.

	Parameters
	----------
	df :
	    DataFrame.
	x, y :
	    Column names for axes.
	c :
	    Column name for color.
	output_dir :
	    Directory to save the plot.
	filename :
	    Output filename.
	title :
	    Plot title.
	cmap :
	    Colormap.
	norm_type :
	    One of ``'log'``, ``None``.
	xscale, yscale :
	    Axis scales (e.g. ``'log'``).
	sizes :
	    Point sizes.
	size_legend :
	    Whether to add a size legend.
	size_labels :
	    Labels for the size legend.
	"""
	fig, ax = plt.subplots(figsize=(10, 6))

	# Handle c column with possible zeros for LogNorm
	c_values = df[c]
	norm = None
	if norm_type == 'log':
		c_positive = c_values[c_values > 0]
		if len(c_positive) > 0:
			vmin, vmax = c_positive.min(), c_positive.max()
			norm = LogNorm(vmin=vmin, vmax=vmax)
			c_values = c_values.replace(0, np.nan)
		else:
			norm = LogNorm(vmin=1e-8, vmax=1)
			c_values = c_values.replace(0, np.nan)

	scatter = ax.scatter(
		df[x],
		df[y],
		c=c_values,
		norm=norm,
		cmap=cmap,
		s=sizes,
		alpha=0.6,
		edgecolors='black',
		linewidth=0.5,
	)
	ax.set_xlabel(x.replace('_', ' ').title())
	ax.set_ylabel(y.replace('_', ' ').title())
	ax.set_title(title)
	if xscale:
		ax.set_xscale(xscale)
	if yscale:
		ax.set_yscale(yscale)
	ax.grid(True, alpha=0.3, which='both')
	cbar = plt.colorbar(scatter)
	cbar.set_label(c.replace('_', ' ').title())

	if size_legend and size_labels is not None and sizes is not None:
		unique_labels = sorted(set(size_labels))
		legend_handles = [
			plt.scatter(
				[], [], s=s, color='gray', alpha=0.5, edgecolors='black', label=f'Width: {w}'
			)
			for w, s in zip(unique_labels, [np.log(w) * 20 for w in unique_labels])
		]
		ax.legend(
			handles=legend_handles,
			title='Bottleneck Width',
			loc='center left',
			bbox_to_anchor=(1.2, 0.5),
		)

	plt.tight_layout()
	save_and_close(fig, output_dir, filename)
