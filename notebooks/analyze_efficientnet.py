"""
Unified EfficientNet Analysis.

Combines efficientnet_results_analisys.py + efficientnet_numerical_analisys.py into a single script.

Usage:
    python notebooks/analyze_efficientnet.py
"""

import matplotlib

matplotlib.use('Agg')

from pathlib import Path
import warnings

from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from experiments.analysis import (
	add_engineered_features,
	analyze_beta_metric_relationship,
	compute_beta_group_statistics,
	compute_correlation_matrix,
	find_significant_beta_regions,
	gradient_boosting_analysis,
	linear_regression_analysis,
	load_with_baseline,
	pairwise_beta_significance,
	plot_accuracy_vs_beta_with_gradient,
	plot_bootstrap_ci_vs_beta,
	plot_bootstrap_ci_vs_beta_with_baseline,
	plot_compression_vs_beta_with_gradient,
	plot_metric_vs_beta_error_bars,
	prepare_regression_data,
	random_forest_analysis,
)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')

BOTTLENECK_CMAP = 'plasma'
NUMERICAL_CMAP = 'viridis'

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / 'results' / 'efficientnet' / 'grid_search_results_final.json'
BASELINE_PATH = PROJECT_ROOT / 'results' / 'baseline' / 'grid_search_results_final.json'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'efficientnet'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.stdout = open(OUTPUT_DIR / 'analysis.log', 'w', encoding='utf-8')
print(f'All output written to: {OUTPUT_DIR / "analysis.log"}', file=sys.stderr)

EFFICIENTNET_MODELS = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_heatmap(df, values, filename, fmt='.1f', norm_type=None, title_suffix=''):
	try:
		df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[values])
		if len(df_clean) == 0:
			return
		pivot = df_clean.pivot_table(
			values=values, index='bottleneck_width', columns='beta', aggfunc='mean'
		)
		finite = pivot.values[np.isfinite(pivot.values)]
		if len(finite) == 0:
			return
		vmin, vmax = np.min(finite), np.max(finite)
		if vmin <= 0:
			vmin = vmin * 0.9 if vmin < 0 else vmax * 0.01
		norm = {
			'log': LogNorm(vmin=vmin, vmax=vmax),
			'power': PowerNorm(gamma=1.5, vmin=vmin, vmax=vmax),
		}.get(norm_type)
		fig, ax = plt.subplots(figsize=(10, 6))
		sns.heatmap(
			pivot,
			annot=True,
			fmt=fmt,
			cmap=BOTTLENECK_CMAP,
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
		fig.savefig(OUTPUT_DIR / filename, dpi=150)
		print(f'Saved: {filename}')
		plt.close('all')
	except Exception as e:
		print(f'  Warning: {filename}: {e}')
		plt.close('all')


# ==============================================================================
# LOAD DATA
# ==============================================================================
print(f'Loading data from: {RESULTS_PATH}')
df = load_with_baseline(str(RESULTS_PATH), str(BASELINE_PATH), model_filter=EFFICIENTNET_MODELS)
print(f'Total: {len(df)} experiments, Models: {df["model_arch"].unique()}')

df_baseline_all = load_with_baseline(
	str(RESULTS_PATH), str(BASELINE_PATH), model_filter=EFFICIENTNET_MODELS
)
df_baseline_only = df_baseline_all[df_baseline_all['beta'] == 0]
df_no_baseline = df[df['beta'] > 0].copy()
df['log_beta'] = np.log10(df['beta'].replace(0, np.nan))
df['log_width'] = np.log2(df['bottleneck_width'])

# ==============================================================================
# PART 1 — DESCRIPTIVE + VISUALIZATIONS
# ==============================================================================
print('\n' + '=' * 60 + '\nPART 1: DESCRIPTIVE ANALYSIS\n' + '=' * 60)

print(f'\nTotal experiments: {len(df)}')
print(f'Models: {df["model_arch"].unique()}')
print(f'Beta range: [{df["beta"].min()}, {df["beta"].max()}]')
print(f'Mean accuracy: {df["test_accuracy"].mean():.2f}%, Best: {df["test_accuracy"].max():.2f}%')

print('\nBEST CONFIGURATIONS')
for _, row in df.nlargest(10, 'test_accuracy').iterrows():
	print(
		f'  {row["test_accuracy"]:.2f}% | {row["model_arch"]} | w={row["bottleneck_width"]} | beta={row["beta"]:.6f} | seed={row["seed"]}'
	)

print('\nSTATISTICS BY ARCHITECTURE')
print(
	df.groupby('model_arch')
	.agg(
		{
			'test_accuracy': ['mean', 'std', 'min', 'max', 'count'],
			'final_train_loss': 'mean',
			'final_val_loss': 'mean',
		}
	)
	.round(2)
)

print('\nSTATISTICS BY WIDTH')
print(
	df.groupby('bottleneck_width')
	.agg(
		{
			'test_accuracy': ['mean', 'std', 'min', 'max'],
			'final_train_loss': 'mean',
			'final_val_loss': 'mean',
			'final_empirical_compression': 'mean',
		}
	)
	.round(2)
)

print('\nEFFECT OF BETA')
print(
	df.groupby('beta')
	.agg(
		{
			'test_accuracy': ['mean', 'std', 'min', 'max', 'count'],
			'final_train_loss': 'mean',
			'final_val_loss': 'mean',
		}
	)
	.round(2)
)

# Heatmaps
print('\nBUILDING HEATMAPS')
_make_heatmap(df, 'test_accuracy', 'heatmap_accuracy_width_beta.png', fmt='.1f')
_make_heatmap(df, 'final_val_loss', 'heatmap_loss_width_beta.png', fmt='.3f', norm_type='log')
_make_heatmap(
	df,
	'final_empirical_compression',
	'heatmap_compression_width_beta.png',
	fmt='.3f',
	norm_type='power',
)
_make_heatmap(
	df,
	'final_effective_capacity_utilization',
	'heatmap_capacity_width_beta.png',
	fmt='.3f',
	norm_type='log',
)

# Scatter: compression vs capacity
fig, ax = plt.subplots(figsize=(10, 6))
s = ax.scatter(
	df['final_empirical_compression'],
	df['final_effective_capacity_utilization'],
	c=df['test_accuracy'],
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
	edgecolors='black',
	linewidth=0.5,
)
ax.set_xlabel('Compression')
ax.set_ylabel('Capacity')
ax.set_title('Accuracy: Compression vs Capacity')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
plt.colorbar(s, label='Accuracy (%)')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'scatter_acc_compression_capacity.png', dpi=150)
print('Saved: scatter_acc_compression_capacity.png')
plt.close('all')

# Scatter: compression vs accuracy
fig, ax = plt.subplots(figsize=(10, 6))
bp = df[df['beta'] > 0]['beta']
bmin, bmax = (bp.min(), bp.max()) if len(bp) > 0 else (1e-8, 1)
w = df['bottleneck_width']
sizes = np.log(w) * 20
s = ax.scatter(
	df['final_empirical_compression'],
	df['test_accuracy'],
	c=df['beta'].replace(0, np.nan),
	norm=LogNorm(bmin, bmax),
	cmap=BOTTLENECK_CMAP,
	s=sizes,
	alpha=0.6,
	edgecolors='black',
	linewidth=0.5,
)
ax.set_xlabel('Compression')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy vs Compression (color=beta, size=width)')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, which='both')
cb = plt.colorbar(s)
cb.set_label('Beta')
lh = [
	plt.scatter(
		[], [], s=np.log(x) * 20, color='gray', alpha=0.5, edgecolors='black', label=f'W:{x}'
	)
	for x in sorted(w.unique())
]
ax.legend(handles=lh, title='Width', loc='center left', bbox_to_anchor=(1.2, 0.5))
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'scatter_acc_compression.png', dpi=150)
print('Saved: scatter_acc_compression.png')
plt.close('all')

# Accuracy vs beta
fig, ax = plt.subplots(figsize=(10, 6))
plot_accuracy_vs_beta_with_gradient(
	ax, df_no_baseline, 'Accuracy vs Beta (color=Width)', baseline_data=df_baseline_only
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy_vs_beta.png', dpi=150)
print('Saved: accuracy_vs_beta.png')
plt.close('all')

# Compression vs beta
fig, ax = plt.subplots(figsize=(10, 6))
plot_compression_vs_beta_with_gradient(
	ax, df_no_baseline, 'Compression vs Beta (color=Width)', baseline_data=df_baseline_only
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'compression_vs_beta.png', dpi=150)
print('Saved: compression_vs_beta.png')
plt.close('all')

# Error bars
fig, ax = plt.subplots(figsize=(10, 6))
plot_metric_vs_beta_error_bars(
	ax,
	df_no_baseline,
	'test_accuracy',
	'Accuracy vs Beta (mean ± std)',
	'Accuracy (%)',
	baseline_data=df_baseline_only,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy_vs_beta_errorbars.png', dpi=150)
print('Saved: accuracy_vs_beta_errorbars.png')
plt.close('all')

fig, ax = plt.subplots(figsize=(10, 6))
plot_metric_vs_beta_error_bars(
	ax,
	df_no_baseline,
	'final_empirical_compression',
	'Compression vs Beta (mean ± std)',
	'Compression',
	baseline_data=df_baseline_only,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'compression_vs_beta_errorbars.png', dpi=150)
print('Saved: compression_vs_beta_errorbars.png')
plt.close('all')

# Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['test_accuracy'], bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(
	df['test_accuracy'].mean(),
	color='red',
	linestyle='--',
	linewidth=2,
	label=f'Mean: {df["test_accuracy"].mean():.2f}%',
)
axes[0].set_xlabel('Accuracy (%)')
axes[0].set_ylabel('Count')
axes[0].set_title('Accuracy Distribution')
axes[0].legend()
df.boxplot(column='test_accuracy', by='model_arch', ax=axes[1])
axes[1].set_xlabel('Architecture')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy by Architecture')
plt.suptitle('')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy_distribution.png', dpi=150)
print('Saved: accuracy_distribution.png')
plt.close('all')

# Train vs val loss
fig, ax = plt.subplots(figsize=(8, 6))
s = ax.scatter(
	df['final_train_loss'],
	df['final_val_loss'],
	c=df['test_accuracy'],
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
)
ax.set_xlabel('Train Loss')
ax.set_ylabel('Val Loss')
ax.set_title('Train vs Val Loss (color=accuracy)')
ax.set_xscale('log')
ax.set_yscale('log')
plt.colorbar(s, label='Accuracy (%)')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'train_vs_val_loss.png', dpi=150)
print('Saved: train_vs_val_loss.png')
plt.close('all')

# Compression metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bv = df[df['beta'] > 0]['beta']
bmin, bmax = (bv.min(), bv.max()) if len(bv) > 0 else (1e-8, 1)
axes[0].scatter(
	df['final_empirical_compression'],
	df['test_accuracy'],
	c=df['beta'].replace(0, np.nan),
	norm=LogNorm(bmin, bmax),
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
)
axes[0].set_xlabel('Compression')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Compression vs Accuracy (color=beta)')
axes[0].grid(True, alpha=0.3)
axes[1].scatter(
	df['final_effective_capacity_utilization'],
	df['test_accuracy'],
	c=df['bottleneck_width'],
	norm=LogNorm(df['bottleneck_width'].min(), df['bottleneck_width'].max()),
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
)
axes[1].set_xlabel('Capacity Utilization')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Capacity vs Accuracy (color=width)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'compression_metrics.png', dpi=150)
print('Saved: compression_metrics.png')
plt.close('all')

# Per-architecture
for model_arch in df['model_arch'].unique():
	mdf = df[df['model_arch'] == model_arch].copy()
	print(
		f'\n--- {model_arch} ({len(mdf)} exp, mean={mdf["test_accuracy"].mean():.2f}%, best={mdf["test_accuracy"].max():.2f}%) ---'
	)
	_make_heatmap(
		mdf,
		'test_accuracy',
		f'heatmap_accuracy_{model_arch}.png',
		fmt='.1f',
		title_suffix=f' ({model_arch})',
	)
	_make_heatmap(
		mdf,
		'final_val_loss',
		f'heatmap_loss_{model_arch}.png',
		fmt='.3f',
		norm_type='log',
		title_suffix=f' ({model_arch})',
	)
	_make_heatmap(
		mdf,
		'final_empirical_compression',
		f'heatmap_compression_{model_arch}.png',
		fmt='.3f',
		norm_type='power',
		title_suffix=f' ({model_arch})',
	)
	_make_heatmap(
		mdf,
		'final_effective_capacity_utilization',
		f'heatmap_capacity_{model_arch}.png',
		fmt='.3f',
		norm_type='log',
		title_suffix=f' ({model_arch})',
	)

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.scatter(
		mdf['final_empirical_compression'],
		mdf['final_effective_capacity_utilization'],
		c=mdf['test_accuracy'],
		cmap=BOTTLENECK_CMAP,
		alpha=0.6,
		s=50,
		edgecolors='black',
		linewidth=0.5,
	)
	ax.set_xlabel('Compression')
	ax.set_ylabel('Capacity')
	ax.set_title(f'Compression vs Capacity ({model_arch})')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid(True, alpha=0.3, which='both')
	plt.colorbar(ax.collections[0], label='Accuracy (%)')
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'scatter_acc_compression_capacity_{model_arch}.png', dpi=150)
	print('  Saved scatter')
	plt.close('all')

	bp = mdf[mdf['beta'] > 0]['beta']
	bmin, bmax = (bp.min(), bp.max()) if len(bp) > 0 else (1e-8, 1)
	w = mdf['bottleneck_width']
	sz = np.log(w) * 20
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.scatter(
		mdf['final_empirical_compression'],
		mdf['test_accuracy'],
		c=mdf['beta'].replace(0, np.nan),
		norm=LogNorm(bmin, bmax),
		cmap=BOTTLENECK_CMAP,
		s=sz,
		alpha=0.6,
		edgecolors='black',
		linewidth=0.5,
	)
	ax.set_xlabel('Compression')
	ax.set_ylabel('Accuracy (%)')
	ax.set_title(f'Accuracy vs Compression ({model_arch})')
	ax.set_xscale('log')
	ax.grid(True, alpha=0.3, which='both')
	plt.colorbar(ax.collections[0], label='Beta')
	uw = sorted(w.unique())
	lh = [
		plt.scatter(
			[], [], s=np.log(x) * 20, color='gray', alpha=0.5, edgecolors='black', label=f'W:{x}'
		)
		for x in uw
	]
	ax.legend(handles=lh, title='Width', loc='center left', bbox_to_anchor=(1.2, 0.5))
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'scatter_acc_compression_{model_arch}.png', dpi=150)
	print('  Saved scatter2')
	plt.close('all')

	ba = df_baseline_only[df_baseline_only['model_arch'] == model_arch]
	mnb = mdf[mdf['beta'] > 0]
	for fname, plot_fn, args in [
		(
			'accuracy_vs_beta',
			plot_accuracy_vs_beta_with_gradient,
			(mnb, f'Accuracy vs Beta ({model_arch})'),
		),
		(
			'compression_vs_beta',
			plot_compression_vs_beta_with_gradient,
			(mnb, f'Compression vs Beta ({model_arch})'),
		),
	]:
		fig, ax = plt.subplots(figsize=(10, 6))
		plot_fn(ax, *args, baseline_data=ba)
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / f'{fname}_{model_arch}.png', dpi=150)
		print(f'  Saved {fname}')
		plt.close('all')

	for fname, metric, title, ylabel in [
		(
			'accuracy_vs_beta_errorbars',
			'test_accuracy',
			f'Accuracy vs Beta ({model_arch})',
			'Accuracy (%)',
		),
		(
			'compression_vs_beta_errorbars',
			'final_empirical_compression',
			f'Compression vs Beta ({model_arch})',
			'Compression',
		),
	]:
		fig, ax = plt.subplots(figsize=(10, 6))
		plot_metric_vs_beta_error_bars(ax, mnb, metric, title, ylabel, baseline_data=ba)
		plt.tight_layout()
		fig.savefig(OUTPUT_DIR / f'{fname}_{model_arch}.png', dpi=150)
		print(f'  Saved {fname}')
		plt.close('all')

# Stability
print('\nSTABILITY BY SEED')
cv = df.groupby(['model_arch', 'bottleneck_width', 'beta']).agg(
	{'test_accuracy': ['mean', 'std', 'count']}
)
cv.columns = ['mean_acc', 'std_acc', 'count']
cv = cv[cv['count'] >= 2].reset_index()
print(
	cv.nlargest(10, 'std_acc')[
		['model_arch', 'bottleneck_width', 'beta', 'mean_acc', 'std_acc']
	].round(2)
)

# Summary
summary = df.groupby(['model_arch', 'bottleneck_width', 'beta']).agg(
	{
		'test_accuracy': ['mean', 'std'],
		'final_val_loss': 'mean',
		'final_empirical_compression': 'mean',
	}
)
summary.columns = ['mean_acc', 'std_acc', 'mean_val_loss', 'mean_compression']
summary = summary.round(3)
summary.to_csv(OUTPUT_DIR / 'summary_by_config.csv')
print('\nSaved: summary_by_config.csv')

# Final report
best_row = df.loc[df['test_accuracy'].idxmax()]
report = f"""EfficientNet Grid Search Results Analysis
{'=' * 60}
Total experiments: {len(df)}
Mean accuracy: {df['test_accuracy'].mean():.2f}%
Best accuracy: {df['test_accuracy'].max():.2f}%

Best configuration:
  Model: {best_row['model_arch']}
  Width: {best_row['bottleneck_width']}
  Beta: {best_row['beta']}
  Seed: {best_row['seed']}
  Test Accuracy: {best_row['test_accuracy']:.2f}%
  Final Val Loss: {best_row['final_val_loss']:.4f}
  Final Train Loss: {best_row['final_train_loss']:.4f}
"""
(OUTPUT_DIR / 'analysis_report.txt').write_text(report, encoding='utf-8')
print('Saved: analysis_report.txt')

# ==============================================================================
# PART 2 — NUMERICAL / STATISTICAL ANALYSIS
# ==============================================================================
print('\n' + '=' * 60 + '\nPART 2: NUMERICAL ANALYSIS\n' + '=' * 60)

# Correlation
corr = compute_correlation_matrix(df)
print('\nCorrelation Matrix:\n', corr.round(3))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
	corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax, square=True, linewidths=0.5
)
ax.set_title('Correlation Matrix')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=150)
print('Saved: correlation_matrix.png')
plt.close('all')

# Bootstrap
print('\n--- Bootstrap CI: Accuracy ---')
acc_stats = compute_beta_group_statistics(df_no_baseline, 'test_accuracy', n_bootstrap=10000)
print(acc_stats.round(3).to_string(index=False))
print('\n--- Bootstrap CI: Compression ---')
comp_stats = compute_beta_group_statistics(
	df_no_baseline, 'final_empirical_compression', n_bootstrap=10000
)
print(comp_stats.round(3).to_string(index=False))
print('\n--- Bootstrap CI (with baseline): Accuracy ---')
acc_stats_full = compute_beta_group_statistics(df, 'test_accuracy', n_bootstrap=10000)
print(acc_stats_full.round(3).to_string(index=False))
print('\n--- Bootstrap CI (with baseline): Compression ---')
comp_stats_full = compute_beta_group_statistics(
	df, 'final_empirical_compression', n_bootstrap=10000
)
print(comp_stats_full.round(3).to_string(index=False))

# Significant regions
acc_sig = find_significant_beta_regions(df_no_baseline, 'test_accuracy', n_bootstrap=10000)
print('\n--- Significant Beta Regions: Accuracy ---')
print(
	acc_sig[['beta_from', 'beta_to', 'mean_a', 'mean_b', 'diff', 'significant', 'cohens_d']]
	.round(3)
	.to_string(index=False)
)
comp_sig = find_significant_beta_regions(
	df_no_baseline, 'final_empirical_compression', n_bootstrap=10000
)
print('\n--- Significant Beta Regions: Compression ---')
print(
	comp_sig[['beta_from', 'beta_to', 'mean_a', 'mean_b', 'diff', 'significant', 'cohens_d']]
	.round(3)
	.to_string(index=False)
)

# Relationship analysis
acc_rel = analyze_beta_metric_relationship(df_no_baseline, 'test_accuracy')
comp_rel = analyze_beta_metric_relationship(df_no_baseline, 'final_empirical_compression')
print(f'\nTwo-Way ANOVA (beta x width interaction) — Accuracy:')
anova_acc = acc_rel['anova_table']
for effect, vals in anova_acc.items():
	print(f'  {effect}: F={vals["F"]:.2f}, p={vals["p"]:.2e}, df={vals["df"]}')
print(f'Model R² = {acc_rel["r_squared"]:.4f}')

print(f'\nTwo-Way ANOVA (beta x width interaction) — Compression:')
anova_comp = comp_rel['anova_table']
for effect, vals in anova_comp.items():
	print(f'  {effect}: F={vals["F"]:.2f}, p={vals["p"]:.2e}, df={vals["df"]}')
print(f'Model R² = {comp_rel["r_squared"]:.4f}')

# Bootstrap CI plots
fig, ax = plt.subplots(figsize=(10, 6))
plot_bootstrap_ci_vs_beta(
	ax,
	acc_stats,
	'test_accuracy',
	'Accuracy vs Beta (Bootstrap 95% CI)',
	'Accuracy (%)',
	significant_regions=acc_sig,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'bootstrap_accuracy_ci.png', dpi=150)
print('Saved: bootstrap_accuracy_ci.png')
plt.close('all')

fig, ax = plt.subplots(figsize=(10, 6))
plot_bootstrap_ci_vs_beta(
	ax,
	comp_stats,
	'final_empirical_compression',
	'Compression vs Beta (Bootstrap 95% CI)',
	'Compression',
	significant_regions=comp_sig,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'bootstrap_compression_ci.png', dpi=150)
print('Saved: bootstrap_compression_ci.png')
plt.close('all')

# Bootstrap vs baseline
if len(acc_stats_full[acc_stats_full['beta'] == 0]) > 0:
	ba_acc = acc_stats_full[acc_stats_full['beta'] == 0].iloc[0]
	fig, ax = plt.subplots(figsize=(12, 6))
	plot_bootstrap_ci_vs_beta_with_baseline(
		ax,
		acc_stats_full,
		ba_acc,
		'test_accuracy',
		'Accuracy vs Beta with Baseline',
		'Accuracy (%)',
	)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / 'bootstrap_accuracy_ci_vs_baseline.png', dpi=150)
	print('Saved: bootstrap_accuracy_ci_vs_baseline.png')
	plt.close('all')

	ba_comp = comp_stats_full[comp_stats_full['beta'] == 0].iloc[0]
	fig, ax = plt.subplots(figsize=(12, 6))
	plot_bootstrap_ci_vs_beta_with_baseline(
		ax,
		comp_stats_full,
		ba_comp,
		'final_empirical_compression',
		'Compression vs Beta with Baseline',
		'Compression',
	)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / 'bootstrap_compression_ci_vs_baseline.png', dpi=150)
	print('Saved: bootstrap_compression_ci_vs_baseline.png')
	plt.close('all')

# Pairwise significance
print('\n--- Pairwise Significance: Accuracy ---')
acc_pw = pairwise_beta_significance(df_no_baseline, 'test_accuracy', n_bootstrap=5000)
bvals = sorted(df_no_baseline['beta'].unique())
sig_m = pd.DataFrame(False, index=bvals, columns=bvals, dtype=bool)
ann_m = pd.DataFrame('', index=bvals, columns=bvals)
for _, r in acc_pw.iterrows():
	a, b, d = r['beta_a'], r['beta_b'], r['cohens_d']
	sig_m.loc[a, b] = sig_m.loc[b, a] = r['p_significant']
	if r['p_significant']:
		lv = '***' if abs(d) >= 0.8 else ('**' if abs(d) >= 0.5 else '*')
		ann_m.loc[a, b] = f'{"A>B" if d > 0 else "A<B"}\n{lv}'
		ann_m.loc[b, a] = f'{"B>A" if d > 0 else "B<A"}\n{lv}'
	else:
		ann_m.loc[a, b] = ann_m.loc[b, a] = 'ns'
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
	sig_m.astype(int),
	annot=ann_m,
	fmt='',
	cmap='RdYlGn_r',
	ax=ax,
	cbar_kws={'label': 'Significant'},
	linewidths=0.5,
	linecolor='gray',
	annot_kws={'fontsize': 8},
)
ax.set_title('Pairwise Significance: Accuracy (95% Bootstrap CI)')
ax.set_xlabel('Beta')
ax.set_ylabel('Beta')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'pairwise_significance_accuracy.png', dpi=150)
print('Saved: pairwise_significance_accuracy.png')
plt.close('all')

# Regression
print('\n--- Linear Regression ---')
df_reg, X, y_acc, y_comp = prepare_regression_data(df)
print(f'Using {len(X)} experiments (excl. baseline)')
lr = linear_regression_analysis(X, y_acc, 'Test Accuracy')
lr_c = linear_regression_analysis(X, y_comp, 'Empirical Compression')

# Regression plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].scatter(y_acc, lr['y_pred'], alpha=0.5, edgecolors='black', linewidth=0.5, s=40)
axes[0, 0].plot([y_acc.min(), y_acc.max()], [y_acc.min(), y_acc.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title(f'Actual vs Predicted (R²={lr["r2"]:.4f})')
axes[0, 0].grid(True, alpha=0.3)
res = lr['residuals']
axes[0, 1].scatter(lr['y_pred'], res, alpha=0.5, edgecolors='black', linewidth=0.5, s=40)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals')
axes[0, 1].grid(True, alpha=0.3)
axes[1, 0].hist(res, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals')
axes[1, 0].set_ylabel('Freq')
axes[1, 0].set_title(f'Residual Dist (μ={res.mean():.2f}, σ={res.std():.2f})')
axes[1, 0].grid(True, alpha=0.3)
z = np.polyfit(df_reg['log_beta'], df_reg['test_accuracy'], 1)
p = np.poly1d(z)
axes[1, 1].scatter(
	df_reg['log_beta'],
	df_reg['test_accuracy'],
	c=df_reg['test_accuracy'],
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=40,
	edgecolors='black',
	linewidth=0.3,
)
axes[1, 1].plot(
	df_reg['log_beta'].sort_values(),
	p(df_reg['log_beta'].sort_values()),
	'r-',
	lw=2,
	label=f'y={z[0]:.2f}x+{z[1]:.2f}',
)
axes[1, 1].set_xlabel('Log Beta')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Log Beta vs Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'regression_predictions.png', dpi=150)
print('Saved: regression_predictions.png')
plt.close('all')

# Non-linear models
print('\n--- Non-Linear Models ---')
X_full = add_engineered_features(df_reg[['log_beta', 'bottleneck_width']])
rf = random_forest_analysis(X_full, y_acc, 'Test Accuracy')
gb = gradient_boosting_analysis(X_full, y_acc, 'Test Accuracy')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
rfi = pd.DataFrame(
	{'Feature': X_full.columns, 'Importance': list(rf['feature_importances'].values())}
).sort_values('Importance', ascending=True)
axes[0].barh(rfi['Feature'], rfi['Importance'])
axes[0].set_xlabel('Importance')
axes[0].set_title('RF Feature Importance')
axes[0].grid(True, alpha=0.3)
gbi = pd.DataFrame(
	{'Feature': X_full.columns, 'Importance': list(gb['feature_importances'].values())}
).sort_values('Importance', ascending=True)
axes[1].barh(gbi['Feature'], gbi['Importance'])
axes[1].set_xlabel('Importance')
axes[1].set_title('GB Feature Importance')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
print('Saved: feature_importance.png')
plt.close('all')

# Clustering
print('\n--- KMeans Clustering ---')
cf = ['log_beta', 'bottleneck_width', 'test_accuracy', 'final_empirical_compression']
Xc = df_reg[cf].copy()
scaler = StandardScaler()
Xcs = scaler.fit_transform(Xc)

from sklearn.metrics import silhouette_score

inertias, sils = [], []
for k in range(2, 11):
	km = KMeans(n_clusters=k, random_state=42, n_init=10)
	lb = km.fit_predict(Xcs)
	inertias.append(km.inertia_)
	sils.append(silhouette_score(Xcs, lb))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(2, 11), inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)
axes[1].plot(range(2, 11), sils, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette')
axes[1].set_title('Silhouette Score')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'elbow_method.png', dpi=150)
print('Saved: elbow_method.png')
plt.close('all')
print(f'Best k by silhouette: {list(range(2, 11))[np.argmax(sils)]} (score={max(sils):.4f})')

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_reg['cluster'] = kmeans.fit_predict(Xcs)
print(f'\nCluster characteristics (k={optimal_k}):')
print(df_reg.groupby('cluster')[cf].agg(['mean', 'std']).round(3))

cc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
cl = ['Low Acc, High Beta', 'High Acc, Low Beta', 'Medium Acc', 'Transition']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
projections = [
	(axes[0, 0], 'log_beta', 'bottleneck_width', 'Beta vs Width', True, False),
	(axes[0, 1], 'log_beta', 'test_accuracy', 'Beta vs Accuracy', False, False),
	(axes[1, 0], 'bottleneck_width', 'test_accuracy', 'Width vs Accuracy', False, True),
	(
		axes[1, 1],
		'final_empirical_compression',
		'test_accuracy',
		'Compression vs Accuracy',
		False,
		False,
	),
]
for ax, xcol, ycol, title, xlog, ylog in projections:
	for cid in range(optimal_k):
		mask = df_reg['cluster'] == cid
		ax.scatter(
			df_reg.loc[mask, xcol],
			df_reg.loc[mask, ycol],
			c=[cc[cid]],
			label=cl[cid],
			alpha=0.6,
			s=50,
			edgecolors='black',
			linewidth=0.5,
		)
	ax.set_xlabel(xcol)
	ax.set_ylabel(ycol)
	ax.set_title(title)
	if xlog:
		ax.set_xscale('log', base=2)
	if ylog:
		ax.set_yscale('log', base=2)
	ax.legend(title='Cluster')
	ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'cluster_visualization.png', dpi=150)
print('Saved: cluster_visualization.png')
plt.close('all')

# DBSCAN
print('\n--- DBSCAN ---')
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_reg['dbscan_cluster'] = dbscan.fit_predict(Xcs)
nc = len(set(df_reg['dbscan_cluster'])) - (1 if -1 in df_reg['dbscan_cluster'].values else 0)
nn = list(df_reg['dbscan_cluster']).count(-1)
print(f'Clusters: {nc}, Noise: {nn}')

# PCA
print('\n--- PCA ---')
pca_f = [
	'log_beta',
	'bottleneck_width',
	'test_accuracy',
	'final_empirical_compression',
	'final_val_loss',
	'final_effective_capacity_utilization',
]
Xp = df[pca_f].replace([np.inf, -np.inf], np.nan).dropna()
print(f'PCA samples: {len(Xp)} (dropped {len(df) - len(Xp)})')
Xps = scaler.fit_transform(Xp)
pca = PCA()
Xpt = pca.fit_transform(Xps)
print('Explained variance:', pca.explained_variance_ratio_)
print('Loadings:')
print(
	pd.DataFrame(
		pca.components_.T, columns=[f'PC{i + 1}' for i in range(len(pca.components_))], index=pca_f
	).round(3)
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
axes[0].set_xlabel('PC')
axes[0].set_ylabel('Explained Variance')
axes[0].set_title('Scree Plot')
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95%')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
np.random.seed(42)
jitter = np.random.normal(0, 0.3, size=Xpt.shape)
s = axes[1].scatter(
	Xpt[:, 0] + jitter[:, 0],
	Xpt[:, 1] + jitter[:, 1],
	c=Xp['test_accuracy'].values,
	cmap=BOTTLENECK_CMAP,
	alpha=0.7,
	s=60,
	edgecolors='black',
	linewidth=0.3,
)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[1].set_title('PCA Projection (color=accuracy)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(s, ax=axes[1], label='Accuracy')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'pca_analysis.png', dpi=150)
print('Saved: pca_analysis.png')
plt.close('all')

# Decision tree
print('\n--- Decision Tree ---')
dt = DecisionTreeRegressor(max_depth=4, min_samples_split=10, random_state=42)
dt.fit(X_full, y_acc)
print(f'DT R²: {r2_score(y_acc, dt.predict(X_full)):.4f}')
for f, i in zip(X_full.columns, dt.feature_importances_):
	print(f'  {f}: {i:.4f}')
fig, ax = plt.subplots(figsize=(16, 10))
plot_tree(
	dt, feature_names=X_full.columns, filled=True, rounded=True, precision=2, ax=ax, fontsize=10
)
ax.set_title('Decision Tree: Test Accuracy')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'decision_tree.png', dpi=150, bbox_inches='tight')
print('Saved: decision_tree.png')
plt.close('all')

# Per-architecture numerical
le = LabelEncoder()
df['model_arch_encoded'] = le.fit_transform(df['model_arch'])
print('\n--- Per-Architecture Analysis ---')
for ma in df['model_arch'].unique():
	mdf = df[df['model_arch'] == ma]
	print(
		f'\n{ma}: n={len(mdf)}, corr(logβ)={mdf["log_beta"].corr(mdf["test_accuracy"]):.4f}, corr(w)={mdf["bottleneck_width"].corr(mdf["test_accuracy"]):.4f}'
	)
	rfm = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
	Xm = mdf[['log_beta', 'bottleneck_width']]
	ym = mdf['test_accuracy']
	rfm.fit(Xm, ym)
	cv = cross_val_score(rfm, Xm, ym, cv=5, scoring='r2')
	print(f'  RF CV R²: {cv.mean():.4f} ± {cv.std() * 2:.4f}')

# Optimal region
top_n = 20
tc = df.nlargest(top_n, 'test_accuracy')
bc = df.nsmallest(top_n, 'test_accuracy')
print(
	f'\n--- Top {top_n} avg: logβ={np.log10(tc["beta"]).mean():.4f}, w={tc["bottleneck_width"].mean():.1f}'
)
print(
	f'--- Bottom {top_n} avg: logβ={np.log10(bc["beta"]).mean():.4f}, w={bc["bottleneck_width"].mean():.1f}'
)

try:
	fig, ax = plt.subplots(figsize=(10, 8))
	vp = df['log_beta'].replace([np.inf, -np.inf], np.nan).notna()
	ax.scatter(
		df.loc[vp, 'log_beta'],
		df.loc[vp, 'bottleneck_width'],
		c=df.loc[vp, 'test_accuracy'],
		cmap='YlOrRd',
		alpha=0.6,
		s=30,
	)
	tlb = np.log10(tc['beta']).replace([np.inf, -np.inf], np.nan)
	tv = tlb.notna()
	ax.scatter(
		tlb[tv],
		tc.loc[tv, 'bottleneck_width'],
		c='gold',
		s=100,
		edgecolors='black',
		linewidth=1.5,
		label=f'Top {top_n}',
		zorder=5,
	)
	ax.set_xlabel('Log Beta')
	ax.set_ylabel('Bottleneck Width')
	ax.set_title('Optimal Configuration Region')
	ax.set_yscale('log', base=2)
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.colorbar(ax.collections[0], ax=ax, label='Accuracy')
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / 'optimal_region.png', dpi=150)
	print('Saved: optimal_region.png')
	plt.close('all')
except Exception as e:
	print(f'Warning: optimal_region.png: {e}')
	plt.close('all')


# Pareto
def find_pareto(dfr, c1='test_accuracy', c2='final_empirical_compression'):
	p = dfr.sort_values([c1, c2], ascending=[False, True])
	frontier, mx = [], -np.inf
	for _, r in p.iterrows():
		if r[c2] > mx:
			frontier.append(r)
			mx = r[c2]
	return pd.DataFrame(frontier)


pareto = find_pareto(df)
print(f'\nPareto-optimal: {len(pareto)}')
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
	df['final_empirical_compression'], df['test_accuracy'], c='gray', alpha=0.3, s=30, label='All'
)
ax.scatter(
	pareto['final_empirical_compression'],
	pareto['test_accuracy'],
	c='red',
	s=80,
	edgecolors='black',
	linewidth=1,
	label='Pareto',
	zorder=5,
)
ax.set_xlabel('Compression')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy-Compression Trade-Off')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'pareto_frontier.png', dpi=150)
print('Saved: pareto_frontier.png')
plt.close('all')

# Summary
print('\n' + '=' * 60 + '\nSUMMARY\n' + '=' * 60)
print(
	f'Total: {len(df)}, Best acc: {df["test_accuracy"].max():.4f}, Mean: {df["test_accuracy"].mean():.4f}'
)
print(f'LR CV R²: {lr["cv_r2_mean"]:.4f}, RF CV R²: {rf["cv_r2_mean"]:.4f}')
print(f'Pareto: {len(pareto)}, Clusters: {optimal_k}')
print(
	f'Spearman acc: logβ r={acc_rel["beta_spearman_corr"]:.4f}, w r={acc_rel["width_spearman_corr"]:.4f}'
)
print(
	f'Spearman comp: logβ r={comp_rel["beta_spearman_corr"]:.4f}, w r={comp_rel["width_spearman_corr"]:.4f}'
)

summary_text = f"""EfficientNet Numerical Analysis Summary
{'=' * 60}
Total experiments: {len(df)}
Best accuracy: {df['test_accuracy'].max():.4f}
Mean accuracy: {df['test_accuracy'].mean():.4f}
LR CV R²: {lr['cv_r2_mean']:.4f}
RF CV R²: {rf['cv_r2_mean']:.4f}
Pareto-optimal: {len(pareto)}
Clusters: {optimal_k}
Spearman acc: logβ r={acc_rel['beta_spearman_corr']:.4f}, w r={acc_rel['width_spearman_corr']:.4f}
Spearman comp: logβ r={comp_rel['beta_spearman_corr']:.4f}, w r={comp_rel['width_spearman_corr']:.4f}
"""
(OUTPUT_DIR / 'analysis_summary.txt').write_text(summary_text, encoding='utf-8')
print('\nSaved: analysis_summary.txt')
import sys
print(f'\nAnalysis complete! Results: {OUTPUT_DIR}', file=sys.__stdout__)
sys.stdout.close()
