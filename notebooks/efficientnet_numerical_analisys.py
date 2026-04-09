# %%
"""
Numerical Analysis of EfficientNet Grid Search Results.

This script performs comprehensive statistical and ML analysis to discover
relationships between:
- Target variables: test_accuracy, final_empirical_compression
- Predictors: beta, bottleneck_width (optionally: model_arch)

Methods used:
- Statistical analysis (correlations, regressions, ANOVA)
- Supervised learning (Random Forest, Gradient Boosting for feature importance)
- Unsupervised learning (clustering, PCA for pattern discovery)

Note: Run with `uv run notebooks/efficientnet_numerical_analisys.py`
      Plots are saved to reports/efficientnet/numerical_analysis/ without blocking.
"""

# Use non-interactive backend to avoid blocking
import matplotlib

matplotlib.use('Agg')

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
	prepare_regression_data,
	random_forest_analysis,
	save_and_close,
)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')

# Fixed colormap
BOTTLENECK_CMAP = 'viridis'


# %%
# Load data
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / 'results' / 'efficientnet' / 'grid_search_results_final.json'
BASELINE_PATH = PROJECT_ROOT / 'results' / 'baseline' / 'grid_search_results_final.json'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'efficientnet' / 'numerical_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading data from: {RESULTS_PATH}')
efficientnet_models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
df = load_with_baseline(RESULTS_PATH, BASELINE_PATH, model_filter=efficientnet_models)
print(f'\nColumns: {list(df.columns)}')
print(f'\nData types:\n{df.dtypes}')


# %%
# =============================================================================
# SECTION 1: BASIC STATISTICAL ANALYSIS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 1: BASIC STATISTICAL ANALYSIS')
print('=' * 60)

# Descriptive statistics
print('\n--- Descriptive Statistics ---')
desc_stats = df[
	['test_accuracy', 'final_empirical_compression', 'beta', 'bottleneck_width']
].describe()
print(desc_stats)

# Correlation analysis
print('\n--- Correlation Matrix ---')
corr_matrix = compute_correlation_matrix(df)
print(corr_matrix.round(3))


# %%
# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
	corr_matrix,
	annot=True,
	fmt='.3f',
	cmap='coolwarm',
	center=0,
	ax=ax,
	square=True,
	linewidths=0.5,
)
ax.set_title('Correlation Matrix: Key Variables')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_matrix.png', dpi=150)
print(f'\nSaved: correlation_matrix.png')
plt.close('all')


# %%
# Analyze beta relationship with targets
print('\n--- Beta vs Test Accuracy ---')
beta_accuracy_corr = df['beta'].corr(df['test_accuracy'])
print(f'Correlation: {beta_accuracy_corr:.4f}')

# Log-transform beta for better analysis (handle beta=0 by replacing with small positive value)
df['log_beta'] = np.log10(df['beta'].replace(0, np.nan))
# Compute correlation only on non-NaN values
valid_mask = df['log_beta'].notna()
log_beta_accuracy_corr = df.loc[valid_mask, 'log_beta'].corr(df.loc[valid_mask, 'test_accuracy'])
print(f'Log-Beta vs Accuracy correlation: {log_beta_accuracy_corr:.4f}')


# %%
# Analyze bottleneck_width relationship with targets
print('\n--- Bottleneck Width vs Test Accuracy ---')
width_accuracy_corr = df['bottleneck_width'].corr(df['test_accuracy'])
print(f'Correlation: {width_accuracy_corr:.4f}')

# Check for non-linear relationship
df['log_width'] = np.log2(df['bottleneck_width'])
log_width_accuracy_corr = df['log_width'].corr(df['test_accuracy'])
print(f'Log2-Width vs Accuracy correlation: {log_width_accuracy_corr:.4f}')


# %%
# =============================================================================
# SECTION 1b: BOOTSTRAP STATISTICAL ANALYSIS OF BETA REGIONS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 1b: BOOTSTRAP STATISTICAL ANALYSIS OF BETA REGIONS')
print('=' * 60)

# Use only non-baseline data for bootstrap analysis
df_no_baseline = df[df['beta'] > 0].copy()

# --- Accuracy bootstrap CIs ---
print('\n--- Bootstrap Confidence Intervals: Test Accuracy ---')
acc_stats = compute_beta_group_statistics(df_no_baseline, 'test_accuracy', n_bootstrap=10000)
print(acc_stats.round(3).to_string(index=False))

# --- Compression bootstrap CIs ---
print('\n--- Bootstrap Confidence Intervals: Empirical Compression ---')
comp_stats = compute_beta_group_statistics(
	df_no_baseline, 'final_empirical_compression', n_bootstrap=10000
)
print(comp_stats.round(3).to_string(index=False))

# --- Bootstrap CIs including baseline for comparison ---
print('\n--- Bootstrap Confidence Intervals (with baseline): Test Accuracy ---')
acc_stats_full = compute_beta_group_statistics(df, 'test_accuracy', n_bootstrap=10000)
print(acc_stats_full.round(3).to_string(index=False))

print('\n--- Bootstrap Confidence Intervals (with baseline): Empirical Compression ---')
comp_stats_full = compute_beta_group_statistics(
	df, 'final_empirical_compression', n_bootstrap=10000
)
print(comp_stats_full.round(3).to_string(index=False))

# --- Significant beta regions for accuracy ---
print('\n--- Significant Beta Regions: Test Accuracy ---')
acc_sig_regions = find_significant_beta_regions(df_no_baseline, 'test_accuracy', n_bootstrap=10000)
print(
	acc_sig_regions[
		['beta_from', 'beta_to', 'mean_a', 'mean_b', 'diff', 'significant', 'cohens_d']
	]
	.round(3)
	.to_string(index=False)
)

# Count significant transitions
n_sig_acc = acc_sig_regions['significant'].sum()
print(
	f'\nSignificant transitions (accuracy): {n_sig_acc} out of {len(acc_sig_regions)} adjacent pairs'
)

# --- Significant beta regions for compression ---
print('\n--- Significant Beta Regions: Empirical Compression ---')
comp_sig_regions = find_significant_beta_regions(
	df_no_baseline, 'final_empirical_compression', n_bootstrap=10000
)
print(
	comp_sig_regions[
		['beta_from', 'beta_to', 'mean_a', 'mean_b', 'diff', 'significant', 'cohens_d']
	]
	.round(3)
	.to_string(index=False)
)

n_sig_comp = comp_sig_regions['significant'].sum()
print(
	f'\nSignificant transitions (compression): {n_sig_comp} out of {len(comp_sig_regions)} adjacent pairs'
)


# %%
# =============================================================================
# SECTION 1c: RELATIONSHIP ANALYSIS - BETA, WIDTH, AND METRICS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 1c: RELATIONSHIP ANALYSIS - BETA, WIDTH, AND METRICS')
print('=' * 60)

# Use only non-baseline data for relationship analysis
print('\n--- Analyzing Relationships: Beta, Width, and Metrics ---')

# Accuracy relationships
print('\n=== Test Accuracy Relationships ===')
acc_rel = analyze_beta_metric_relationship(df_no_baseline, 'test_accuracy')
print(f'Samples: {acc_rel["n_samples"]}')
print(f'\nSpearman Correlations:')
print(
	f'  log(beta) vs accuracy: r={acc_rel["beta_spearman_corr"]:.4f}, p={acc_rel["beta_spearman_p"]:.2e}'
)
print(
	f'  width vs accuracy:     r={acc_rel["width_spearman_corr"]:.4f}, p={acc_rel["width_spearman_p"]:.2e}'
)

print(f'\nKruskal-Wallis Tests (group differences):')
print(f'  Beta groups:    H={acc_rel["kw_beta_stat"]:.2f}, p={acc_rel["kw_beta_p"]:.2e}')
print(f'  Width groups:   H={acc_rel["kw_width_stat"]:.2f}, p={acc_rel["kw_width_p"]:.2e}')

print(f'\nPartial Correlations (controlling for other variable):')
print(
	f'  log(beta) | width:  r={acc_rel["partial_beta_corr"]:.4f}, p={acc_rel["partial_beta_p"]:.2e}'
)
print(
	f'  width | log(beta):  r={acc_rel["partial_width_corr"]:.4f}, p={acc_rel["partial_width_p"]:.2e}'
)

print(f'\nTwo-Way ANOVA (beta x width interaction):')
anova_acc = acc_rel['anova_table']
for effect, vals in anova_acc.items():
	print(f'  {effect}: F={vals["F"]:.2f}, p={vals["p"]:.2e}, df={vals["df"]}')
print(f'\nModel R2 = {acc_rel["r_squared"]:.4f}')

# Compression relationships
print('\n=== Empirical Compression Relationships ===')
comp_rel = analyze_beta_metric_relationship(df_no_baseline, 'final_empirical_compression')
print(f'Samples: {comp_rel["n_samples"]}')
print(f'\nSpearman Correlations:')
print(
	f'  log(beta) vs compression: r={comp_rel["beta_spearman_corr"]:.4f}, p={comp_rel["beta_spearman_p"]:.2e}'
)
print(
	f'  width vs compression:     r={comp_rel["width_spearman_corr"]:.4f}, p={comp_rel["width_spearman_p"]:.2e}'
)

print(f'\nKruskal-Wallis Tests (group differences):')
print(f'  Beta groups:    H={comp_rel["kw_beta_stat"]:.2f}, p={comp_rel["kw_beta_p"]:.2e}')
print(f'  Width groups:   H={comp_rel["kw_width_stat"]:.2f}, p={comp_rel["kw_width_p"]:.2e}')

print(f'\nPartial Correlations (controlling for other variable):')
print(
	f'  log(beta) | width:  r={comp_rel["partial_beta_corr"]:.4f}, p={comp_rel["partial_beta_p"]:.2e}'
)
print(
	f'  width | log(beta):  r={comp_rel["partial_width_corr"]:.4f}, p={comp_rel["partial_width_p"]:.2e}'
)

print(f'\nTwo-Way ANOVA (beta x width interaction):')
anova_comp = comp_rel['anova_table']
for effect, vals in anova_comp.items():
	print(f'  {effect}: F={vals["F"]:.2f}, p={vals["p"]:.2e}, df={vals["df"]}')
print(f'\nModel R2 = {comp_rel["r_squared"]:.4f}')


# %%
# Visualize relationships: Accuracy (independent plots with larger fonts)
plt.rcParams.update(
	{'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 12}
)

# Scatter: log_beta vs accuracy
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
	df_no_baseline['log_beta'],
	df_no_baseline['test_accuracy'],
	alpha=0.4,
	s=30,
	edgecolors='gray',
	linewidth=0.5,
)
ax.set_xlabel('Log Beta', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title(
	f'Accuracy vs Log Beta\n(r={acc_rel["beta_spearman_corr"]:.3f}, p={acc_rel["beta_spearman_p"]:.2e})',
	fontsize=16,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_accuracy_vs_logbeta.png', dpi=150)
print(f'\nSaved: rel_accuracy_vs_logbeta.png')
plt.close('all')

# Scatter: width vs accuracy
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
	df_no_baseline['bottleneck_width'],
	df_no_baseline['test_accuracy'],
	alpha=0.4,
	s=30,
	edgecolors='gray',
	linewidth=0.5,
)
ax.set_xlabel('Bottleneck Width', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title(
	f'Accuracy vs Width\n(r={acc_rel["width_spearman_corr"]:.3f}, p={acc_rel["width_spearman_p"]:.2e})',
	fontsize=16,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_accuracy_vs_width.png', dpi=150)
print(f'Saved: rel_accuracy_vs_width.png')
plt.close('all')

# Box plot: accuracy by beta group
fig, ax = plt.subplots(figsize=(12, 7))
beta_groups_acc = df_no_baseline.groupby('beta')['test_accuracy'].apply(list)
labels = [f'{b:.0e}' for b in beta_groups_acc.index]
bp = ax.boxplot(beta_groups_acc.values, labels=labels, showfliers=False, patch_artist=True)
for patch in bp['boxes']:
	patch.set_facecolor('#1f77b4')
	patch.set_alpha(0.7)
ax.set_xlabel('Beta', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Accuracy Distribution by Beta', fontsize=16)
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_accuracy_by_beta.png', dpi=150)
print(f'Saved: rel_accuracy_by_beta.png')
plt.close('all')

# Box plot: accuracy by width group
fig, ax = plt.subplots(figsize=(12, 7))
width_groups_acc = df_no_baseline.groupby('bottleneck_width')['test_accuracy'].apply(list)
bp = ax.boxplot(
	width_groups_acc.values,
	labels=width_groups_acc.index.astype(str),
	showfliers=False,
	patch_artist=True,
)
for patch in bp['boxes']:
	patch.set_facecolor('#2ca02c')
	patch.set_alpha(0.7)
ax.set_xlabel('Bottleneck Width', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Accuracy Distribution by Width', fontsize=16)
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_accuracy_by_width.png', dpi=150)
print(f'Saved: rel_accuracy_by_width.png')
plt.close('all')

plt.rcParams.update(
	{'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 10}
)


# %%
# Visualize relationships: Compression (independent plots with larger fonts)
plt.rcParams.update(
	{'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'legend.fontsize': 12}
)

# Scatter: log_beta vs compression
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
	df_no_baseline['log_beta'],
	df_no_baseline['final_empirical_compression'],
	alpha=0.4,
	s=30,
	edgecolors='gray',
	linewidth=0.5,
)
ax.set_xlabel('Log Beta', fontsize=14)
ax.set_ylabel('Empirical Compression', fontsize=14)
ax.set_title(
	f'Compression vs Log Beta\n(r={comp_rel["beta_spearman_corr"]:.3f}, p={comp_rel["beta_spearman_p"]:.2e})',
	fontsize=16,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_compression_vs_logbeta.png', dpi=150)
print(f'\nSaved: rel_compression_vs_logbeta.png')
plt.close('all')

# Scatter: width vs compression
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(
	df_no_baseline['bottleneck_width'],
	df_no_baseline['final_empirical_compression'],
	alpha=0.4,
	s=30,
	edgecolors='gray',
	linewidth=0.5,
)
ax.set_xlabel('Bottleneck Width', fontsize=14)
ax.set_ylabel('Empirical Compression', fontsize=14)
ax.set_title(
	f'Compression vs Width\n(r={comp_rel["width_spearman_corr"]:.3f}, p={comp_rel["width_spearman_p"]:.2e})',
	fontsize=16,
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_compression_vs_width.png', dpi=150)
print(f'Saved: rel_compression_vs_width.png')
plt.close('all')

# Box plot: compression by beta group
fig, ax = plt.subplots(figsize=(12, 7))
beta_groups_comp = df_no_baseline.groupby('beta')['final_empirical_compression'].apply(list)
labels = [f'{b:.0e}' for b in beta_groups_comp.index]
bp = ax.boxplot(beta_groups_comp.values, labels=labels, showfliers=False, patch_artist=True)
for patch in bp['boxes']:
	patch.set_facecolor('#ff7f0e')
	patch.set_alpha(0.7)
ax.set_xlabel('Beta', fontsize=14)
ax.set_ylabel('Empirical Compression', fontsize=14)
ax.set_title('Compression Distribution by Beta', fontsize=16)
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_compression_by_beta.png', dpi=150)
print(f'Saved: rel_compression_by_beta.png')
plt.close('all')

# Box plot: compression by width group
fig, ax = plt.subplots(figsize=(12, 7))
width_groups_comp = df_no_baseline.groupby('bottleneck_width')[
	'final_empirical_compression'
].apply(list)
bp = ax.boxplot(
	width_groups_comp.values,
	labels=width_groups_comp.index.astype(str),
	showfliers=False,
	patch_artist=True,
)
for patch in bp['boxes']:
	patch.set_facecolor('#d62728')
	patch.set_alpha(0.7)
ax.set_xlabel('Bottleneck Width', fontsize=14)
ax.set_ylabel('Empirical Compression', fontsize=14)
ax.set_title('Compression Distribution by Width', fontsize=16)
ax.tick_params(axis='x', rotation=45, labelsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'rel_compression_by_width.png', dpi=150)
print(f'Saved: rel_compression_by_width.png')
plt.close('all')

plt.rcParams.update(
	{'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'legend.fontsize': 10}
)


# %%
# Visualize bootstrap CIs for accuracy
fig, ax = plt.subplots(figsize=(10, 6))
plot_bootstrap_ci_vs_beta(
	ax,
	acc_stats,
	'test_accuracy',
	'Test Accuracy vs Beta (Bootstrap 95% CI)',
	'Test Accuracy (%)',
	significant_regions=acc_sig_regions,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'bootstrap_accuracy_ci.png', dpi=150)
print(f'\nSaved: bootstrap_accuracy_ci.png')
plt.close('all')


# %%
# Visualize bootstrap CIs for compression
fig, ax = plt.subplots(figsize=(10, 6))
plot_bootstrap_ci_vs_beta(
	ax,
	comp_stats,
	'final_empirical_compression',
	'Empirical Compression vs Beta (Bootstrap 95% CI)',
	'Empirical Compression',
	significant_regions=comp_sig_regions,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'bootstrap_compression_ci.png', dpi=150)
print(f'Saved: bootstrap_compression_ci.png')
plt.close('all')


# %%
# Bootstrap CIs with baseline comparison for accuracy
print('\n--- Bootstrap CI vs Baseline: Test Accuracy ---')
baseline_acc = acc_stats_full[acc_stats_full['beta'] == 0].iloc[0]
print(
	f'Baseline (beta=0) accuracy: {baseline_acc["mean"]:.2f} '
	f'[{baseline_acc["ci_lower"]:.2f}, {baseline_acc["ci_upper"]:.2f}]'
)

fig, ax = plt.subplots(figsize=(12, 6))
plot_bootstrap_ci_vs_beta_with_baseline(
	ax,
	acc_stats_full,
	baseline_acc,
	'test_accuracy',
	'Test Accuracy vs Beta with Baseline Comparison (Bootstrap 95% CI)',
	'Test Accuracy (%)',
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'bootstrap_accuracy_ci_vs_baseline.png', dpi=150)
print(f'Saved: bootstrap_accuracy_ci_vs_baseline.png')
plt.close('all')

print('\nBeta regions significantly different from baseline (accuracy):')
for _, row in acc_stats_full[acc_stats_full['beta'] > 0].iterrows():
	is_sig = (row['ci_lower'] > baseline_acc['ci_upper']) or (
		row['ci_upper'] < baseline_acc['ci_lower']
	)
	direction = 'higher' if row['mean'] > baseline_acc['mean'] else 'lower'
	if is_sig:
		print(
			f'  beta={row["beta"]:.0e}: {direction} than baseline '
			f'(mean={row["mean"]:.2f}, CI=[{row["ci_lower"]:.2f}, {row["ci_upper"]:.2f}])'
		)


# %%
# Bootstrap CIs with baseline comparison for compression
print('\n--- Bootstrap CI vs Baseline: Empirical Compression ---')
baseline_comp = comp_stats_full[comp_stats_full['beta'] == 0].iloc[0]
print(
	f'Baseline (beta=0) compression: {baseline_comp["mean"]:.3f} '
	f'[{baseline_comp["ci_lower"]:.3f}, {baseline_comp["ci_upper"]:.3f}]'
)

fig, ax = plt.subplots(figsize=(12, 6))
plot_bootstrap_ci_vs_beta_with_baseline(
	ax,
	comp_stats_full,
	baseline_comp,
	'final_empirical_compression',
	'Empirical Compression vs Beta with Baseline Comparison (Bootstrap 95% CI)',
	'Empirical Compression',
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'bootstrap_compression_ci_vs_baseline.png', dpi=150)
print(f'Saved: bootstrap_compression_ci_vs_baseline.png')
plt.close('all')

print('\nBeta regions significantly different from baseline (compression):')
for _, row in comp_stats_full[comp_stats_full['beta'] > 0].iterrows():
	is_sig = (row['ci_lower'] > baseline_comp['ci_upper']) or (
		row['ci_upper'] < baseline_comp['ci_lower']
	)
	direction = 'higher' if row['mean'] > baseline_comp['mean'] else 'lower'
	if is_sig:
		print(
			f'  beta={row["beta"]:.0e}: {direction} than baseline '
			f'(mean={row["mean"]:.3f}, CI=[{row["ci_lower"]:.3f}, {row["ci_upper"]:.3f}])'
		)


# %%
# Pairwise significance matrix for accuracy
print('\n--- Pairwise Significance Matrix: Test Accuracy ---')
acc_pairwise = pairwise_beta_significance(df_no_baseline, 'test_accuracy', n_bootstrap=5000)
beta_vals = sorted(df_no_baseline['beta'].unique())
sig_matrix = pd.DataFrame(index=beta_vals, columns=beta_vals, dtype=bool)
for _, row in acc_pairwise.iterrows():
	sig_matrix.loc[row['beta_a'], row['beta_b']] = row['p_significant']
	sig_matrix.loc[row['beta_b'], row['beta_a']] = row['p_significant']
sig_matrix = sig_matrix.fillna(False)

# Create annotation matrix with significance levels and direction
annot_matrix = pd.DataFrame('', index=beta_vals, columns=beta_vals)
for _, row in acc_pairwise.iterrows():
	a, b = row['beta_a'], row['beta_b']
	d = row['cohens_d']  # signed Cohen's d: positive means a > b
	if row['p_significant']:
		if abs(d) >= 0.8:
			level = '***'
		elif abs(d) >= 0.5:
			level = '**'
		else:
			level = '*'
		direction = 'A>B' if d > 0 else 'A<B'
		annot_matrix.loc[a, b] = f'{direction}\n{level}'
		annot_matrix.loc[b, a] = f'{"B>A" if d > 0 else "B<A"}\n{level}'
	else:
		annot_matrix.loc[a, b] = 'ns'
		annot_matrix.loc[b, a] = 'ns'

fig, ax = plt.subplots(figsize=(12, 10))
sig_numeric = sig_matrix.astype(int)
sns.heatmap(
	sig_numeric,
	annot=annot_matrix,
	fmt='',
	cmap='RdYlGn_r',
	ax=ax,
	cbar_kws={'label': 'Significant (1=Yes, 0=No)'},
	linewidths=0.5,
	linecolor='gray',
	annot_kws={'fontsize': 8},
)
ax.set_title(
	'Pairwise Significance: Test Accuracy (95% Bootstrap CI)\n'
	'A>B/B>A = direction, *** large (|d|≥0.8), ** medium (|d|≥0.5), * small, ns not significant'
)
ax.set_xlabel('Beta')
ax.set_ylabel('Beta')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'pairwise_significance_accuracy.png', dpi=150)
print(f'Saved: pairwise_significance_accuracy.png')
plt.close('all')

# Print directional summary
print('\n--- Directional Summary: Test Accuracy ---')
print('Significant pairwise comparisons (A vs B shows which has higher accuracy):')
for _, row in acc_pairwise.iterrows():
	if row['p_significant']:
		direction = 'A > B' if row['cohens_d'] > 0 else 'B > A'
		d_mag = abs(row['cohens_d'])
		effect = 'large' if d_mag >= 0.8 else ('medium' if d_mag >= 0.5 else 'small')
		print(
			f'  beta={row["beta_a"]:.0e} vs beta={row["beta_b"]:.0e}: '
			f"{direction} (Cohen's d={row['cohens_d']:+.3f}, {effect})"
		)


# %%
# Pairwise significance matrix for compression
print('\n--- Pairwise Significance Matrix: Empirical Compression ---')
comp_pairwise = pairwise_beta_significance(
	df_no_baseline, 'final_empirical_compression', n_bootstrap=5000
)
sig_matrix_comp = pd.DataFrame(index=beta_vals, columns=beta_vals, dtype=bool)
for _, row in comp_pairwise.iterrows():
	sig_matrix_comp.loc[row['beta_a'], row['beta_b']] = row['p_significant']
	sig_matrix_comp.loc[row['beta_b'], row['beta_a']] = row['p_significant']
sig_matrix_comp = sig_matrix_comp.fillna(False)

# Create annotation matrix with significance levels and direction
annot_matrix_comp = pd.DataFrame('', index=beta_vals, columns=beta_vals)
for _, row in comp_pairwise.iterrows():
	a, b = row['beta_a'], row['beta_b']
	d = row['cohens_d']  # signed Cohen's d
	if row['p_significant']:
		if abs(d) >= 0.8:
			level = '***'
		elif abs(d) >= 0.5:
			level = '**'
		else:
			level = '*'
		direction = 'A>B' if d > 0 else 'A<B'
		annot_matrix_comp.loc[a, b] = f'{direction}\n{level}'
		annot_matrix_comp.loc[b, a] = f'{"B>A" if d > 0 else "B<A"}\n{level}'
	else:
		annot_matrix_comp.loc[a, b] = 'ns'
		annot_matrix_comp.loc[b, a] = 'ns'

fig, ax = plt.subplots(figsize=(12, 10))
sig_numeric_comp = sig_matrix_comp.astype(int)
sns.heatmap(
	sig_numeric_comp,
	annot=annot_matrix_comp,
	fmt='',
	cmap='RdYlGn_r',
	ax=ax,
	cbar_kws={'label': 'Significant (1=Yes, 0=No)'},
	linewidths=0.5,
	linecolor='gray',
	annot_kws={'fontsize': 8},
)
ax.set_title(
	'Pairwise Significance: Empirical Compression (95% Bootstrap CI)\n'
	'A>B/B>A = direction, *** large (|d|≥0.8), ** medium (|d|≥0.5), * small, ns not significant'
)
ax.set_xlabel('Beta')
ax.set_ylabel('Beta')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'pairwise_significance_compression.png', dpi=150)
print(f'Saved: pairwise_significance_compression.png')
plt.close('all')

# Print directional summary
print('\n--- Directional Summary: Empirical Compression ---')
print('Significant pairwise comparisons (A vs B shows which has higher compression):')
for _, row in comp_pairwise.iterrows():
	if row['p_significant']:
		direction = 'A > B' if row['cohens_d'] > 0 else 'B > A'
		d_mag = abs(row['cohens_d'])
		effect = 'large' if d_mag >= 0.8 else ('medium' if d_mag >= 0.5 else 'small')
		print(
			f'  beta={row["beta_a"]:.0e} vs beta={row["beta_b"]:.0e}: '
			f"{direction} (Cohen's d={row['cohens_d']:+.3f}, {effect})"
		)


# %%
# =============================================================================
# SECTION 2: LINEAR REGRESSION ANALYSIS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 2: LINEAR REGRESSION ANALYSIS')
print('=' * 60)

# Prepare features - exclude beta=0 (baseline) from log-based regression
df_regression, X, y_accuracy, y_compression = prepare_regression_data(df)
print(f'Using {len(X)} experiments (excluding baseline beta=0) for regression analysis')

# Model 1: Predict test_accuracy
lr_results = linear_regression_analysis(X, y_accuracy, name='Test Accuracy')

# Model 2: Predict final_empirical_compression
print('\n--- Model 2: Predicting Empirical Compression ---')
lr_comp_results = linear_regression_analysis(X, y_compression, name='Empirical Compression')


# %%
# Visualize regression results with residual analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Actual vs Predicted (Accuracy)
ax = axes[0, 0]
ax.scatter(y_accuracy, lr_results['y_pred'], alpha=0.5, edgecolors='black', linewidth=0.5, s=40)
ax.plot([y_accuracy.min(), y_accuracy.max()], [y_accuracy.min(), y_accuracy.max()], 'r--', lw=2)
ax.set_xlabel('Actual Test Accuracy')
ax.set_ylabel('Predicted Test Accuracy')
ax.set_title(f'Linear Regression: Test Accuracy\nR2 = {lr_results["r2"]:.4f}')
ax.grid(True, alpha=0.3)

# Top-right: Residuals (Accuracy)
ax = axes[0, 1]
residuals_acc = lr_results['residuals']
ax.scatter(lr_results['y_pred'], residuals_acc, alpha=0.5, edgecolors='black', linewidth=0.5, s=40)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Predicted Test Accuracy')
ax.set_ylabel('Residuals (Actual - Predicted)')
ax.set_title('Residual Plot: Test Accuracy')
ax.grid(True, alpha=0.3)

# Bottom-left: Residual distribution (Accuracy)
ax = axes[1, 0]
ax.hist(residuals_acc, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
ax.set_title(
	f'Residual Distribution\nMean={residuals_acc.mean():.2f}, Std={residuals_acc.std():.2f}'
)
ax.grid(True, alpha=0.3)

# Bottom-right: Feature relationship (log_beta vs accuracy)
ax = axes[1, 1]
scatter = ax.scatter(
	df_regression['log_beta'],
	df_regression['test_accuracy'],
	c=df_regression['test_accuracy'],
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=40,
	edgecolors='black',
	linewidth=0.3,
)
# Add regression line
z = np.polyfit(df_regression['log_beta'], df_regression['test_accuracy'], 1)
p = np.poly1d(z)
ax.plot(
	df_regression['log_beta'].sort_values(),
	p(df_regression['log_beta'].sort_values()),
	'r-',
	lw=2,
	label=f'y={z[0]:.2f}x+{z[1]:.2f}',
)
ax.set_xlabel('Log Beta')
ax.set_ylabel('Test Accuracy')
ax.set_title('Log Beta vs Test Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'regression_predictions.png', dpi=150)
print(f'\nSaved: regression_predictions.png')
plt.close('all')


# %%
# =============================================================================
# SECTION 3: NON-LINEAR MODELS (Random Forest, Gradient Boosting)
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 3: NON-LINEAR MODELS')
print('=' * 60)

# Prepare features with more variables
X_full = add_engineered_features(df_regression[['log_beta', 'bottleneck_width']])

# Model 3: Random Forest for test_accuracy
print('\n--- Random Forest: Test Accuracy ---')
rf_results = random_forest_analysis(X_full, y_accuracy, name='Test Accuracy')


# %%
# Model 4: Gradient Boosting for test_accuracy
print('\n--- Gradient Boosting: Test Accuracy ---')
gb_results = gradient_boosting_analysis(X_full, y_accuracy, name='Test Accuracy')


# %%
# Compare model performance
print('\n--- Model Comparison ---')
models_comparison = pd.DataFrame(
	{
		'Model': ['Linear Regression', 'Ridge', 'Random Forest', 'Gradient Boosting'],
		'CV_R2_Mean': [
			lr_results['cv_r2_mean'],
			lr_results['cv_r2_mean'],
			rf_results['cv_r2_mean'],
			gb_results['cv_r2_mean'],
		],
		'CV_R2_Std': [
			lr_results['cv_r2_std'],
			lr_results['cv_r2_std'],
			rf_results['cv_r2_std'],
			gb_results['cv_r2_std'],
		],
	}
)
print(models_comparison)

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
rf_importance = pd.DataFrame(
	{'Feature': X_full.columns, 'Importance': list(rf_results['feature_importances'].values())}
).sort_values('Importance', ascending=True)
ax.barh(rf_importance['Feature'], rf_importance['Importance'])
ax.set_xlabel('Importance')
ax.set_title('Random Forest Feature Importance')
ax.grid(True, alpha=0.3)

ax = axes[1]
gb_importance = pd.DataFrame(
	{'Feature': X_full.columns, 'Importance': list(gb_results['feature_importances'].values())}
).sort_values('Importance', ascending=True)
ax.barh(gb_importance['Feature'], gb_importance['Importance'])
ax.set_xlabel('Importance')
ax.set_title('Gradient Boosting Feature Importance')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
print(f'\nSaved: feature_importance.png')
plt.close('all')


# %%
# =============================================================================
# SECTION 4: UNSUPERVISED LEARNING - CLUSTERING
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 4: UNSUPERVISED LEARNING - CLUSTERING')
print('=' * 60)

# Prepare data for clustering (use regression data without NaN)
cluster_features = ['log_beta', 'bottleneck_width', 'test_accuracy', 'final_empirical_compression']
X_cluster = df_regression[cluster_features].copy()

# Standardize features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# K-Means clustering
print('\n--- K-Means Clustering ---')

# Find optimal k using elbow method AND silhouette score
from sklearn.metrics import silhouette_score

inertias = []
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
	kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
	labels = kmeans.fit_predict(X_cluster_scaled)
	inertias.append(kmeans.inertia_)
	silhouette_scores.append(silhouette_score(X_cluster_scaled, labels))

# Plot elbow curve with silhouette score
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal K')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score for Optimal K')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'elbow_method.png', dpi=150)
print(f'Saved: elbow_method.png')

# Print best k by silhouette score
best_k_idx = np.argmax(silhouette_scores)
print(
	f'Best k by silhouette score: {list(k_range)[best_k_idx]} (score={silhouette_scores[best_k_idx]:.4f})'
)
plt.close('all')


# %%
# Apply K-Means with optimal k (typically 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_regression['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Analyze cluster characteristics
print(f'\n--- Cluster Characteristics (k={optimal_k}) ---')
cluster_stats = df_regression.groupby('cluster')[cluster_features].agg(['mean', 'std'])
print(cluster_stats.round(3))


# %%
# Visualize clusters - improved with discrete legend
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Create discrete color map for clusters
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
cluster_labels = ['Low Acc, High Beta', 'High Acc, Low Beta', 'Medium Acc', 'Transition']

# Cluster visualization in different projections
ax = axes[0, 0]
for cluster_id in range(optimal_k):
	cluster_mask = df_regression['cluster'] == cluster_id
	ax.scatter(
		df_regression.loc[cluster_mask, 'log_beta'],
		df_regression.loc[cluster_mask, 'bottleneck_width'],
		c=[cluster_colors[cluster_id]],
		label=cluster_labels[cluster_id],
		alpha=0.6,
		s=50,
		edgecolors='black',
		linewidth=0.5,
	)
ax.set_xlabel('Log Beta')
ax.set_ylabel('Bottleneck Width')
ax.set_title('Clusters: Beta vs Width')
ax.set_yscale('log', base=2)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for cluster_id in range(optimal_k):
	cluster_mask = df_regression['cluster'] == cluster_id
	ax.scatter(
		df_regression.loc[cluster_mask, 'log_beta'],
		df_regression.loc[cluster_mask, 'test_accuracy'],
		c=[cluster_colors[cluster_id]],
		alpha=0.6,
		s=50,
		edgecolors='black',
		linewidth=0.5,
	)
ax.set_xlabel('Log Beta')
ax.set_ylabel('Test Accuracy')
ax.set_title('Clusters: Beta vs Accuracy')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for cluster_id in range(optimal_k):
	cluster_mask = df_regression['cluster'] == cluster_id
	ax.scatter(
		df_regression.loc[cluster_mask, 'bottleneck_width'],
		df_regression.loc[cluster_mask, 'test_accuracy'],
		c=[cluster_colors[cluster_id]],
		alpha=0.6,
		s=50,
		edgecolors='black',
		linewidth=0.5,
	)
ax.set_xlabel('Bottleneck Width')
ax.set_ylabel('Test Accuracy')
ax.set_title('Clusters: Width vs Accuracy')
ax.set_xscale('log', base=2)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for cluster_id in range(optimal_k):
	cluster_mask = df_regression['cluster'] == cluster_id
	ax.scatter(
		df_regression.loc[cluster_mask, 'final_empirical_compression'],
		df_regression.loc[cluster_mask, 'test_accuracy'],
		c=[cluster_colors[cluster_id]],
		alpha=0.6,
		s=50,
		edgecolors='black',
		linewidth=0.5,
	)
ax.set_xlabel('Empirical Compression')
ax.set_ylabel('Test Accuracy')
ax.set_title('Clusters: Compression vs Accuracy')
ax.grid(True, alpha=0.3)

plt.tight_layout()
try:
	plt.savefig(OUTPUT_DIR / 'cluster_visualization.png', dpi=150)
	print(f'Saved: cluster_visualization.png')
except Exception as e:
	print(f'Warning: Could not save cluster_visualization.png: {e}')
plt.close('all')


# %%
# DBSCAN for density-based clustering
print('\n--- DBSCAN Clustering ---')
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_regression['dbscan_cluster'] = dbscan.fit_predict(X_cluster_scaled)

n_clusters_dbscan = len(set(df_regression['dbscan_cluster'])) - (
	1 if -1 in df_regression['dbscan_cluster'].values else 0
)
n_noise = list(df_regression['dbscan_cluster']).count(-1)
print(f'Number of clusters: {n_clusters_dbscan}')
print(f'Number of noise points: {n_noise}')


# %%
# =============================================================================
# SECTION 5: PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 5: PRINCIPAL COMPONENT ANALYSIS')
print('=' * 60)

# PCA on all numerical features
pca_features = [
	'log_beta',
	'bottleneck_width',
	'test_accuracy',
	'final_empirical_compression',
	'final_val_loss',
	'final_effective_capacity_utilization',
]
# Drop rows with NaN or infinite values for PCA
X_pca = df[pca_features].copy()
X_pca = X_pca.replace([np.inf, -np.inf], np.nan)
X_pca = X_pca.dropna()
print(f'\nPCA samples after cleaning: {len(X_pca)} (dropped {len(df) - len(X_pca)} rows)')

X_pca_scaled = scaler.fit_transform(X_pca)

pca = PCA()
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# Explained variance
print('\n--- Explained Variance Ratio ---')
for i, var in enumerate(pca.explained_variance_ratio_):
	print(f'PC{i + 1}: {var:.4f}')

print(f'\nCumulative Explained Variance:')
cumsum = np.cumsum(pca.explained_variance_ratio_)
for i, c in enumerate(cumsum):
	print(f'PC{i + 1}: {c:.4f}')


# %%
# PCA loadings
print('\n--- PCA Loadings ---')
loadings = pd.DataFrame(
	pca.components_.T,
	columns=[f'PC{i + 1}' for i in range(len(pca.components_))],
	index=pca_features,
)
print(loadings.round(3))

# Visualize PCA - improved
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax = axes[0]
ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Scree Plot')
ax.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
ax.legend()
ax.grid(True, alpha=0.3)

# PCA projection with better visibility
ax = axes[1]
accuracy_colors = X_pca['test_accuracy'].values

# Add jitter for better visibility
np.random.seed(42)
jitter = np.random.normal(0, 0.3, size=X_pca_transformed.shape)
X_pca_jittered = X_pca_transformed + jitter

scatter = ax.scatter(
	X_pca_jittered[:, 0],
	X_pca_jittered[:, 1],
	c=accuracy_colors,
	cmap=BOTTLENECK_CMAP,
	alpha=0.7,
	s=60,
	edgecolors='black',
	linewidth=0.3,
)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax.set_title('PCA Projection (colored by accuracy)')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Test Accuracy')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pca_analysis.png', dpi=150)
print(f'\nSaved: pca_analysis.png')
plt.close('all')


# %%
# =============================================================================
# SECTION 6: DECISION TREE FOR INTERPRETABILITY
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 6: DECISION TREE FOR INTERPRETABILITY')
print('=' * 60)

# Train a shallow decision tree
dt = DecisionTreeRegressor(max_depth=4, min_samples_split=10, random_state=42)
dt.fit(X_full, y_accuracy)
print(f'Decision Tree R2: {r2_score(y_accuracy, dt.predict(X_full)):.4f}')

# Feature importance from decision tree
print('\nDecision Tree Feature Importance:')
for feat, imp in zip(X_full.columns, dt.feature_importances_):
	print(f'  {feat}: {imp:.4f}')


# %%
# Visualize decision tree
fig, ax = plt.subplots(figsize=(16, 10))
plot_tree(
	dt,
	feature_names=X_full.columns,
	filled=True,
	rounded=True,
	precision=2,
	ax=ax,
	fontsize=10,
)
ax.set_title('Decision Tree for Test Accuracy Prediction')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'decision_tree.png', dpi=150, bbox_inches='tight')
print(f'\nSaved: decision_tree.png')
plt.close('all')


# %%
# =============================================================================
# SECTION 7: ANALYSIS BY MODEL ARCHITECTURE
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 7: ANALYSIS BY MODEL ARCHITECTURE')
print('=' * 60)

# Encode model architecture
le = LabelEncoder()
df['model_arch_encoded'] = le.fit_transform(df['model_arch'])

# Analyze each architecture separately
for model_arch in df['model_arch'].unique():
	model_df = df[df['model_arch'] == model_arch]

	print(f'\n--- {model_arch} ---')
	print(f'  Samples: {len(model_df)}')

	# Correlations for this architecture
	corr_beta = model_df['log_beta'].corr(model_df['test_accuracy'])
	corr_width = model_df['bottleneck_width'].corr(model_df['test_accuracy'])
	print(f'  Correlation (log_beta, accuracy): {corr_beta:.4f}')
	print(f'  Correlation (width, accuracy): {corr_width:.4f}')

	# Fit simple model
	X_model = model_df[['log_beta', 'bottleneck_width']]
	y_model = model_df['test_accuracy']

	rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
	rf_model.fit(X_model, y_model)
	cv_model = cross_val_score(rf_model, X_model, y_model, cv=5, scoring='r2')

	print(f'  RF CV R2: {cv_model.mean():.4f} (+/- {cv_model.std() * 2:.4f})')


# %%
# =============================================================================
# SECTION 8: OPTIMAL CONFIGURATION ANALYSIS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 8: OPTIMAL CONFIGURATION ANALYSIS')
print('=' * 60)

# Find top configurations
top_n = 20
top_configs = df.nlargest(top_n, 'test_accuracy')

print(f'\n--- Top {top_n} Configurations by Test Accuracy ---')
print(
	top_configs[
		['model_arch', 'bottleneck_width', 'beta', 'test_accuracy', 'final_empirical_compression']
	].to_string()
)

# Analyze common patterns in top configurations
print('\n--- Patterns in Top Configurations ---')
print(f'Average log_beta: {np.log10(top_configs["beta"]).mean():.4f}')
print(f'Average bottleneck_width: {top_configs["bottleneck_width"].mean():.4f}')
print(f'Average compression: {top_configs["final_empirical_compression"].mean():.4f}')

# Compare with bottom configurations
bottom_configs = df.nsmallest(top_n, 'test_accuracy')
print(f'\n--- Bottom {top_n} Configurations ---')
print(f'Average log_beta: {np.log10(bottom_configs["beta"]).mean():.4f}')
print(f'Average bottleneck_width: {bottom_configs["bottleneck_width"].mean():.4f}')


# %%
# Visualize optimal region
try:
	fig, ax = plt.subplots(figsize=(10, 8))

	# Filter out -inf log_beta values for plotting
	valid_plot = df['log_beta'].replace([np.inf, -np.inf], np.nan).notna()
	ax.scatter(
		df.loc[valid_plot, 'log_beta'],
		df.loc[valid_plot, 'bottleneck_width'],
		c=df.loc[valid_plot, 'test_accuracy'],
		cmap='YlOrRd',
		alpha=0.6,
		s=30,
	)

	# Top configurations (filter -inf)
	top_log_beta = np.log10(top_configs['beta']).replace([np.inf, -np.inf], np.nan)
	top_valid = top_log_beta.notna()
	ax.scatter(
		top_log_beta[top_valid],
		top_configs.loc[top_valid, 'bottleneck_width'],
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
	plt.colorbar(ax.collections[0], ax=ax, label='Test Accuracy')

	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / 'optimal_region.png', dpi=150)
	print(f'\nSaved: optimal_region.png')
	plt.close('all')
except Exception as e:
	print(f'\nWarning: Could not create optimal_region.png: {e}')
	plt.close('all')


# %%
# =============================================================================
# SECTION 9: ACCURACY-COMPRESSION TRADE-OFF
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 9: ACCURACY-COMPRESSION TRADE-OFF')
print('=' * 60)


def find_pareto_frontier(df_results, col1='test_accuracy', col2='final_empirical_compression'):
	"""Find Pareto-optimal configurations."""
	pareto = df_results.copy()
	pareto = pareto.sort_values([col1, col2], ascending=[False, True])
	pareto_frontier = []
	max_col2 = -np.inf
	for _, row in pareto.iterrows():
		if row[col2] > max_col2:
			pareto_frontier.append(row)
			max_col2 = row[col2]
	return pd.DataFrame(pareto_frontier)


pareto_df = find_pareto_frontier(df)
print(f'\nPareto-optimal configurations: {len(pareto_df)}')
print(
	pareto_df[
		['model_arch', 'bottleneck_width', 'beta', 'test_accuracy', 'final_empirical_compression']
	].head(10)
)


# %%
# Visualize Pareto frontier
fig, ax = plt.subplots(figsize=(10, 8))

# All configurations
ax.scatter(
	df['final_empirical_compression'],
	df['test_accuracy'],
	c='gray',
	alpha=0.3,
	s=30,
	label='All configurations',
)

# Pareto-optimal
ax.scatter(
	pareto_df['final_empirical_compression'],
	pareto_df['test_accuracy'],
	c='red',
	s=80,
	edgecolors='black',
	linewidth=1,
	label='Pareto-optimal',
	zorder=5,
)

ax.set_xlabel('Final Empirical Compression')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Accuracy-Compression Trade-Off (Pareto Frontier)')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pareto_frontier.png', dpi=150)
print(f'\nSaved: pareto_frontier.png')
plt.close('all')


# %%
# =============================================================================
# SECTION 10: SUMMARY
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 10: SUMMARY')
print('=' * 60)

print('\n--- Summary Statistics ---')
print(f'Total experiments: {len(df)}')
print(f'Best accuracy: {df["test_accuracy"].max():.4f}')
print(f'Mean accuracy: {df["test_accuracy"].mean():.4f}')
print(f'Best compression: {df["final_empirical_compression"].min():.4f}')
print(f'Mean compression: {df["final_empirical_compression"].mean():.4f}')

print('\n--- Key Findings ---')
print(f'\n1. CORRELATION PATTERNS:')
print(f'   - Log-beta correlation with accuracy: {log_beta_accuracy_corr:.4f}')
print(f'   - Width correlation with accuracy: {width_accuracy_corr:.4f}')

print(f'\n2. MODEL PERFORMANCE:')
print(f'   - Linear Regression CV R2: {lr_results["cv_r2_mean"]:.4f}')
print(f'   - Random Forest CV R2: {rf_results["cv_r2_mean"]:.4f}')
improvement = (
	(rf_results['cv_r2_mean'] - lr_results['cv_r2_mean']) / lr_results['cv_r2_mean'] * 100
)
print(f'   - Non-linear models improve prediction by: {improvement:.1f}%')

print(f'\n3. OPTIMAL CONFIGURATIONS:')
print(f'   - Average log_beta in top 20: {np.log10(top_configs["beta"]).mean():.4f}')
print(f'   - Average width in top 20: {top_configs["bottleneck_width"].mean():.1f}')

print(f'\n4. PARETO-OPTIMAL SOLUTIONS: {len(pareto_df)} configurations found')

print(f'\n5. CLUSTER STRUCTURE: {optimal_k} distinct regimes identified')

print(f'\n6. STATISTICAL RELATIONSHIP ANALYSIS:')
print(f'   --- Accuracy ---')
print(
	f'   - Spearman log(beta) vs accuracy: r={acc_rel["beta_spearman_corr"]:.4f}, p={acc_rel["beta_spearman_p"]:.2e}'
)
print(
	f'   - Spearman width vs accuracy:     r={acc_rel["width_spearman_corr"]:.4f}, p={acc_rel["width_spearman_p"]:.2e}'
)
print(
	f'   - Partial log(beta)|width:        r={acc_rel["partial_beta_corr"]:.4f}, p={acc_rel["partial_beta_p"]:.2e}'
)
print(
	f'   - Partial width|log(beta):        r={acc_rel["partial_width_corr"]:.4f}, p={acc_rel["partial_width_p"]:.2e}'
)
anova_acc = acc_rel['anova_table']
for effect, vals in anova_acc.items():
	print(f'   - ANOVA {effect}: F={vals["F"]:.2f}, p={vals["p"]:.2e}')
print(f'   - Model R2: {acc_rel["r_squared"]:.4f}')

print(f'   --- Compression ---')
print(
	f'   - Spearman log(beta) vs compression: r={comp_rel["beta_spearman_corr"]:.4f}, p={comp_rel["beta_spearman_p"]:.2e}'
)
print(
	f'   - Spearman width vs compression:     r={comp_rel["width_spearman_corr"]:.4f}, p={comp_rel["width_spearman_p"]:.2e}'
)
print(
	f'   - Partial log(beta)|width:           r={comp_rel["partial_beta_corr"]:.4f}, p={comp_rel["partial_beta_p"]:.2e}'
)
print(
	f'   - Partial width|log(beta):           r={comp_rel["partial_width_corr"]:.4f}, p={comp_rel["partial_width_p"]:.2e}'
)
anova_comp = comp_rel['anova_table']
for effect, vals in anova_comp.items():
	print(f'   - ANOVA {effect}: F={vals["F"]:.2f}, p={vals["p"]:.2e}')
print(f'   - Model R2: {comp_rel["r_squared"]:.4f}')


# %%
# Save summary
anova_acc_text = '\n'.join(
	[
		f'   - ANOVA {eff}: F={v["F"]:.2f}, p={v["p"]:.2e}'
		for eff, v in acc_rel['anova_table'].items()
	]
)
anova_comp_text = '\n'.join(
	[
		f'   - ANOVA {eff}: F={v["F"]:.2f}, p={v["p"]:.2e}'
		for eff, v in comp_rel['anova_table'].items()
	]
)

summary_text = f"""
EfficientNet Numerical Analysis Summary
========================================

Total experiments: {len(df)}
Best accuracy: {df['test_accuracy'].max():.4f}
Mean accuracy: {df['test_accuracy'].mean():.4f}
Best compression: {df['final_empirical_compression'].min():.4f}
Mean compression: {df['final_empirical_compression'].mean():.4f}

Key Findings:
1. Log-beta correlation with accuracy: {log_beta_accuracy_corr:.4f}
2. Width correlation with accuracy: {width_accuracy_corr:.4f}
3. Linear Regression CV R2: {lr_results['cv_r2_mean']:.4f}
4. Random Forest CV R2: {rf_results['cv_r2_mean']:.4f}
5. Pareto-optimal configurations: {len(pareto_df)}
6. Clusters identified: {optimal_k}

Statistical Relationship Analysis:
  Accuracy:
  - Spearman log(beta) vs accuracy: r={acc_rel['beta_spearman_corr']:.4f}, p={acc_rel['beta_spearman_p']:.2e}
  - Spearman width vs accuracy:     r={acc_rel['width_spearman_corr']:.4f}, p={acc_rel['width_spearman_p']:.2e}
  - Partial log(beta)|width:        r={acc_rel['partial_beta_corr']:.4f}, p={acc_rel['partial_beta_p']:.2e}
  - Partial width|log(beta):        r={acc_rel['partial_width_corr']:.4f}, p={acc_rel['partial_width_p']:.2e}
{anova_acc_text}
  - Model R2: {acc_rel['r_squared']:.4f}

  Compression:
  - Spearman log(beta) vs compression: r={comp_rel['beta_spearman_corr']:.4f}, p={comp_rel['beta_spearman_p']:.2e}
  - Spearman width vs compression:     r={comp_rel['width_spearman_corr']:.4f}, p={comp_rel['width_spearman_p']:.2e}
  - Partial log(beta)|width:           r={comp_rel['partial_beta_corr']:.4f}, p={comp_rel['partial_beta_p']:.2e}
  - Partial width|log(beta):           r={comp_rel['partial_width_corr']:.4f}, p={comp_rel['partial_width_p']:.2e}
{anova_comp_text}
  - Model R2: {comp_rel['r_squared']:.4f}
"""

with open(OUTPUT_DIR / 'analysis_summary.txt', 'w', encoding='utf-8') as f:
	f.write(summary_text)

print(f'\n\nSaved: analysis_summary.txt')

print(f'\nAnalysis complete! Results saved to: {OUTPUT_DIR}')
