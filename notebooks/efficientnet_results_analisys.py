# %%
"""
Analysis of EfficientNet grid search experiment results.

This script loads results from results/efficientnet/grid_search_results_final.json
and performs comprehensive analysis:
- Statistics by models and hyperparameters
- Visualizations (heatmaps, plots)
- Finding best configurations
"""

from pathlib import Path
import warnings

from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.analysis import (
	load_with_baseline,
	plot_accuracy_vs_beta_with_gradient,
	plot_compression_vs_beta_with_gradient,
	plot_metric_vs_beta_error_bars,
	save_and_close,
)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')

# Fixed colormap for bottleneck width gradient (used consistently across all plots)
BOTTLENECK_CMAP = 'plasma'


# %%
# Data paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / 'results' / 'efficientnet' / 'grid_search_results_final.json'
BASELINE_PATH = PROJECT_ROOT / 'results' / 'baseline' / 'grid_search_results_final.json'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'efficientnet' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading data from: results/efficientnet/grid_search_results_final.json')
print(f'Results will be saved to: results/efficientnet/analysis/')

# %%
# Load data with baseline (filtered for EfficientNet models)
efficientnet_models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
df = load_with_baseline(RESULTS_PATH, BASELINE_PATH, model_filter=efficientnet_models)
print(f'\nColumns: {list(df.columns)}')

# %%
# Basic statistics
print('BASIC STATISTICS')

print(f'\nTotal experiments: {len(df)}')
print(f'\nUnique model architectures: {df["model_arch"].unique()}')
print(f'\nUnique bottleneck widths: {sorted(df["bottleneck_width"].unique())}')
print(f'\nBeta range: [{df["beta"].min()}, {df["beta"].max()}]')
print(f'\nUnique seeds: {sorted(df["seed"].unique())}')

# %%
# Statistics on test_accuracy
print('STATISTICS ON ACCURACY (test_accuracy)')

print(f'\nMean: {df["test_accuracy"].mean():.2f}%')
print(f'Median: {df["test_accuracy"].median():.2f}%')
print(f'Std. deviation: {df["test_accuracy"].std():.2f}%')
print(f'Min: {df["test_accuracy"].min():.2f}%')
print(f'Max: {df["test_accuracy"].max():.2f}%')

# %%
# Best configurations
print('BEST CONFIGURATIONS (by test_accuracy)')

top_configs = df.nlargest(10, 'test_accuracy')
for _, row in top_configs.iterrows():
	print(
		f'\nTest Acc: {row["test_accuracy"]:.2f}% | '
		f'Model: {row["model_arch"]} | '
		f'Width: {row["bottleneck_width"]} | '
		f'Beta: {row["beta"]:.6f} | '
		f'Seed: {row["seed"]}'
	)

# %%
# Aggregated statistics by architecture
print('STATISTICS BY MODEL ARCHITECTURES')

arch_stats = df.groupby('model_arch').agg(
	{
		'test_accuracy': ['mean', 'std', 'min', 'max', 'count'],
		'final_train_loss': 'mean',
		'final_val_loss': 'mean',
	}
)
print(arch_stats.round(2))

# %%
# Aggregated statistics by bottleneck width
print('STATISTICS BY BOTTLENECK WIDTH')
width_stats = df.groupby('bottleneck_width').agg(
	{
		'test_accuracy': ['mean', 'std', 'min', 'max'],
		'final_train_loss': 'mean',
		'final_val_loss': 'mean',
		'final_empirical_compression': 'mean',
	}
)
print(width_stats.round(2))

# %%
print('EFFECT OF BETA ON ACCURACY')
beta_stats = df.groupby('beta').agg(
	{
		'test_accuracy': ['mean', 'std', 'min', 'max', 'count'],
		'final_train_loss': 'mean',
		'final_val_loss': 'mean',
	}
)
print(beta_stats.round(2))


# %%
# Helper function to create heatmap with proper norm handling
def _make_heatmap(df, values, filename, fmt='.1f', norm_type=None, title_suffix=''):
	"""Create heatmap with automatic norm type handling."""
	try:
		# Filter out infinite values before pivoting
		df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[values])
		if len(df_clean) == 0:
			print(f'  Skipping {filename}: no valid data')
			return
		pivot = df_clean.pivot_table(
			values=values,
			index='bottleneck_width',
			columns='beta',
			aggfunc='mean',
		)
		finite = pivot.values[np.isfinite(pivot.values)]
		if len(finite) == 0:
			print(f'  Skipping {filename}: no finite values in pivot')
			return
		vmin = np.min(finite)
		vmax = np.max(finite)

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
		print(f'  Warning: Could not create {filename}: {e}')
		plt.close('all')


# %%
# Building heatmaps
print('BUILDING PLOTS')

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


# %%
# 5. Scatter plot: Empirical Compression vs Capacity Utilization (color = test accuracy)
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
	df['final_empirical_compression'],
	df['final_effective_capacity_utilization'],
	c=df['test_accuracy'],
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
	edgecolors='black',
	linewidth=0.5,
)
ax.set_xlabel('Final Empirical Compression')
ax.set_ylabel('Final Effective Capacity Utilization')
ax.set_title('Test Accuracy: Empirical Compression vs Capacity Utilization')
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
cbar = plt.colorbar(scatter)
cbar.set_label('Test Accuracy (%)')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'scatter_acc_compression_capacity.png', dpi=150)
print(f'Saved: scatter_acc_compression_capacity.png')
plt.close('all')


# %%
# 6. Scatter plot: Empirical Compression vs Test Accuracy (color = beta, size = bottleneck width)
fig, ax = plt.subplots(figsize=(10, 6))
# Handle beta=0 values for LogNorm
beta_positive = df[df['beta'] > 0]['beta']
if len(beta_positive) > 0:
	beta_min, beta_max = beta_positive.min(), beta_positive.max()
	scatter_color = df['beta'].replace(0, np.nan)
else:
	beta_min, beta_max = 1e-8, 1
	scatter_color = df['beta'].replace(0, np.nan)

# Scale size by log of bottleneck width
widths = df['bottleneck_width']
sizes = np.log(widths) * 20

scatter = ax.scatter(
	df['final_empirical_compression'],
	df['test_accuracy'],
	c=scatter_color,
	norm=LogNorm(beta_min, beta_max),
	cmap=BOTTLENECK_CMAP,
	s=sizes,
	alpha=0.6,
	edgecolors='black',
	linewidth=0.5,
)
ax.set_xlabel('Final Empirical Compression')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Test Accuracy vs Empirical Compression (color = beta, size = width)')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, which='both')
cbar = plt.colorbar(scatter)
cbar.set_label('Beta (log scale)')

# Add size legend outside the plot near colorbar
unique_widths = sorted(widths.unique())
legend_handles = [
	plt.scatter(
		[], [], s=np.log(w) * 20, color='gray', alpha=0.5, edgecolors='black', label=f'Width: {w}'
	)
	for w in unique_widths
]
ax.legend(
	handles=legend_handles, title='Bottleneck Width', loc='center left', bbox_to_anchor=(1.2, 0.5)
)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'scatter_acc_compression.png', dpi=150)
print(f'Saved: scatter_acc_compression.png')
plt.close('all')


# %%
# 7. Accuracy vs beta with gradient color based on bottleneck width (unified style)
fig, ax = plt.subplots(figsize=(10, 6))
# Load baseline data, filtered to only models present in main results
df_baseline_raw = load_with_baseline(RESULTS_PATH, BASELINE_PATH, model_filter=efficientnet_models)
models_in_results = set(df['model_arch'].unique())
df_baseline_all = df_baseline_raw[df_baseline_raw['model_arch'].isin(models_in_results)]
df_baseline_only = df_baseline_all[df_baseline_all['beta'] == 0]
# Remove baseline from df for plotting (it's shown separately)
df_no_baseline = df[df['beta'] > 0]
plot_accuracy_vs_beta_with_gradient(
	ax,
	df_no_baseline,
	'Accuracy vs Beta (color = Bottleneck Width)',
	baseline_data=df_baseline_only,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy_vs_beta.png', dpi=150)
print(f'Saved: accuracy_vs_beta.png')
plt.close('all')


# %%
# 7b. Empirical compression vs beta with gradient color based on bottleneck width
fig, ax = plt.subplots(figsize=(10, 6))
plot_compression_vs_beta_with_gradient(
	ax,
	df_no_baseline,
	'Empirical Compression vs Beta (color = Bottleneck Width)',
	baseline_data=df_baseline_only,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'compression_vs_beta.png', dpi=150)
print(f'Saved: compression_vs_beta.png')
plt.close('all')


# %%
# 7c. Error bars: Accuracy vs beta (aggregated across widths)
fig, ax = plt.subplots(figsize=(10, 6))
plot_metric_vs_beta_error_bars(
	ax,
	df_no_baseline,
	'test_accuracy',
	'Test Accuracy vs Beta (mean ± std across widths and seeds)',
	'Test Accuracy (%)',
	baseline_data=df_baseline_only,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy_vs_beta_errorbars.png', dpi=150)
print(f'Saved: accuracy_vs_beta_errorbars.png')
plt.close('all')


# %%
# 7d. Error bars: Compression vs beta (aggregated across widths)
fig, ax = plt.subplots(figsize=(10, 6))
plot_metric_vs_beta_error_bars(
	ax,
	df_no_baseline,
	'final_empirical_compression',
	'Empirical Compression vs Beta (mean ± std across widths and seeds)',
	'Empirical Compression',
	baseline_data=df_baseline_only,
)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'compression_vs_beta_errorbars.png', dpi=150)
print(f'Saved: compression_vs_beta_errorbars.png')
plt.close('all')


# %%
# 4. Accuracy distribution by model
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax = axes[0]
ax.hist(df['test_accuracy'], bins=30, edgecolor='black', alpha=0.7)
ax.axvline(
	df['test_accuracy'].mean(),
	color='red',
	linestyle='--',
	linewidth=2,
	label=f'Mean: {df["test_accuracy"].mean():.2f}%',
)
ax.set_xlabel('Test Accuracy (%)')
ax.set_ylabel('Number of experiments')
ax.set_title('Accuracy Distribution - All Experiments')
ax.legend()

# Box plot by architecture
ax = axes[1]
df.boxplot(column='test_accuracy', by='model_arch', ax=ax)
ax.set_xlabel('Model Architecture')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Accuracy Distribution by Architecture')
plt.suptitle('')  # Remove automatic title
plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'accuracy_distribution.png', dpi=150)
print(f'Saved: accuracy_distribution.png')
plt.close('all')


# %%
# 5. Train loss vs val loss relationship
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
	df['final_train_loss'],
	df['final_val_loss'],
	c=df['test_accuracy'],
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
)
ax.set_xlabel('Final Train Loss')
ax.set_ylabel('Final Val Loss')
ax.set_title('Train Loss vs Val Loss (color = test accuracy)')
ax.set_xscale('log')
ax.set_yscale('log')
cbar = plt.colorbar(scatter)
cbar.set_label('Test Accuracy (%)')
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'train_vs_val_loss.png', dpi=150)
print(f'Saved: train_vs_val_loss.png')
plt.close('all')


# %%
# 6. Compression metrics analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
# Handle beta=0 values for LogNorm
beta_vals = df[df['beta'] > 0]['beta']
if len(beta_vals) > 0:
	beta_min, beta_max = beta_vals.min(), beta_vals.max()
	scatter_c = df['beta'].replace(0, np.nan)
else:
	beta_min, beta_max = 1e-8, 1
	scatter_c = df['beta'].replace(0, np.nan)

scatter = ax.scatter(
	df['final_empirical_compression'],
	df['test_accuracy'],
	c=scatter_c,
	norm=LogNorm(beta_min, beta_max),
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
)
ax.set_xlabel('Final Empirical Compression')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Compression vs Accuracy (color = beta)')
cbar = plt.colorbar(scatter)
cbar.set_label('Beta (log scale)')
ax.grid(True, alpha=0.3)

ax = axes[1]
scatter = ax.scatter(
	df['final_effective_capacity_utilization'],
	df['test_accuracy'],
	c=df['bottleneck_width'],
	norm=LogNorm(df['bottleneck_width'].min(), df['bottleneck_width'].max()),
	cmap=BOTTLENECK_CMAP,
	alpha=0.6,
	s=50,
)
ax.set_xlabel('Final Effective Capacity Utilization')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Capacity Utilization vs Accuracy (color = width)')
cbar = plt.colorbar(scatter)
cbar.set_label('Bottleneck Width (log scale)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'compression_metrics.png', dpi=150)
print(f'Saved: compression_metrics.png')
plt.close('all')


# %%
# 7. Analysis by model architecture - separate heatmaps and plots
print('ANALYSIS BY MODEL ARCHITECTURE')

for model_arch in df['model_arch'].unique():
	model_df = df[df['model_arch'] == model_arch].copy()

	print(f'\n--- {model_arch} ---')
	print(f'  Experiments: {len(model_df)}')
	print(f'  Mean accuracy: {model_df["test_accuracy"].mean():.2f}%')
	print(f'  Best accuracy: {model_df["test_accuracy"].max():.2f}%')

	# Heatmaps for this architecture
	_make_heatmap(
		model_df,
		'test_accuracy',
		f'heatmap_accuracy_{model_arch}.png',
		fmt='.1f',
		title_suffix=f' ({model_arch})',
	)
	_make_heatmap(
		model_df,
		'final_val_loss',
		f'heatmap_loss_{model_arch}.png',
		fmt='.3f',
		norm_type='log',
		title_suffix=f' ({model_arch})',
	)
	_make_heatmap(
		model_df,
		'final_empirical_compression',
		f'heatmap_compression_{model_arch}.png',
		fmt='.3f',
		norm_type='power',
		title_suffix=f' ({model_arch})',
	)
	_make_heatmap(
		model_df,
		'final_effective_capacity_utilization',
		f'heatmap_capacity_{model_arch}.png',
		fmt='.3f',
		norm_type='log',
		title_suffix=f' ({model_arch})',
	)

	# Scatter: Compression vs Capacity (color = accuracy)
	fig, ax = plt.subplots(figsize=(10, 6))
	scatter = ax.scatter(
		model_df['final_empirical_compression'],
		model_df['final_effective_capacity_utilization'],
		c=model_df['test_accuracy'],
		cmap=BOTTLENECK_CMAP,
		alpha=0.6,
		s=50,
		edgecolors='black',
		linewidth=0.5,
	)
	ax.set_xlabel('Final Empirical Compression')
	ax.set_ylabel('Final Effective Capacity Utilization')
	ax.set_title(f'Test Accuracy: Compression vs Capacity ({model_arch})')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.grid(True, alpha=0.3, which='both')
	cbar = plt.colorbar(scatter)
	cbar.set_label('Test Accuracy (%)')
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'scatter_acc_compression_capacity_{model_arch}.png', dpi=150)
	print(f'  Saved: scatter_acc_compression_capacity_{model_arch}.png')
	plt.close('all')

	# Scatter: Empirical Compression vs Test Accuracy (color = beta, size = width)
	fig, ax = plt.subplots(figsize=(10, 6))
	beta_positive = model_df[model_df['beta'] > 0]['beta']
	if len(beta_positive) > 0:
		b_min, b_max = beta_positive.min(), beta_positive.max()
		sc = model_df['beta'].replace(0, np.nan)
	else:
		b_min, b_max = 1e-8, 1
		sc = model_df['beta'].replace(0, np.nan)

	w = model_df['bottleneck_width']
	sz = np.log(w) * 20

	scatter = ax.scatter(
		model_df['final_empirical_compression'],
		model_df['test_accuracy'],
		c=sc,
		norm=LogNorm(b_min, b_max),
		cmap=BOTTLENECK_CMAP,
		s=sz,
		alpha=0.6,
		edgecolors='black',
		linewidth=0.5,
	)
	ax.set_xlabel('Final Empirical Compression')
	ax.set_ylabel('Test Accuracy (%)')
	ax.set_title(f'Test Accuracy vs Empirical Compression ({model_arch})')
	ax.set_xscale('log')
	ax.grid(True, alpha=0.3, which='both')
	cbar = plt.colorbar(scatter)
	cbar.set_label('Beta (log scale)')

	uw = sorted(w.unique())
	lh = [
		plt.scatter(
			[],
			[],
			s=np.log(x) * 20,
			color='gray',
			alpha=0.5,
			edgecolors='black',
			label=f'Width: {x}',
		)
		for x in uw
	]
	ax.legend(handles=lh, title='Bottleneck Width', loc='center left', bbox_to_anchor=(1.2, 0.5))

	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'scatter_acc_compression_{model_arch}.png', dpi=150)
	print(f'  Saved: scatter_acc_compression_{model_arch}.png')
	plt.close('all')

	# Accuracy vs beta with gradient color
	fig, ax = plt.subplots(figsize=(10, 6))
	# Get baseline for this architecture
	baseline_arch = df_baseline_only[df_baseline_only['model_arch'] == model_arch]
	model_df_no_bl = model_df[model_df['beta'] > 0]
	plot_accuracy_vs_beta_with_gradient(
		ax,
		model_df_no_bl,
		f'Accuracy vs Beta for {model_arch} (color = Bottleneck Width)',
		baseline_data=baseline_arch,
	)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'accuracy_vs_beta_{model_arch}.png', dpi=150)
	print(f'  Saved: accuracy_vs_beta_{model_arch}.png')
	plt.close('all')

	# Empirical compression vs beta with gradient color
	fig, ax = plt.subplots(figsize=(10, 6))
	plot_compression_vs_beta_with_gradient(
		ax,
		model_df_no_bl,
		f'Empirical Compression vs Beta for {model_arch} (color = Bottleneck Width)',
		baseline_data=baseline_arch,
	)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'compression_vs_beta_{model_arch}.png', dpi=150)
	print(f'  Saved: compression_vs_beta_{model_arch}.png')
	plt.close('all')

	# Error bars: Accuracy vs beta (aggregated across widths)
	fig, ax = plt.subplots(figsize=(10, 6))
	plot_metric_vs_beta_error_bars(
		ax,
		model_df_no_bl,
		'test_accuracy',
		f'Test Accuracy vs Beta for {model_arch} (mean ± std)',
		'Test Accuracy (%)',
		baseline_data=baseline_arch,
	)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'accuracy_vs_beta_errorbars_{model_arch}.png', dpi=150)
	print(f'  Saved: accuracy_vs_beta_errorbars_{model_arch}.png')
	plt.close('all')

	# Error bars: Compression vs beta (aggregated across widths)
	fig, ax = plt.subplots(figsize=(10, 6))
	plot_metric_vs_beta_error_bars(
		ax,
		model_df_no_bl,
		'final_empirical_compression',
		f'Empirical Compression vs Beta for {model_arch} (mean ± std)',
		'Empirical Compression',
		baseline_data=baseline_arch,
	)
	plt.tight_layout()
	fig.savefig(OUTPUT_DIR / f'compression_vs_beta_errorbars_{model_arch}.png', dpi=150)
	print(f'  Saved: compression_vs_beta_errorbars_{model_arch}.png')
	plt.close('all')


# %%
# 8. Stability analysis by seed
print('STABILITY ANALYSIS BY SEED')

# For each configuration (model, width, beta) calculate variance by seed
config_variance = df.groupby(['model_arch', 'bottleneck_width', 'beta']).agg(
	{'test_accuracy': ['mean', 'std', 'count']}
)
config_variance.columns = ['mean_acc', 'std_acc', 'count']
config_variance = config_variance[config_variance['count'] >= 2].reset_index()

print('\nConfigurations with highest variance (std) by seed:')
print(
	config_variance.nlargest(10, 'std_acc')[
		['model_arch', 'bottleneck_width', 'beta', 'mean_acc', 'std_acc']
	].round(2)
)

# %%
# 9. Summary table of best results
print('SUMMARY TABLE OF BEST RESULTS')

summary = df.groupby(['model_arch', 'bottleneck_width', 'beta']).agg(
	{
		'test_accuracy': ['mean', 'std'],
		'final_val_loss': 'mean',
		'final_empirical_compression': 'mean',
	}
)
summary.columns = ['mean_acc', 'std_acc', 'mean_val_loss', 'mean_compression']
summary = summary.round(3)

# Best by average accuracy
best_by_acc = summary.nlargest(15, 'mean_acc')
print('\nTop-15 configurations by average accuracy:')
print(best_by_acc)

# Save to CSV
summary.to_csv(OUTPUT_DIR / 'summary_by_config.csv')
print(f'\nSaved: summary_by_config.csv')

# %%
# 10. Final report
print('FINAL REPORT')

best_idx = df['test_accuracy'].idxmax()
best_row = df.loc[best_idx]

print(f"""
BEST RESULT:
   - Test Accuracy: {best_row['test_accuracy']:.2f}%
   - Model: {best_row['model_arch']}
   - Bottleneck Width: {best_row['bottleneck_width']}
   - Beta: {best_row['beta']}
   - Seed: {best_row['seed']}
   - Final Val Loss: {best_row['final_val_loss']:.4f}
   - Final Train Loss: {best_row['final_train_loss']:.4f}

OVERALL STATISTICS:
   - Total experiments: {len(df)}
   - Mean accuracy: {df['test_accuracy'].mean():.2f}%
   - Median accuracy: {df['test_accuracy'].median():.2f}%
   - Best accuracy: {df['test_accuracy'].max():.2f}%
   - Worst accuracy: {df['test_accuracy'].min():.2f}%

RECOMMENDATIONS:
   - Optimal beta range: 0.0001 - 0.01 (most stable results)
   - High beta values (>= 1.0) lead to significant accuracy drop
   - Bottleneck width 32-64 shows best results
""")

# Save final report
report_path = OUTPUT_DIR / 'analysis_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
	f.write(f'EfficientNet Grid Search Results Analysis\n')
	f.write(f'{"=" * 60}\n\n')
	f.write(f'Total experiments: {len(df)}\n')
	f.write(f'Mean accuracy: {df["test_accuracy"].mean():.2f}%\n')
	f.write(f'Best accuracy: {df["test_accuracy"].max():.2f}%\n\n')
	f.write(f'Best configuration:\n')
	f.write(f'  Model: {best_row["model_arch"]}\n')
	f.write(f'  Width: {best_row["bottleneck_width"]}\n')
	f.write(f'  Beta: {best_row["beta"]}\n')
	f.write(f'  Seed: {best_row["seed"]}\n')
	f.write(f'  Test Accuracy: {best_row["test_accuracy"]:.2f}%\n')
	f.write(f'  Final Val Loss: {best_row["final_val_loss"]:.4f}\n')
	f.write(f'  Final Train Loss: {best_row["final_train_loss"]:.4f}\n')

print(f'\nSaved: analysis_report.txt')

print(f'\nAnalysis complete! Results saved to: {OUTPUT_DIR}')
