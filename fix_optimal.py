content = open('notebooks/efficientnet_numerical_analisys.py', 'r', encoding='utf-8').read()

# Remove the LogNorm import line
content = content.replace('from matplotlib.colors import LogNorm\n', '')

# Replace the entire optimal region section
old = """# Visualize optimal region with heatmap background
fig, ax = plt.subplots(figsize=(10, 8))

# Create heatmap background showing accuracy density

# All points as heatmap
heatmap = ax.scatter(
	df['log_beta'],
	df['bottleneck_width'],
	c=df['test_accuracy'],

	cmap='YlOrRd',

	alpha=0.6,
)

# Top configurations
ax.scatter(
	np.log10(top_configs['beta']),
	top_configs['bottleneck_width'],
	c='gold',
	s=100,
	edgecolors='black',
	linewidth=1.5,
	label=f'Top {top_n}',
	zorder=5,
)"""

new = """# Visualize optimal region
fig, ax = plt.subplots(figsize=(10, 8))

# Filter out -inf log_beta values
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
)"""

if old in content:
	content = content.replace(old, new)
	print('Replaced optimal_region section')
else:
	print('WARNING: Could not find optimal_region section to replace')

# Also fix colorbar
content = content.replace('plt.colorbar(heatmap,', 'plt.colorbar(ax.collections[0],')

open('notebooks/efficientnet_numerical_analisys.py', 'w', encoding='utf-8').write(content)
print('Done')
