"""Statistics and ML utilities for experiment results."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def compute_correlation_matrix(
	df: pd.DataFrame,
	cols: list[str] | None = None,
) -> pd.DataFrame:
	"""Compute correlation matrix for selected columns."""
	if cols is None:
		cols = [
			'test_accuracy',
			'final_empirical_compression',
			'beta',
			'bottleneck_width',
			'final_val_loss',
			'final_effective_capacity_utilization',
		]
	return df[cols].corr()


def linear_regression_analysis(
	X: pd.DataFrame,
	y: pd.Series,
	name: str = 'Target',
	alpha_ridge: float = 1.0,
	cv: int = 5,
) -> dict:
	"""Fit Linear and Ridge regression, print metrics, return results."""
	print(f'\n--- Predicting {name} ---')
	lr = LinearRegression()
	lr.fit(X, y)
	y_pred = lr.predict(X)

	r2 = r2_score(y, y_pred)
	rmse = np.sqrt(mean_squared_error(y, y_pred))
	print(f'R2 Score: {r2:.4f}')
	print(f'RMSE: {rmse:.4f}')
	print('\nCoefficients:')
	for feat, coef in zip(X.columns, lr.coef_):
		print(f'  {feat}: {coef:.4f}')
	print(f'  Intercept: {lr.intercept_:.4f}')

	cv_scores = cross_val_score(lr, X, y, cv=cv, scoring='r2')
	print(f'\nCross-validation R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

	# Ridge
	ridge = Ridge(alpha=alpha_ridge)
	ridge.fit(X, y)
	ridge_r2 = r2_score(y, ridge.predict(X))
	print(f'\nRidge R2: {ridge_r2:.4f}')
	print(
		'Ridge Coefficients: ' + ', '.join(f'{f}={c:.4f}' for f, c in zip(X.columns, ridge.coef_))
	)

	return {
		'lr': lr,
		'ridge': ridge,
		'r2': r2,
		'rmse': rmse,
		'cv_r2_mean': cv_scores.mean(),
		'cv_r2_std': cv_scores.std(),
		'y_pred': y_pred,
		'residuals': y - y_pred,
	}


def random_forest_analysis(
	X: pd.DataFrame,
	y: pd.Series,
	name: str = 'Target',
	cv: int = 5,
	**rf_kwargs,
) -> dict:
	"""Fit Random Forest, print metrics and feature importance."""
	defaults = dict(
		n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
	)
	defaults.update(rf_kwargs)
	rf = RandomForestRegressor(**defaults)
	rf.fit(X, y)
	y_pred = rf.predict(X)

	r2 = r2_score(y, y_pred)
	rmse = np.sqrt(mean_squared_error(y, y_pred))
	print(f'Train R2: {r2:.4f}')
	print(f'RMSE: {rmse:.4f}')

	print('\nFeature Importance (Random Forest):')
	for feat, imp in zip(X.columns, rf.feature_importances_):
		print(f'  {feat}: {imp:.4f}')

	cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
	print(f'\nCross-validation R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

	return {
		'model': rf,
		'r2': r2,
		'rmse': rmse,
		'cv_r2_mean': cv_scores.mean(),
		'cv_r2_std': cv_scores.std(),
		'y_pred': y_pred,
		'feature_importances': dict(zip(X.columns, rf.feature_importances_)),
	}


def gradient_boosting_analysis(
	X: pd.DataFrame,
	y: pd.Series,
	name: str = 'Target',
	cv: int = 5,
	**gb_kwargs,
) -> dict:
	"""Fit Gradient Boosting, print metrics and feature importance."""
	defaults = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
	defaults.update(gb_kwargs)
	gb = GradientBoostingRegressor(**defaults)
	gb.fit(X, y)
	y_pred = gb.predict(X)

	r2 = r2_score(y, y_pred)
	rmse = np.sqrt(mean_squared_error(y, y_pred))
	print(f'Train R2: {r2:.4f}')
	print(f'RMSE: {rmse:.4f}')

	print('\nFeature Importance (Gradient Boosting):')
	for feat, imp in zip(X.columns, gb.feature_importances_):
		print(f'  {feat}: {imp:.4f}')

	cv_scores = cross_val_score(gb, X, y, cv=cv, scoring='r2')
	print(f'\nCross-validation R2: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

	return {
		'model': gb,
		'r2': r2,
		'rmse': rmse,
		'cv_r2_mean': cv_scores.mean(),
		'cv_r2_std': cv_scores.std(),
		'y_pred': y_pred,
		'feature_importances': dict(zip(X.columns, gb.feature_importances_)),
	}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
	"""Add interaction and polynomial features."""
	X = X.copy()
	if 'log_beta' in X.columns and 'bottleneck_width' in X.columns:
		X['beta_x_width'] = X['log_beta'] * X['bottleneck_width']
		X['width_squared'] = X['bottleneck_width'] ** 2
	return X


# =============================================================================
# Bootstrap resampling and statistical significance analysis
# =============================================================================


def bootstrap_ci(
	data: np.ndarray | pd.Series,
	stat_func: Callable = np.mean,
	n_bootstrap: int = 10000,
	alpha: float = 0.05,
	seed: int = 42,
) -> dict:
	"""Compute bootstrap confidence interval for a statistic.

	Parameters
	----------
	data :
	    1D array of observations.
	stat_func :
	    Function to compute the statistic (default: mean).
	n_bootstrap :
	    Number of bootstrap resamples.
	alpha :
	    Significance level (default 0.05 → 95% CI).
	seed :
	    Random seed for reproducibility.

	Returns
	-------
	dict with keys: statistic, ci_lower, ci_upper, std, n
	"""
	rng = np.random.RandomState(seed)
	n = len(data)
	bootstrap_stats = np.empty(n_bootstrap)

	for i in range(n_bootstrap):
		sample = rng.choice(data, size=n, replace=True)
		bootstrap_stats[i] = stat_func(sample)

	observed = stat_func(data)
	ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
	ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

	return {
		'statistic': observed,
		'ci_lower': ci_lower,
		'ci_upper': ci_upper,
		'std': np.std(bootstrap_stats, ddof=1),
		'n': n,
		'bootstrap_values': bootstrap_stats,
	}


def compute_beta_group_statistics(
	df: pd.DataFrame,
	metric_col: str,
	n_bootstrap: int = 10000,
	alpha: float = 0.05,
) -> pd.DataFrame:
	"""Compute bootstrap CIs for each beta group.

	Returns DataFrame with columns: beta, mean, ci_lower, ci_upper, std, n.
	"""
	results = []
	for beta_val, group in df.groupby('beta'):
		ci = bootstrap_ci(group[metric_col].values, n_bootstrap=n_bootstrap, alpha=alpha)
		results.append(
			{
				'beta': beta_val,
				'mean': ci['statistic'],
				'ci_lower': ci['ci_lower'],
				'ci_upper': ci['ci_upper'],
				'std': ci['std'],
				'n': ci['n'],
			}
		)
	return pd.DataFrame(results).sort_values('beta').reset_index(drop=True)


def pairwise_beta_significance(
	df: pd.DataFrame,
	metric_col: str,
	n_bootstrap: int = 10000,
	alpha: float = 0.05,
) -> pd.DataFrame:
	"""Pairwise bootstrap significance testing between beta groups.

	For each pair of beta values, computes the bootstrap distribution of
	the difference in means and determines if the CI excludes zero.

	Returns DataFrame with columns: beta_a, beta_b, diff_mean, diff_ci_lower,
	diff_ci_upper, p_significant, effect_size (Cohen's d).
	"""
	beta_groups = sorted(df['beta'].unique())
	results = []

	for i, beta_a in enumerate(beta_groups):
		for beta_b in beta_groups[i + 1 :]:
			group_a = df[df['beta'] == beta_a][metric_col].values
			group_b = df[df['beta'] == beta_b][metric_col].values

			rng = np.random.RandomState(42)
			n_a, n_b = len(group_a), len(group_b)
			n_iter = n_bootstrap
			diffs = np.empty(n_iter)

			for j in range(n_iter):
				sample_a = rng.choice(group_a, size=n_a, replace=True)
				sample_b = rng.choice(group_b, size=n_b, replace=True)
				diffs[j] = np.mean(sample_a) - np.mean(sample_b)

			diff_mean = np.mean(diffs)
			ci_lower = np.percentile(diffs, 100 * alpha / 2)
			ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

			# CI excludes zero → significant difference
			is_significant = (ci_lower > 0) or (ci_upper < 0)

			# Cohen's d
			pooled_std = np.sqrt(
				(
					(n_a - 1) * np.std(group_a, ddof=1) ** 2
					+ (n_b - 1) * np.std(group_b, ddof=1) ** 2
				)
				/ (n_a + n_b - 2)
			)
			cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0

			results.append(
				{
					'beta_a': beta_a,
					'beta_b': beta_b,
					'diff_mean': diff_mean,
					'diff_ci_lower': ci_lower,
					'diff_ci_upper': ci_upper,
					'p_significant': is_significant,
					'cohens_d': cohens_d,
				}
			)

	return pd.DataFrame(results)


def find_significant_beta_regions(
	df: pd.DataFrame,
	metric_col: str,
	n_bootstrap: int = 10000,
	alpha: float = 0.05,
) -> pd.DataFrame:
	"""Identify contiguous beta regions with statistically significant differences.

	Compares adjacent beta values and identifies where the difference is significant.
	Returns a summary of regions where significant transitions occur.
	"""
	beta_sorted = sorted(df['beta'].unique())
	results = []

	for i in range(len(beta_sorted) - 1):
		beta_a = beta_sorted[i]
		beta_b = beta_sorted[i + 1]

		group_a = df[df['beta'] == beta_a][metric_col].values
		group_b = df[df['beta'] == beta_b][metric_col].values

		rng = np.random.RandomState(42)
		n_a, n_b = len(group_a), len(group_b)
		diffs = np.empty(n_bootstrap)

		for j in range(n_bootstrap):
			sample_a = rng.choice(group_a, size=n_a, replace=True)
			sample_b = rng.choice(group_b, size=n_b, replace=True)
			diffs[j] = np.mean(sample_a) - np.mean(sample_b)

		ci_lower = np.percentile(diffs, 100 * alpha / 2)
		ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))
		is_significant = (ci_lower > 0) or (ci_upper < 0)

		# Effect size
		pooled_std = np.sqrt(
			((n_a - 1) * np.std(group_a, ddof=1) ** 2 + (n_b - 1) * np.std(group_b, ddof=1) ** 2)
			/ (n_a + n_b - 2)
		)
		cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0

		results.append(
			{
				'beta_from': beta_a,
				'beta_to': beta_b,
				'mean_a': np.mean(group_a),
				'mean_b': np.mean(group_b),
				'diff': np.mean(group_a) - np.mean(group_b),
				'diff_ci_lower': ci_lower,
				'diff_ci_upper': ci_upper,
				'significant': is_significant,
				'cohens_d': cohens_d,
				'effect_magnitude': abs(cohens_d),
			}
		)

	return pd.DataFrame(results)


# =============================================================================
# Relationship analysis: beta, bottleneck width, and metrics
# =============================================================================


def analyze_beta_metric_relationship(
	df: pd.DataFrame,
	metric_col: str = 'test_accuracy',
	beta_col: str = 'beta',
	width_col: str = 'bottleneck_width',
) -> dict:
	"""Analyze relationships between beta, bottleneck width, and a metric.

	Returns dict with:
	- beta_correlation: Spearman correlation between log(beta) and metric
	- width_correlation: Spearman correlation between width and metric
	- anova_beta: Kruskal-Wallis test for beta groups
	- anova_width: Kruskal-Wallis test for width groups
	- partial_correlations: Partial correlations controlling for other variables
	- interaction_effect: Two-way ANOVA for beta x width interaction
	"""
	df_analysis = df.copy()
	df_analysis['log_beta'] = np.log10(df_analysis[beta_col].replace(0, np.nan))
	df_analysis = df_analysis.dropna(subset=['log_beta', metric_col])

	# Spearman correlations
	beta_corr, beta_p = stats.spearmanr(df_analysis['log_beta'], df_analysis[metric_col])
	width_corr, width_p = stats.spearmanr(df_analysis[width_col], df_analysis[metric_col])

	# Kruskal-Wallis tests for group differences
	beta_groups = [g[metric_col].values for _, g in df_analysis.groupby(beta_col)]
	width_groups = [g[metric_col].values for _, g in df_analysis.groupby(width_col)]

	kw_beta_stat, kw_beta_p = stats.kruskal(*[g for g in beta_groups if len(g) > 0])
	kw_width_stat, kw_width_p = stats.kruskal(*[g for g in width_groups if len(g) > 0])

	# Partial correlations (controlling for the other variable)
	from scipy.stats import pearsonr

	# Residualize metric on width, then correlate with log_beta
	from sklearn.linear_model import LinearRegression

	lr_width = LinearRegression().fit(df_analysis[[width_col]], df_analysis[metric_col])
	residuals_metric = df_analysis[metric_col] - lr_width.predict(df_analysis[[width_col]])

	lr_beta_width = LinearRegression().fit(df_analysis[[width_col]], df_analysis['log_beta'])
	residuals_beta = df_analysis['log_beta'] - lr_beta_width.predict(df_analysis[[width_col]])

	partial_beta_corr, partial_beta_p = pearsonr(residuals_beta, residuals_metric)

	# Residualize metric on log_beta, then correlate with width
	lr_beta = LinearRegression().fit(df_analysis[['log_beta']], df_analysis[metric_col])
	residuals_metric2 = df_analysis[metric_col] - lr_beta.predict(df_analysis[['log_beta']])

	lr_width_beta = LinearRegression().fit(df_analysis[['log_beta']], df_analysis[width_col])
	residuals_width = df_analysis[width_col] - lr_width_beta.predict(df_analysis[['log_beta']])

	partial_width_corr, partial_width_p = pearsonr(residuals_width, residuals_metric2)

	# Two-way ANOVA for interaction effect (using scipy)
	# Create categorical versions
	df_anova = df_analysis[[beta_col, width_col, metric_col]].copy()
	df_anova['beta_cat'] = df_anova[beta_col].astype('category').cat.codes
	df_anova['width_cat'] = df_anova[width_col].astype('category').cat.codes

	# Use sklearn for two-way ANOVA-like analysis
	from sklearn.linear_model import LinearRegression

	# Main effects model
	X_main = df_anova[['beta_cat', 'width_cat']].values
	y = df_anova[metric_col].values
	lr_main = LinearRegression().fit(X_main, y)
	ss_main = np.sum((lr_main.predict(X_main) - y.mean()) ** 2)

	# Interaction model
	X_inter = np.column_stack(
		[
			df_anova['beta_cat'],
			df_anova['width_cat'],
			df_anova['beta_cat'] * df_anova['width_cat'],
		]
	)
	lr_inter = LinearRegression().fit(X_inter, y)
	ss_inter = np.sum((lr_inter.predict(X_inter) - y.mean()) ** 2)

	# Residuals
	ss_resid = np.sum((y - lr_inter.predict(X_inter)) ** 2)
	ss_total = np.sum((y - y.mean()) ** 2)

	# Degrees of freedom
	n_beta = df_anova['beta_cat'].nunique()
	n_width = df_anova['width_cat'].nunique()
	n = len(df_anova)

	df_beta = n_beta - 1
	df_width = n_width - 1
	df_inter = df_beta * df_width
	df_resid = n - (n_beta * n_width) - 1

	# Mean squares
	ms_beta = ss_main / (df_beta + df_width) if (df_beta + df_width) > 0 else 0
	ms_inter = (ss_inter - ss_main) / df_inter if df_inter > 0 else 0
	ms_resid = ss_resid / df_resid if df_resid > 0 else 1

	# F-statistics
	f_beta = ms_beta / ms_resid if ms_resid > 0 else 0
	f_width = ms_beta / ms_resid if ms_resid > 0 else 0
	f_inter = ms_inter / ms_resid if ms_resid > 0 else 0

	# P-values
	p_beta = stats.f.sf(f_beta, df_beta + df_width, df_resid) if df_resid > 0 else 1
	p_width = stats.f.sf(f_width, df_beta + df_width, df_resid) if df_resid > 0 else 1
	p_inter = stats.f.sf(f_inter, df_inter, df_resid) if df_resid > 0 else 1

	anova_results = {
		'beta': {'F': f_beta, 'p': p_beta, 'df': df_beta + df_width},
		'width': {'F': f_width, 'p': p_width, 'df': df_beta + df_width},
		'beta:width': {'F': f_inter, 'p': p_inter, 'df': df_inter},
	}

	return {
		'beta_spearman_corr': beta_corr,
		'beta_spearman_p': beta_p,
		'width_spearman_corr': width_corr,
		'width_spearman_p': width_p,
		'kw_beta_stat': kw_beta_stat,
		'kw_beta_p': kw_beta_p,
		'kw_width_stat': kw_width_stat,
		'kw_width_p': kw_width_p,
		'partial_beta_corr': partial_beta_corr,
		'partial_beta_p': partial_beta_p,
		'partial_width_corr': partial_width_corr,
		'partial_width_p': partial_width_p,
		'anova_table': anova_results,
		'r_squared': lr_inter.score(X_inter, y),
		'n_samples': len(df_analysis),
	}
