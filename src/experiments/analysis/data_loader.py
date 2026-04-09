"""Data loading utilities for experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_grid_search_results(path: str | Path) -> pd.DataFrame:
	"""Load grid search results from a JSON file into a DataFrame."""
	path = Path(path)
	with open(path, 'r', encoding='utf-8') as f:
		raw_data = json.load(f)
	return pd.DataFrame(raw_data)


def load_with_baseline(
	results_path: str | Path,
	baseline_path: str | Path | None = None,
	model_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
	"""Load main results and optionally merge with baseline data.

	Parameters
	----------
	results_path:
	    Path to the main grid search results JSON.
	baseline_path:
	    Path to the baseline results JSON. If ``None``, no baseline is merged.
	model_filter:
	    If provided, only baseline rows whose ``model_arch`` is in this list
	    are kept.

	Returns
	-------
	pd.DataFrame
	    Combined DataFrame with baseline appended (if requested).
	"""
	df = load_grid_search_results(results_path)
	print(f'Loaded {len(df)} experiments')

	if baseline_path is None:
		return df

	df_baseline = load_grid_search_results(baseline_path)
	print(f'Loaded {len(df_baseline)} baseline experiments (beta=0)')

	if model_filter is not None:
		df_baseline = df_baseline[df_baseline['model_arch'].isin(model_filter)]
		print(f'Filtered to {len(df_baseline)} baseline experiments for {model_filter}')

	df = pd.concat([df, df_baseline], ignore_index=True)
	print(f'Total experiments (including baseline): {len(df)}')
	return df


def prepare_regression_data(
	df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Prepare features and targets for regression, handling beta=0.

	Returns
	-------
	df_regression : DataFrame with ``log_beta`` column (NaN for beta=0).
	X : Feature matrix (excludes NaN ``log_beta`` rows).
	y_accuracy : Target series for accuracy.
	y_compression : Target series for compression.
	"""
	df = df.copy()
	df['log_beta'] = _safe_log10(df['beta'])
	valid_mask = df['log_beta'].notna()
	df_regression = df[valid_mask].copy()
	X = df_regression[['log_beta', 'bottleneck_width']].copy()
	y_accuracy = df_regression['test_accuracy']
	y_compression = df_regression['final_empirical_compression']
	return df_regression, X, y_accuracy, y_compression


def _safe_log10(series: pd.Series) -> pd.Series:
	"""Compute log10, replacing 0 with NaN."""
	result = series.astype(float).copy()
	result = result.replace(0, np.nan)
	return result.apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)
