# %%
"""
Robustness Validation Analysis.

Analyzes robustness results grouped by:
- Beta (with baseline at beta=0 marked)
- Architecture

Note: Run with `uv run notebooks/robustness_validation_analysis.py`
"""

import matplotlib
matplotlib.use('Agg')

import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')

# %%
# Load data
PROJECT_ROOT = Path(__file__).parent.parent
ROBUSTNESS_DIR = PROJECT_ROOT / 'results' / 'robustness'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'robustness_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCH_TO_WIDTH = {
    'efficientnet_b0': 16,
    'efficientnet_b1': 32,
    'efficientnet_b2': 48,
    'resnet18': 64,
    'vgg11': 64,
}

results = []
for model_dir in ROBUSTNESS_DIR.iterdir():
    if not model_dir.is_dir():
        continue
    summary_file = model_dir / 'robustness_summary.json'
    if not summary_file.exists():
        continue

    with open(summary_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    name_parts = model_dir.name.split('_')
    if len(name_parts) >= 3:
        arch = '_'.join(name_parts[1:-1])
        param = name_parts[-1]

        data['model_arch'] = arch
        data['bottleneck_width'] = ARCH_TO_WIDTH.get(arch, 64)

        try:
            data['beta'] = float(param)
        except ValueError:
            data['beta'] = float(eval(param.replace('e', 'e')))

        data['is_baseline'] = data['beta'] == 0.0
        data['failed'] = data['clean_accuracy'] < 0.15
        results.append(data)

df = pd.DataFrame(results)
print(f'Loaded {len(df)} results ({df["failed"].sum()} failed)')
architectures = sorted(df['model_arch'].unique())
print(f'Architectures: {architectures}')

# %%
# Metrics to analyze
metrics = [
    ('clean_accuracy', 'Clean Accuracy', 'higher'),
    ('mCE', 'mCE (Corruption Error)', 'lower'),
    ('robust_accuracy', 'Robust Accuracy', 'higher'),
    ('mAA', 'mAA (Adversarial Accuracy)', 'higher'),
]

# All beta values we want to display (including gaps)
ALL_BETAS = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

# Betas that have data (for heatmaps)
BETAS_WITH_DATA = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-2, 1e-1, 1, 10, 100]

# %%
# =============================================================================
# SECTION 1: METRICS vs BETA (overall) - include ALL runs to show failures
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 1: METRICS vs BETA (all, including failed)')
print('=' * 60)

df_valid = df.copy()  # Include ALL runs

for metric, title, direction in metrics:
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline = df_valid[df_valid['is_baseline']]
    ddib = df_valid[~df_valid['is_baseline']]

    # Group by beta and reindex to include ALL_BETAS (including gaps)
    ddib_by_beta = ddib.groupby('beta')[metric].agg(['mean', 'std'])

    # Reindex to include all betas (missing ones become NaN)
    ddib_full = ddib_by_beta.reindex(ALL_BETAS)

    ax.errorbar(
        range(len(ALL_BETAS)),
        ddib_full['mean'].values,
        yerr=ddib_full['std'].values,
        fmt='o-',
        capsize=4,
        color='steelblue',
        label='DDIB (beta>0)',
        markersize=8,
    )

    if len(baseline) > 0:
        base_val = baseline[metric].mean()
        base_std = baseline[metric].std()
        ax.axhline(
            y=base_val,
            color='coral',
            linestyle='--',
            linewidth=2,
            label=f'Baseline (beta=0): {base_val:.3f}',
        )
        ax.fill_between(
            [-0.5, len(ALL_BETAS) - 0.5],
            base_val - base_std,
            base_val + base_std,
            alpha=0.2,
            color='coral',
        )

    ax.set_xlabel('Beta', fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f'{title} vs Beta', fontsize=14)
    ax.set_xticks(range(len(ALL_BETAS)))
    ax.set_xticklabels([f'{b:.0e}' for b in ALL_BETAS], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'metric_vs_beta_{metric}.png'.replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / filename}')

# %%
# =============================================================================
# SECTION 2: METRICS vs BETA by ARCHITECTURE (separate plots)
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 2: METRICS vs BETA by ARCHITECTURE')
print('=' * 60)

for arch in architectures:
    print(f'\n--- {arch} ---')

    for metric, title, direction in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        df_arch = df_valid[df_valid['model_arch'] == arch]

        baseline_arch = df_arch[df_arch['is_baseline']]
        ddib_arch = df_arch[~df_arch['is_baseline']]

        # Reindex to include all betas (missing ones become NaN)
        if len(ddib_arch) > 0:
            ddib_by_beta = ddib_arch.groupby('beta')[metric].agg(['mean', 'std'])
            ddib_full = ddib_by_beta.reindex(ALL_BETAS)

            ax.errorbar(
                range(len(ALL_BETAS)),
                ddib_full['mean'].values,
                yerr=ddib_full['std'].values,
                fmt='o-',
                capsize=4,
                color='steelblue',
                label='DDIB (beta>0)',
                markersize=8,
            )

        if len(baseline_arch) > 0:
            base_val = baseline_arch[metric].mean()
            base_std = baseline_arch[metric].std()
            ax.axhline(
                y=base_val,
                color='coral',
                linestyle='--',
                linewidth=2,
                label=f'Baseline (beta=0): {base_val:.3f}',
            )
            ax.fill_between(
                [-0.5, len(ALL_BETAS) - 0.5],
                base_val - base_std,
                base_val + base_std,
                alpha=0.2,
                color='coral',
            )

        ax.set_xlabel('Beta', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Beta - {arch}', fontsize=14)
        ax.set_xticks(range(len(ALL_BETAS)))
        ax.set_xticklabels([f'{b:.0e}' for b in ALL_BETAS], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_arch = arch.replace('_', '')
        filename = f'metric_vs_beta_{safe_arch}_{metric}.png'.replace('(', '').replace(')', '')
        plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {OUTPUT_DIR / filename}')

# %%
# =============================================================================
# SECTION 3: HEATMAPS (beta vs model architecture)
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 3: HEATMAPS (beta x architecture)')
print('=' * 60)

for metric, title, direction in metrics:
    # Create pivot: beta x architecture
    pivot = df_valid.pivot_table(
        values=metric,
        index='beta',
        columns='model_arch',
        aggfunc='mean'
    )
    # Reindex rows to include only betas with data
    pivot = pivot.reindex(BETAS_WITH_DATA)
    # Sort columns
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f' if 'accuracy' in metric else '.2f',
        cmap='YlOrRd' if direction == 'higher' else 'YlOrRd_r',
        ax=ax,
        cbar_kws={'label': title},
        mask=pivot.isna(),
    )
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Beta')
    ax.set_title(f'{title} by Beta and Architecture')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    filename = f'heatmap_{metric}_beta_arch.png'.replace('(', '').replace(')', '')
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / filename}')

# %%
# =============================================================================
# SECTION 4: CORRUPTION TYPES ANALYSIS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 4: CORRUPTION TYPES ANALYSIS')
print('=' * 60)

# Get CE columns (corruption error for each type)
ce_cols = [col for col in df_valid.columns if col.startswith('CE_')]
corruption_types = [col.replace('CE_', '') for col in ce_cols]
print(f'Corruption types: {corruption_types}')

# Create DataFrame: beta x corruption_type (averaged across architectures)
corruption_data = []
for beta in BETAS_WITH_DATA:
    row = {'beta': beta}
    for col in ce_cols:
        cor_name = col.replace('CE_', '')
        subset = df_valid[df_valid['beta'] == beta]
        if len(subset) > 0:
            row[cor_name] = subset[col].mean()
        else:
            row[cor_name] = np.nan
    corruption_data.append(row)

pivot_corruption = pd.DataFrame(corruption_data).set_index('beta')

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    pivot_corruption,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd_r',
    ax=ax,
    cbar_kws={'label': 'Mean CE'},
)
ax.set_xlabel('Corruption Type')
ax.set_ylabel('Beta')
ax.set_title('Mean Corruption Error by Beta and Corruption Type')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels([f'{b:.0e}' for b in pivot_corruption.index], rotation=0)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'heatmap_corruption_types_beta.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUTPUT_DIR / "heatmap_corruption_types_beta.png"}')

# Individual corruption types by beta (overall)
for col in ce_cols:
    corruption = col.replace('CE_', '')

    pivot = df_valid.pivot_table(
        values=col,
        index='beta',
        columns='model_arch',
        aggfunc='mean'
    )
    pivot = pivot.reindex(BETAS_WITH_DATA)
    pivot = pivot.reindex(columns=sorted(pivot.columns))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd_r',
        ax=ax,
        cbar_kws={'label': 'CE'},
    )
    ax.set_xlabel('Architecture')
    ax.set_ylabel('Beta')
    ax.set_title(f'Corruption Error: {corruption} by Beta and Architecture')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'heatmap_{corruption}_beta_arch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / f"heatmap_{corruption}_beta_arch.png"}')

# %%
# =============================================================================
# SECTION 5: CORRUPTION TYPES vs BETA (line plots)
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 5: CORRUPTION TYPES vs BETA (line plots)')
print('=' * 60)

for col in ce_cols:
    corruption = col.replace('CE_', '')
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline = df_valid[df_valid['is_baseline']]
    ddib = df_valid[~df_valid['is_baseline']]

    # Group by beta
    ddib_by_beta = ddib.groupby('beta')[col].agg(['mean', 'std'])
    ddib_full = ddib_by_beta.reindex(ALL_BETAS)

    ax.errorbar(
        range(len(ALL_BETAS)),
        ddib_full['mean'].values,
        yerr=ddib_full['std'].values,
        fmt='o-',
        capsize=4,
        color='steelblue',
        label='DDIB (beta>0)',
        markersize=8,
    )

    if len(baseline) > 0:
        base_val = baseline[col].mean()
        base_std = baseline[col].std()
        ax.axhline(
            y=base_val,
            color='coral',
            linestyle='--',
            linewidth=2,
            label=f'Baseline (beta=0): {base_val:.3f}',
        )
        ax.fill_between(
            [-0.5, len(ALL_BETAS) - 0.5],
            base_val - base_std,
            base_val + base_std,
            alpha=0.2,
            color='coral',
        )

    ax.set_xlabel('Beta', fontsize=12)
    ax.set_ylabel(f'CE - {corruption}', fontsize=12)
    ax.set_title(f'Corruption Error: {corruption} vs Beta', fontsize=14)
    ax.set_xticks(range(len(ALL_BETAS)))
    ax.set_xticklabels([f'{b:.0e}' for b in ALL_BETAS], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'corruption_{corruption}_vs_beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {OUTPUT_DIR / f"corruption_{corruption}_vs_beta.png"}')

# %%
# =============================================================================
# SECTION 6: SUMMARY STATISTICS
# =============================================================================
print('\n' + '=' * 60)
print('SECTION 3: SUMMARY STATISTICS')
print('=' * 60)

print('\n--- Baseline vs DDIB (all architectures) ---')
summary = df_valid.groupby('is_baseline')[['clean_accuracy', 'mCE', 'robust_accuracy', 'mAA']].agg(['mean', 'std'])
summary.index = ['DDIB (beta>0)', 'Baseline (beta=0)']
print(summary.round(4))

print('\n--- By Architecture: Baseline vs DDIB ---')
for arch in architectures:
    df_arch = df_valid[df_valid['model_arch'] == arch]
    baseline = df_arch[df_arch['is_baseline']]
    ddib = df_arch[~df_arch['is_baseline']]

    if len(baseline) > 0 and len(ddib) > 0:
        print(f'\n{arch}:')
        base_clean = baseline['clean_accuracy'].mean()
        base_mce = baseline['mCE'].mean()
        base_robust = baseline['robust_accuracy'].mean()
        ddib_clean = ddib['clean_accuracy'].mean()
        ddib_mce = ddib['mCE'].mean()
        ddib_robust = ddib['robust_accuracy'].mean()
        print(f"  Baseline: clean={base_clean:.4f}, mCE={base_mce:.4f}, robust={base_robust:.4f}")
        print(f"  DDIB:    clean={ddib_clean:.4f}, mCE={ddib_mce:.4f}, robust={ddib_robust:.4f}")
        print(f"  Delta:   clean={ddib_clean - base_clean:+.4f}, mCE={ddib_mce - base_mce:+.4f}, robust={ddib_robust - base_robust:+.4f}")

# Save data
df_valid.to_csv(OUTPUT_DIR / 'robustness_data.csv', index=False)
print(f'\nSaved: {OUTPUT_DIR / "robustness_data.csv"}')

print('\n' + '=' * 60)
print('ANALYSIS COMPLETE')
print('=' * 60)