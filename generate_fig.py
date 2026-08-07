import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16
})

df = pd.read_csv('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/experiment_results_v2_cylinder.csv')

# Clean failure modes as requested
df['failure_mode'] = df['failure_mode'].replace({
    'executed_dropped': 'dropped',
    'executed_ejected': 'ejected'
})

fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# Panel (a): Success Rate vs sigma_d
ax = axs[0, 0]
succ_mean = df.groupby('sigma_d')['success'].mean()
succ_sem = df.groupby('sigma_d')['success'].sem() * 1.96 # 95% CI approx
ax.plot(succ_mean.index, succ_mean.values, marker='o', color='#2b8cbe', label='Mean Success')
ax.fill_between(succ_mean.index, succ_mean.values - succ_sem.values, succ_mean.values + succ_sem.values, color='#2b8cbe', alpha=0.3, label='95% CI')
ax.set_xlabel(r'$\sigma_d$')
ax.set_ylabel('Success Rate')
ax.legend()
ax.text(-0.1, 1.1, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel (b): Mean n_grasps vs sigma_d, split by rho
ax = axs[0, 1]
colors = sns.color_palette("muted", n_colors=df['rho'].nunique())
for i, rho in enumerate(sorted(df['rho'].unique())):
    dfr = df[df['rho'] == rho]
    n_mean = dfr.groupby('sigma_d')['n_grasps'].mean()
    ax.plot(n_mean.index, n_mean.values, marker='s', color=colors[i], label=rf'$\rho = {rho}$')
ax.set_xlabel(r'$\sigma_d$')
ax.set_ylabel(r'Mean $n_{grasps}$')
ax.legend()
ax.text(-0.1, 1.1, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel (c): Success rate heatmap
ax = axs[1, 0]
heatmap_data = df.pivot_table(values='success', index='sigma_d', columns='rho', aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Success Rate'})
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\sigma_d$')
ax.text(-0.1, 1.1, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel (d): Failure mode proportions vs sigma_d
ax = axs[1, 1]
modes = ['success', 'pregrasp_collision', 'dropped', 'ejected', 'no_grasps']
counts = df.groupby(['sigma_d', 'failure_mode']).size().unstack(fill_value=0)
for mode in modes:
    if mode not in counts.columns:
        counts[mode] = 0
counts = counts[modes]
props = counts.div(counts.sum(axis=1), axis=0)

bottom = np.zeros(len(props))
colors_fail = sns.color_palette("muted", n_colors=len(modes))
sigma_vals = props.index.values
bar_width = (sigma_vals.max() - sigma_vals.min()) / len(sigma_vals) * 0.8 if len(sigma_vals) > 1 else 0.04

for i, mode in enumerate(modes):
    ax.bar(props.index, props[mode], bottom=bottom, width=bar_width, label=mode.replace('_', ' ').title(), color=colors_fail[i], edgecolor='white', alpha=0.9)
    bottom += props[mode]

ax.set_xlabel(r'$\sigma_d$')
ax.set_ylabel('Proportion')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.text(-0.1, 1.1, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

os.makedirs('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures', exist_ok=True)
plt.savefig('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures/fig_summary_panel.pdf')
plt.savefig('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures/fig_summary_panel.png', dpi=300)
