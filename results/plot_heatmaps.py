import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.makedirs('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures', exist_ok=True)

# Matplotlib configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Load data
df = pd.read_csv('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/experiment_results_v2_cylinder.csv')

# --- Figure 1: Mean n_grasps (sigma_d x rho) ---
df_ngrasps = df.groupby(['sigma_d', 'rho'])['n_grasps'].mean().reset_index()
pivot_ngrasps = df_ngrasps.pivot(index='sigma_d', columns='rho', values='n_grasps')

# Ensure correct ordering
sigma_d_levels = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.04]
rho_levels = [0.25, 0.5, 0.75, 1.0]

pivot_ngrasps = pivot_ngrasps.reindex(index=sigma_d_levels, columns=rho_levels)

fig, ax = plt.subplots(figsize=(6, 5))
cmap1 = mpl.cm.YlGnBu
norm1 = mpl.colors.Normalize(vmin=pivot_ngrasps.min().min(), vmax=pivot_ngrasps.max().max())
im1 = ax.imshow(pivot_ngrasps, cmap=cmap1, norm=norm1, aspect='auto')

# Ticks
ax.set_xticks(np.arange(len(rho_levels)))
ax.set_yticks(np.arange(len(sigma_d_levels)))
ax.set_xticklabels(rho_levels)
ax.set_yticklabels(sigma_d_levels)

# Labels
ax.set_ylabel(r'$\sigma_d$')
ax.set_xlabel(r'$\rho$')

# Colorbar
cbar = ax.figure.colorbar(im1, ax=ax)
cbar.ax.set_ylabel('Mean $n_{grasps}$', rotation=-90, va="bottom", labelpad=15)

# Annotations
threshold = pivot_ngrasps.min().min() + (pivot_ngrasps.max().max() - pivot_ngrasps.min().min()) / 2.
for i in range(len(sigma_d_levels)):
    for j in range(len(rho_levels)):
        val = pivot_ngrasps.iloc[i, j]
        if not np.isnan(val):
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color)

fig.tight_layout()
fig.savefig('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures/fig_heatmap_ngrasps.pdf', bbox_inches='tight')
fig.savefig('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures/fig_heatmap_ngrasps.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# --- Figure 2: Success rate (sigma_d x phi) ---
df_success = df.groupby(['sigma_d', 'phi'])['success'].mean().reset_index()
pivot_success = df_success.pivot(index='sigma_d', columns='phi', values='success')

phi_levels = [30, 45, 50, 55, 60, 65]
# Try to convert phi column to match these exactly or map closest
pivot_success.columns = pivot_success.columns.astype(float)
phi_levels_float = [float(x) for x in phi_levels]
pivot_success = pivot_success.reindex(index=sigma_d_levels, columns=phi_levels_float)

fig, ax = plt.subplots(figsize=(6, 5))
cmap2 = mpl.cm.YlOrRd
norm2 = mpl.colors.Normalize(vmin=pivot_success.min().min(), vmax=pivot_success.max().max())
im2 = ax.imshow(pivot_success, cmap=cmap2, norm=norm2, aspect='auto')

ax.set_xticks(np.arange(len(phi_levels)))
ax.set_yticks(np.arange(len(sigma_d_levels)))
ax.set_xticklabels(phi_levels)
ax.set_yticklabels(sigma_d_levels)

ax.set_ylabel(r'$\sigma_d$')
ax.set_xlabel(r'$\phi$ (degrees)')

cbar = ax.figure.colorbar(im2, ax=ax)
cbar.ax.set_ylabel('Success Rate', rotation=-90, va="bottom", labelpad=15)

threshold2 = pivot_success.min().min() + (pivot_success.max().max() - pivot_success.min().min()) / 2.
for i in range(len(sigma_d_levels)):
    for j in range(len(phi_levels)):
        val = pivot_success.iloc[i, j]
        if not np.isnan(val):
            color = "white" if val > threshold2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

fig.tight_layout()
fig.savefig('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures/fig_heatmap_success_phi.pdf', bbox_inches='tight')
fig.savefig('/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/figures/fig_heatmap_success_phi.png', dpi=300, bbox_inches='tight')
plt.close(fig)
