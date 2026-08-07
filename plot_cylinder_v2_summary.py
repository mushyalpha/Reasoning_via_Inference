"""
Thesis/poster-grade summary figure for the cylinder v2 (floating-gripper
shake-test) results.

Redesigned per supervisor-meeting feedback (7 Aug):
  - Success-rate curve is the hero panel (a): charcoal line, single accent
    colour, larger markers, peak annotated, no decoration.
  - Failure-mode composition is a supporting mechanism panel (b):
    grayscale for failure modes + one accent (teal) reserved for success,
    so the eye reads "success vs. everything that went wrong" first.
  - sigma_d is treated as a categorical experimental condition (Option A):
    equally spaced ticks, but explicitly labelled as discrete tested
    levels, not a continuous/uniformly-sampled axis -- avoids the
    misleading-spacing issue (0.02->0.04 is numerically 8x the
    0.0025->0.005 gap, but both render as one tick-step).
  - Titles carry the scientific claim, not the plot mechanics.

Input:  /Users/bonolomasima/Downloads/experiment_results_v2_cylinder.csv
Output: results/figures/cylinder_v2_failure_modes.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/Users/bonolomasima/Downloads/experiment_results_v2_cylinder.csv"
OUT = "results/figures/cylinder_v2_failure_modes.png"

df = pd.read_csv(CSV)

MODE_ORDER = ["success", "pregrasp_collision", "executed_dropped",
              "executed_ejected", "no_grasps", "no_visible_object"]

# Grayscale ramp for "everything that went wrong", one accent for success.
ACCENT = "#0B7A75"          # dark teal -- success only
CHARCOAL = "#2B2B2B"        # hero-line colour, panel (a)
GRAY_RAMP = {
    "pregrasp_collision": "#3D3D3D",   # dark gray  -- biggest failure mode
    "executed_dropped":   "#8C8C8C",   # medium gray
    "executed_ejected":   "#BFBFBF",   # light gray
    "no_grasps":          "#EDEDED",   # near-white
    "no_visible_object":  "#F7F7F7",
}
COLORS = {"success": ACCENT, **GRAY_RAMP}
HATCH = {"no_grasps": "///", "no_visible_object": "///"}

ct = pd.crosstab(df.sigma_d, df.failure_mode, normalize="index")
ct = ct.reindex(columns=[m for m in MODE_ORDER if m in ct.columns], fill_value=0)
sigma_levels = ct.index.tolist()
xt = np.arange(len(sigma_levels))
xlabels = [f"{s:g}" for s in sigma_levels]

succ = df.groupby("sigma_d").success.mean().reindex(sigma_levels)
peak_i = int(np.argmax(succ.values))
peak_sigma, peak_val = sigma_levels[peak_i], succ.values[peak_i]
collapse_i = len(sigma_levels) - 1  # sigma_d = 0.04, the collapsed end

fig, axes = plt.subplots(
    1, 2, figsize=(12.5, 5.4),
    gridspec_kw={"width_ratios": [1.15, 1.0]}
)

# ---------------------------------------------------------------- panel (a)
ax = axes[0]
ax.axvline(peak_i, color="#CCCCCC", linewidth=1.2, zorder=0, linestyle="--")
ax.plot(xt, succ.values, color=CHARCOAL, linewidth=2.2, zorder=2)
ax.scatter(xt, succ.values, s=90, color=ACCENT, edgecolor=CHARCOAL,
           linewidth=1.0, zorder=3)

ax.annotate(
    f"peak: {peak_val:.1%}",
    xy=(peak_i, peak_val), xytext=(peak_i + 0.35, peak_val + 0.035),
    fontsize=11, fontweight="bold", color=ACCENT,
    arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8),
)
ax.annotate(
    "collapse", xy=(collapse_i, succ.values[collapse_i]),
    xytext=(collapse_i - 1.15, succ.values[collapse_i] + 0.045),
    fontsize=10, color="#555555",
    arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8),
)

ax.set_xticks(xt)
ax.set_xticklabels(xlabels)
ax.set_xlabel(r"$\sigma_d$  (depth-noise std, m)  —  discrete tested levels")
ax.set_ylabel("Success rate")
ax.set_ylim(0, max(succ.values) * 1.28)
ax.set_title("(a)  Success rate", loc="left", fontsize=13, fontweight="bold")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#EFEFEF", linewidth=0.8, zorder=0)

# ---------------------------------------------------------------- panel (b)
ax2 = axes[1]
bottom = np.zeros(len(sigma_levels))
for mode in ct.columns:
    vals = ct[mode].values
    ax2.bar(xt, vals, bottom=bottom, width=0.68, label=mode,
             color=COLORS.get(mode, "#999"), hatch=HATCH.get(mode),
             edgecolor="white", linewidth=0.6)
    bottom = bottom + vals

ax2.set_xticks(xt)
ax2.set_xticklabels(xlabels)
ax2.set_xlabel(r"$\sigma_d$  (depth-noise std, m)  —  discrete tested levels")
ax2.set_ylabel("Fraction of trials")
ax2.set_ylim(0, 1.3)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_title("(b)  Failure-mode composition", loc="left", fontsize=13,
              fontweight="bold", pad=32)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)
ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22),
            ncol=3, frameon=False)

ax2.annotate(
    "collision-dominated\n(sigma_d 0\u20130.01)", xy=(1.5, 1.0), xytext=(1.5, 1.16),
    fontsize=8.5, color="#333333", ha="center",
    arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8),
)
ax2.annotate(
    "failure shifts:\ncollision \u2192 execution/no-grasp", xy=(5, 1.0), xytext=(5, 1.16),
    fontsize=8.5, color="#333333", ha="center",
    arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.8),
)

fig.suptitle("Depth noise produces a non-monotonic effect on grasp success",
             fontsize=14, fontweight="bold", y=1.02)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print("Saved", OUT)

print("\nOverall failure-mode breakdown:")
print((df.failure_mode.value_counts(normalize=True) * 100).round(1))
print(f"\nPeak success: {peak_val:.1%} at sigma_d={peak_sigma}")
