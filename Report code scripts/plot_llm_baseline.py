"""
plot_llm_baseline.py
====================
Generates thesis-quality figures from llm_baseline_summary.json.
Run at any point — works with partial data (e.g. T1 only).
Saves to results/figures/llm_baseline_*.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent
SUMMARY     = ROOT / "results" / "llm_baseline_summary.json"
FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style — matches existing SCM figures in results/figures/
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
})

MAJORITY_BASELINE = 0.572   # always-none baseline on full 292-trial set

# Colours
COL_SIGMA = "#2176AE"
COL_PHI   = "#F7931E"
COL_THETA = "#57B349"
COL_RHO   = "#9B59B6"
COL_GREY  = "#BBBBBB"
COL_RED   = "#E74C3C"

CAUSE_COLORS = {
    "sigma_d": COL_SIGMA,
    "phi":     COL_PHI,
    "theta":   COL_THETA,
    "rho":     COL_RHO,
}
CAUSE_LABELS = {
    "sigma_d": r"$\sigma_d$ (depth noise)",
    "phi":     r"$\phi$ (elevation)",
    "theta":   r"$\theta$ (azimuth)",
    "rho":     r"$\rho$ (sparsity, n=1)",
}

TIER_LABELS = {"T1": "T1\n(variable names only)",
               "T2": "T2\n(+camera config)",
               "T3": "T3\n(+perception metrics)"}

# ---------------------------------------------------------------------------
# Load summary
# ---------------------------------------------------------------------------
with open(SUMMARY) as f:
    summary = json.load(f)

available_tiers = sorted(summary.keys())
print(f"Tiers in summary: {available_tiers}")


# ---------------------------------------------------------------------------
# Figure 1 — Primary accuracy per variable × tier
# ---------------------------------------------------------------------------
def fig_primary_accuracy():
    """
    Grouped bar chart: one group per tier, bars coloured by variable.
    Only sigma_d / phi / theta shown (rho n=1 shown as hatched).
    """
    causes_to_plot = ["sigma_d", "phi", "theta"]  # rho n=1 — shown separately
    n_causes = len(causes_to_plot)
    n_tiers  = len(available_tiers)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    bar_width = 0.22
    group_gap = 0.1
    tier_spacing = n_causes * bar_width + group_gap

    tier_xticks = []
    tier_xlabels = []

    for t_idx, tier in enumerate(available_tiers):
        tier_data = summary[tier]["primary_per_cause"]
        group_x = t_idx * tier_spacing

        for c_idx, cause in enumerate(causes_to_plot):
            x = group_x + c_idx * bar_width
            acc = tier_data.get(cause, {}).get("accuracy")
            n   = tier_data.get(cause, {}).get("n", 0)

            if acc is None:
                # Partial / not yet run — hatched placeholder
                ax.bar(x, 0.05, width=bar_width * 0.9,
                       color=COL_GREY, alpha=0.4,
                       hatch="//", edgecolor="white")
                ax.text(x, 0.06, "pending", ha="center", va="bottom",
                        fontsize=7, color="grey", rotation=45)
            else:
                ax.bar(x, acc, width=bar_width * 0.9,
                       color=CAUSE_COLORS[cause], alpha=0.85,
                       edgecolor="white", linewidth=0.8)
                ax.text(x, acc + 0.012, f"{acc:.0%}",
                        ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color=CAUSE_COLORS[cause])

        group_centre = group_x + (n_causes - 1) * bar_width / 2
        tier_xticks.append(group_centre)
        tier_xlabels.append(TIER_LABELS[tier])

    # Mean primary accuracy dots
    for t_idx, tier in enumerate(available_tiers):
        mean_acc = summary[tier].get("primary_accuracy")
        if mean_acc is not None:
            group_x = t_idx * tier_spacing
            group_centre = group_x + (n_causes - 1) * bar_width / 2
            ax.plot(group_centre, mean_acc, marker="D", markersize=7,
                    color="black", zorder=5, label="Mean (primary)" if t_idx == 0 else "")
            ax.text(group_centre + 0.08, mean_acc + 0.005,
                    f"mean={mean_acc:.0%}", fontsize=8, va="bottom", color="black")

    ax.set_xticks(tier_xticks)
    ax.set_xticklabels(tier_xlabels)
    ax.set_ylabel("Attribution accuracy (primary scope)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("LLM Baseline — Primary Attribution Accuracy\n"
                 "(95 single-variable-fixable trials; SCM comparison scope)",
                 pad=10)

    # Legend for variable colours
    legend_patches = [
        mpatches.Patch(color=CAUSE_COLORS[c], alpha=0.85, label=CAUSE_LABELS[c])
        for c in causes_to_plot
    ]
    legend_patches.append(
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                   markersize=7, label="Mean accuracy")
    )
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.85)

    fig.tight_layout()
    out = FIGURES_DIR / "llm_baseline_primary_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Full-set accuracy vs majority-class baseline across tiers
# ---------------------------------------------------------------------------
def fig_fullset_accuracy():
    """
    Bar chart: LLM full-set accuracy per tier vs majority-class baseline.
    Shows the secondary evaluation scope (all 292 trials).
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))

    xs = np.arange(len(available_tiers))
    accs = [summary[t].get("full_accuracy") or 0.0 for t in available_tiers]
    labels = [TIER_LABELS[t] for t in available_tiers]

    bars = ax.bar(xs, accs, width=0.45, color=COL_SIGMA, alpha=0.82,
                  edgecolor="white", linewidth=0.8, label="LLM accuracy")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.008,
                f"{acc:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COL_SIGMA)

    # Majority-class baseline
    ax.axhline(MAJORITY_BASELINE, color=COL_RED, linewidth=1.6,
               linestyle="--", label=f"Majority-class baseline ({MAJORITY_BASELINE:.1%})")
    ax.text(len(available_tiers) - 0.52, MAJORITY_BASELINE + 0.012,
            f"Majority class\n({MAJORITY_BASELINE:.1%})", fontsize=8.5,
            color=COL_RED, va="bottom", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Attribution accuracy (full set, n=292)")
    ax.set_ylim(0, 0.80)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_title("LLM Baseline — Full-Set Accuracy vs Majority-Class Baseline\n"
                 "(all 292 failed trials; secondary metric)",
                 pad=10)
    ax.legend(loc="upper left", framealpha=0.85)

    fig.tight_layout()
    out = FIGURES_DIR / "llm_baseline_fullset_accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Consistency (agreement rate) per tier
# ---------------------------------------------------------------------------
def fig_consistency():
    """
    Horizontal bar chart of mean agreement rate per tier.
    Annotates what 1.0 (SCM determinism) looks like vs LLM stochasticity.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    ys = np.arange(len(available_tiers))
    rates = [summary[t].get("mean_agreement_rate") or 0.0 for t in available_tiers]
    labels = [TIER_LABELS[t] for t in available_tiers]

    ax.barh(ys, rates, height=0.4, color=COL_PHI, alpha=0.82,
            edgecolor="white", linewidth=0.8)

    for y, rate in zip(ys, rates):
        ax.text(rate + 0.005, y, f"{rate:.3f}",
                va="center", fontsize=9, fontweight="bold", color=COL_PHI)

    # SCM reference
    ax.axvline(1.0, color="black", linewidth=1.4, linestyle="-",
               label="SCM (deterministic = 1.000)")
    ax.text(0.995, len(available_tiers) - 0.55, "SCM\n(1.000)",
            fontsize=8, color="black", ha="right", va="top")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean agreement rate (3 stochastic calls)")
    ax.set_xlim(0, 1.10)
    ax.set_title("LLM Baseline — Attribution Consistency\n"
                 "(agreement rate across 3 queries at T=1.0)",
                 pad=10)
    ax.legend(loc="lower right", framealpha=0.85)

    fig.tight_layout()
    out = FIGURES_DIR / "llm_baseline_consistency.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_primary_accuracy()
    fig_fullset_accuracy()
    fig_consistency()
    print("\nAll figures saved to results/figures/")
    print("Re-run after T2/T3 complete — pending bars will fill in automatically.")
