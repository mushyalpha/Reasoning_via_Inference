"""
Decomposition of grasp outcome into a geometric gate and a post-gate
execution channel, for the v2 floating-gripper dataset.

Every trial that produces at least one CGN candidate is either rejected at
the open-hand collision check (the gate) or executed. Marginal success
therefore factorises as

    P(Y=1) = P(has_grasps) * P(gate_pass | has_grasps) * P(Y=1 | gate_pass)

Plotting the two right-hand factors separately separates variables that act
by making poses geometrically invalid from variables that act by making
valid poses physically bad. Depth noise moves the two factors in opposite
directions, which is why its marginal effect is non-monotone.

Visual style follows plot_cylinder_v2_summary.py (supervisor feedback, 7 Aug):
sigma_d treated as a categorical condition with equal tick spacing, one
accent colour, claim-bearing panel titles.

Input:  results/experiment_results_v2.csv
Output: results/figures/gate_decomposition.{png,pdf}
        results/figures/failure_modes_v2.{png,pdf}
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/experiment_results_v2.csv"
OUT = "results/figures/gate_decomposition"
OUT_MODES = "results/figures/failure_modes_v2"

ACCENT = "#0B7A75"      # dark teal -- post-gate execution channel
CHARCOAL = "#2B2B2B"    # marginal success (the observable outcome)
RUST = "#A8432B"        # geometric gate channel

# Grayscale ramp for failure modes, one accent reserved for success, so the
# eye reads "success vs everything that went wrong" first.
MODE_ORDER = ["success", "pregrasp_collision", "executed_dropped",
              "executed_ejected", "no_grasps"]
MODE_LABEL = {
    "success": "Success",
    "pregrasp_collision": "Pre-grasp collision (too close)",
    "executed_dropped": "Closed on nothing / dropped (too far)",
    "executed_ejected": "Object ejected",
    "no_grasps": "No CGN proposal",
}
MODE_COLOR = {
    "success": ACCENT,
    "pregrasp_collision": "#3D3D3D",
    "executed_dropped": "#8C8C8C",
    "executed_ejected": "#BFBFBF",
    "no_grasps": "#EDEDED",
}


def decompose(df, key):
    """Gate yield, post-gate success, and marginal success at each level."""
    rows = []
    for level, g in df.groupby(key):
        has_grasps = g[g["collision_free"].notna()]
        gate_pass = has_grasps[has_grasps["collision_free"] == 1]
        rows.append({
            key: level,
            "n": len(g),
            "has_grasps": 100 * len(has_grasps) / len(g),
            "gate_pass": 100 * len(gate_pass) / len(has_grasps),
            "post_gate": 100 * gate_pass["success"].mean(),
            "marginal": 100 * g["success"].mean(),
            "n_gate_pass": len(gate_pass),
        })
    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def draw(ax, tab, key, xlabel, title):
    x = range(len(tab))
    ax.plot(x, tab["gate_pass"], "o-", color=RUST, lw=2, ms=7,
            label="Pose clears open-hand collision check")
    ax.plot(x, tab["post_gate"], "s-", color=ACCENT, lw=2, ms=7,
            label="Lift succeeds | pose cleared check")
    ax.plot(x, tab["marginal"], "^--", color=CHARCOAL, lw=2, ms=7,
            label="Marginal success (observed outcome)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v:g}" for v in tab[key]])
    ax.set_xlabel(xlabel)
    ax.set_ylim(-4, 100)
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_failure_modes(df):
    """Composition of outcomes vs noise, one panel per object."""
    objects = sorted(df["object"].unique())
    fig, axes = plt.subplots(1, len(objects), figsize=(13.5, 4.2),
                             sharey=True)

    for ax, obj in zip(axes, objects):
        sub = df[df["object"] == obj]
        levels = sorted(sub["sigma_d"].unique())
        comp = (sub.groupby(["sigma_d", "failure_mode"]).size()
                .unstack(fill_value=0))
        comp = comp.reindex(columns=MODE_ORDER, fill_value=0)
        comp = 100 * comp.div(comp.sum(axis=1), axis=0)

        bottom = pd.Series(0.0, index=comp.index)
        x = range(len(levels))
        for mode in MODE_ORDER:
            ax.bar(x, comp[mode], bottom=bottom, width=0.72,
                   color=MODE_COLOR[mode], edgecolor="white", lw=0.6,
                   label=MODE_LABEL[mode])
            bottom += comp[mode]

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{v:g}" for v in levels], fontsize=8.5)
        ax.set_xlabel(r"$\sigma_d$ (m)")
        ax.set_title(obj, fontsize=11)
        ax.set_ylim(0, 100)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("Share of trials (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Depth noise converts one failure mode into its opposite: poses stop "
        "colliding with the object and start missing it entirely",
        fontsize=12.5, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_MODES}.{ext}", dpi=200, bbox_inches="tight")


def main():
    df = pd.read_csv(CSV)

    sig = decompose(df, "sigma_d")
    phi = decompose(df, "phi")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    draw(axes[0], sig, "sigma_d",
         r"Depth noise $\sigma_d$ (m, discrete tested levels)",
         "(a) Depth noise moves the two channels in opposite directions,\n"
         "so its marginal effect peaks rather than decaying")
    draw(axes[1], phi, "phi",
         r"Camera elevation $\phi$ (deg)",
         "(b) Elevation moves neither channel:\n"
         "its apparent effect is not a perception effect")

    axes[0].set_ylabel("Rate (%)")

    peak = sig.loc[sig["marginal"].idxmax()]
    ix = int(sig.index[sig["sigma_d"] == peak["sigma_d"]][0])
    axes[0].annotate(
        f"peak {peak['marginal']:.1f}%\nat $\\sigma_d$={peak['sigma_d']:g}",
        xy=(ix, peak["marginal"]), xytext=(ix + 0.35, peak["marginal"] + 22),
        fontsize=9, color=CHARCOAL,
        arrowprops=dict(arrowstyle="-", color=CHARCOAL, lw=0.9))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "Grasp failure decomposes into a geometric gate and an execution "
        "channel  ($n$ = {:,} trials, floating gripper, top-1 pose)".format(len(df)),
        fontsize=12.5, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")

    plot_failure_modes(df)

    fmt = lambda v: f"{v:.1f}"
    pd.set_option("display.width", 120)
    print("\nBy sigma_d:\n", sig.to_string(index=False, float_format=fmt))
    print("\nBy phi:\n", phi.to_string(index=False, float_format=fmt))
    print(f"\nWrote {OUT}.png/.pdf and {OUT_MODES}.png/.pdf")


if __name__ == "__main__":
    main()
