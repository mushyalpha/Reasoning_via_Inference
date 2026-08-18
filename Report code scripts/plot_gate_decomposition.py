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
        results/failure_mode_balance.csv
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/experiment_results_v2.csv"
OUT = "results/figures/gate_decomposition"
OUT_MODES = "results/figures/failure_modes_v2"
OUT_BALANCE = "results/failure_mode_balance.csv"

ACCENT = "#0B7A75"      # dark teal -- post-gate execution channel
CHARCOAL = "#2B2B2B"    # marginal success (the observable outcome)
RUST = "#A8432B"        # geometric gate channel

EXEC_MODES = ("executed_dropped", "executed_ejected")
OBJECT_STYLE = {
    "cylinder": dict(label="Cylinder", color=CHARCOAL, marker="o"),
    "box": dict(label="Sugar box", color=RUST, marker="s"),
    "mustard": dict(label="Mustard bottle", color=ACCENT, marker="^"),
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


def _style(ax):
    ax.grid(axis="y", color="#E6E6E6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_channels(ax, tab, key, xlabel, title):
    """Gate and post-gate only; y-limit follows the data in this panel."""
    x = range(len(tab))
    ax.plot(x, tab["gate_pass"], "o-", color=RUST, lw=2, ms=7,
            label="Pose clears open-hand collision check")
    ax.plot(x, tab["post_gate"], "s-", color=ACCENT, lw=2, ms=7,
            label="Lift succeeds | pose cleared check")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v:g}" for v in tab[key]])
    ax.set_xlabel(xlabel)
    ymax = max(tab["gate_pass"].max(), tab["post_gate"].max())
    ax.set_ylim(0, ymax * 1.12)
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    _style(ax)


def draw_marginal(ax, tab, key, xlabel, title):
    """Observed P(Y=1). Callers share a zoomed y-axis across the three factors."""
    x = range(len(tab))
    ax.plot(x, tab["marginal"], "^--", color=CHARCOAL, lw=2, ms=7,
            label="Marginal success $P(Y{=}1)$")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v:g}" for v in tab[key]])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, loc="left", pad=8)
    _style(ax)


def failure_balance(df):
    """B = P(execution failure) - P(pre-grasp collision), from trial counts.

    Execution failure is the union of executed_dropped and executed_ejected:
    the pose cleared the open-hand gate but did not hold. Shares are of all
    trials at that (object, sigma_d) cell. Standard errors use the
    multinomial variance of a difference of two category proportions.
    """
    rows = []
    for (obj, sd), g in df.groupby(["object", "sigma_d"]):
        n = len(g)
        p_coll = (g["failure_mode"] == "pregrasp_collision").mean()
        p_exec = g["failure_mode"].isin(EXEC_MODES).mean()
        B = p_exec - p_coll
        se = np.sqrt((p_exec + p_coll - B ** 2) / n)
        rows.append({
            "object": obj,
            "sigma_d": sd,
            "n": n,
            "p_collision": p_coll,
            "p_execution": p_exec,
            "B": B,
            "se": se,
        })
    return pd.DataFrame(rows).sort_values(["object", "sigma_d"]).reset_index(drop=True)


def plot_failure_modes(df):
    """Signed failure-mode balance against depth noise, by object."""
    tab = failure_balance(df)
    tab.to_csv(OUT_BALANCE, index=False)
    levels = sorted(tab["sigma_d"].unique())
    xmap = {v: i for i, v in enumerate(levels)}

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0, color="black", lw=0.9, zorder=2)
    ax.axhspan(-105, 0, facecolor="#F2F2F2", zorder=0)
    ax.text(
        0.02, 0.08, r"$B<0$: too close (collision)",
        transform=ax.transAxes, fontsize=8.5, color="#555555", va="bottom",
    )
    ax.text(
        0.02, 0.92, r"$B>0$: too far (execution)",
        transform=ax.transAxes, fontsize=8.5, color="#555555", va="top",
    )

    for obj, style in OBJECT_STYLE.items():
        sub = tab[tab["object"] == obj]
        x = [xmap[v] for v in sub["sigma_d"]]
        y = 100 * sub["B"].to_numpy()
        yerr = 100 * 1.96 * sub["se"].to_numpy()
        ax.errorbar(
            x, y, yerr=yerr, fmt=style["marker"] + "-",
            color=style["color"], lw=1.6, ms=7, capsize=2.5,
            elinewidth=0.9, label=style["label"], zorder=3,
        )

    ax.set_xticks(list(range(len(levels))))
    ax.set_xticklabels([f"{v:g}" for v in levels])
    ax.set_xlabel(r"Depth noise $\sigma_d$ (m, discrete tested levels)")
    ax.set_ylabel(
        r"$B(\sigma_d)=P(\mathrm{execution\ failure})-P(\mathrm{pre\text{-}grasp\ collision})$"
        "\n(percentage points)"
    )
    ax.set_ylim(-105, 70)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title(
        "Depth noise replaces collision-dominated failure with execution-dominated failure",
        fontsize=11, loc="left", pad=8,
    )
    _style(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_MODES}.{ext}", dpi=200, bbox_inches="tight")


def main():
    df = pd.read_csv(CSV)

    sig = decompose(df, "sigma_d")
    phi = decompose(df, "phi")
    th = decompose(df, "theta")

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.4),
                             gridspec_kw={"height_ratios": [1.15, 0.85]})
    top, bot = axes[0], axes[1]

    draw_channels(top[0], sig, "sigma_d",
                  r"Depth noise $\sigma_d$ (m)",
                  "(a) Depth noise moves the two channels\n"
                  "in opposite directions")
    draw_channels(top[1], phi, "phi",
                  r"Camera elevation $\phi$ (deg)",
                  "(b) Elevation moves neither channel")
    draw_channels(top[2], th, "theta",
                  r"Camera azimuth $\theta$ (deg)",
                  "(c) Azimuth acts on execution only\n"
                  "(pooled; see object split)")
    top[0].set_ylabel("Rate (%)")

    # Shared zoomed y-axis: the three factors do not share an x-axis
    # (different units), but P(Y=1) is the one quantity they have in common.
    draw_marginal(bot[0], sig, "sigma_d",
                  r"Depth noise $\sigma_d$ (m)",
                  r"(d) Marginal $P(Y{=}1)$ against $\sigma_d$")
    draw_marginal(bot[1], phi, "phi",
                  r"Camera elevation $\phi$ (deg)",
                  r"(e) against $\phi$")
    draw_marginal(bot[2], th, "theta",
                  r"Camera azimuth $\theta$ (deg)",
                  r"(f) against $\theta$")
    for ax in bot:
        ax.set_ylim(0, 12.5)
    bot[0].set_ylabel("Marginal success (%)")
    bot[1].tick_params(labelleft=False)
    bot[2].tick_params(labelleft=False)

    peak = sig.loc[sig["marginal"].idxmax()]
    ix = int(sig.index[sig["sigma_d"] == peak["sigma_d"]][0])
    bot[0].annotate(
        f"peak {peak['marginal']:.1f}% at $\\sigma_d$={peak['sigma_d']:g}",
        xy=(ix, peak["marginal"]), xytext=(ix + 0.55, peak["marginal"] + 1.6),
        fontsize=9, color=CHARCOAL,
        arrowprops=dict(arrowstyle="-", color=CHARCOAL, lw=0.9),
    )

    h_top, l_top = top[0].get_legend_handles_labels()
    h_bot, l_bot = bot[0].get_legend_handles_labels()
    fig.legend(h_top + h_bot, l_top + l_bot, loc="lower center", ncol=3,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Grasp failure decomposes into a geometric gate and an execution "
        "channel  ($n$ = {:,} trials, floating gripper, top-1 pose)".format(len(df)),
        fontsize=12.5, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")

    plot_failure_modes(df)

    fmt = lambda v: f"{v:.1f}"
    pd.set_option("display.width", 120)
    print("\nBy sigma_d:\n", sig.to_string(index=False, float_format=fmt))
    print("\nBy phi:\n", phi.to_string(index=False, float_format=fmt))
    print("\nBy theta:\n", th.to_string(index=False, float_format=fmt))
    bal = pd.read_csv(OUT_BALANCE)
    print("\nFailure-mode balance B (pp):\n",
          (100 * bal.pivot(index="sigma_d", columns="object", values="B"))
          .to_string(float_format=fmt))
    print(f"\nWrote {OUT}.png/.pdf and {OUT_MODES}.png/.pdf")


if __name__ == "__main__":
    main()
