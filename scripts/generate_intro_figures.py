#!/usr/bin/env python3
"""Generate Introduction figures adapted from Yang & Bareinboim (2025)
for Sections 1.2 (Diagnostic Gap) and 1.3 (Mechanistic Causal Audit).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Thesis-friendly serif styling
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


def fig_diagnostic_gap():
    """Cartwright Line / PCH mapping of existing methods vs this thesis.

    Matches the Yang & Bareinboim–style Diagnostic Gap diagram: matched points
    on the diagonal, VLM mismatch above the Cartwright line, and L2→L3 gaps.
    Cartwright line starts at the origin. Vertical gap label emphasises that
    the diagnostic problem needs L3 queries and L3 assumptions.
    """
    # Layer centres on a 0–1 square
    L1, L2, L3 = 1 / 6, 0.5, 5 / 6

    fig = plt.figure(figsize=(12.0, 8.0))
    # Main axes leave room for left query sidebar + bottom PCH bar + legend
    AX_LEFT, AX_WIDTH = 0.275, 0.555
    AX_BOTTOM, AX_HEIGHT = 0.285, 0.565
    ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_WIDTH, AX_HEIGHT])

    # Cartwright (Matched Expressiveness) line — starts at origin.
    # Explicit endpoints and zero margins make the first rendered pixel coincide
    # with the axes' (0, 0) origin. Plot background is left white (no zone fills).
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#1b5e20", lw=2.2, zorder=3,
            solid_capstyle="butt", clip_on=True)

    # Zone labels (placed to avoid overlapping point annotations)
    ax.text(
        0.56, 0.88,
        "Invalid Inference\n(query more expressive than model assumptions)",
        ha="center", va="center", fontsize=8, color="#c62828",
        style="italic", zorder=2, linespacing=1.2,
    )
    ax.text(
        1.09, 0.32,
        "Valid Inference\n(query no more expressive\nthan model assumptions)",
        ha="center", va="center", fontsize=8, color="#2e7d32",
        style="italic", zorder=2, linespacing=1.2, clip_on=False,
    )

    # Layer guides
    for v in (L1, L2, L3):
        ax.axvline(v, color="#90a4ae", lw=0.7, ls="--", alpha=0.55, zorder=1)
        ax.axhline(v, color="#90a4ae", lw=0.7, ls="--", alpha=0.55, zorder=1)

    # --- Matched points on the Cartwright line ---
    # Coordinates chosen within each layer's band (not just its centre) so
    # that all six families named in Section 2.2 can be shown as distinct,
    # non-overlapping points rather than collapsed into one archetype.
    pts = {
        "L1": (0.17, 0.17, "#f48fb1", "#c2185b",
               "Statistical Models\n(Regression, Correlation)"),
        "L2": (L2, L2, "#ffeb3b", "#f9a825",
               "Reinforcement Learning"),
        "L3": (L3, L3, "#81c784", "#2e7d32",
               "Mechanistic Causal Audit\n(Our Approach)"),
    }
    for key, (x, y, face, edge, label) in pts.items():
        ax.scatter([x], [y], s=170, c=face, edgecolors=edge, linewidths=1.8,
                   zorder=6)
        if key == "L1":
            ax.annotate(
                label, xy=(x, y), xytext=(x + 0.035, y - 0.01),
                fontsize=8, ha="left", va="center", color="#212121",
                arrowprops=None,
                zorder=7, linespacing=1.15,
            )
        elif key == "L2":
            ax.annotate(
                label, xy=(x, y), xytext=(x - 0.02, y + 0.075),
                fontsize=8.5, ha="right", va="bottom", color="#b26a00",
                zorder=7,
            )
        else:
            ax.annotate(
                label, xy=(x, y), xytext=(x + 0.045, y + 0.015),
                fontsize=8.5, ha="left", va="center", color="#1b5e20",
                fontweight="bold", zorder=7, linespacing=1.15,
            )

    # --- Anomaly detection: L1 query, L1 model (matched, but distinct from
    # the generic statistical-models archetype above it) ---
    ax.scatter([0.085], [0.26], s=140, c="#f8bbd0", edgecolors="#ad1457",
               linewidths=1.6, zorder=6)
    ax.annotate(
        "Anomaly Detection\n" + r"($P(\text{anomalous} \mid \text{signal})$)",
        xy=(0.085, 0.26), xytext=(0.02, 0.30),
        fontsize=7.4, ha="left", va="bottom", color="#ad1457",
        arrowprops=dict(arrowstyle="-", color="#ad1457", lw=0.7),
        zorder=7, linespacing=1.15,
    )

    # --- Retry strategies: not a causal model at all ("PCH layer: none,
    # strictly"); plotted near the origin, hollow to mark that it makes no
    # real claim rather than a confirmed L1 match ---
    ax.scatter([0.03], [0.035], s=110, facecolors="#eeeeee", edgecolors="#616161",
               linewidths=1.6, linestyle="--", zorder=6, marker="o")
    ax.annotate(
        "Retry Policies\n(no causal model;\nassumes " + r"$\mathcal{L}_1$" + "-exchangeability)",
        xy=(0.03, 0.035), xytext=(0.045, 0.115),
        fontsize=7.2, ha="left", va="bottom", color="#616161",
        arrowprops=dict(arrowstyle="-", color="#616161", lw=0.7),
        zorder=7, linespacing=1.15,
    )

    # --- Causal discovery and Diehl & Ramirez-Amaro (2022): both dots sit on
    # the diagonal, with their labels routed to open space on the right
    # (below the thesis label and the vertical-gap label, above the
    # "Valid Inference" zone text) so no label text overlaps another. ---

    # Causal discovery: matched only conditionally ("in principle L2-L3, if
    # the discovered graph happens to be correct") -- hollow/dashed to flag
    # that correctness is unverified, unlike the confirmed thesis point.
    ax.scatter([0.62], [0.62], s=150, facecolors="#ede7f6", edgecolors="#5e35b1",
               linewidths=1.8, linestyle="--", zorder=6)
    ax.annotate(
        "Causal Discovery\n(valid only if the discovered\ngraph is correct)",
        xy=(0.62, 0.62), xytext=(1.05, 0.44),
        fontsize=7.3, ha="left", va="center", color="#5e35b1",
        arrowprops=dict(arrowstyle="-", color="#5e35b1", lw=0.7, ls="--"),
        zorder=7, linespacing=1.2, annotation_clip=False,
    )

    # Closest prior work: Diehl & Ramirez-Amaro (2022) -- genuinely asks an
    # L3 counterfactual query from a *learned* (not pre-registered) graph, so
    # it sits near the thesis point but hollow/dashed since its graph's
    # correctness (and hence its L3 model claim) is never checked against
    # simulated do-operator ground truth.
    ax.scatter([0.74], [0.74], s=160, facecolors="#e8eaf6", edgecolors="#283593",
               linewidths=1.8, linestyle="--", zorder=6)
    ax.annotate(
        "Diehl & Ramirez-Amaro (2022):\n"
        "learned " + r"$\mathcal{L}_3$" + " graph, unverified vs. do-operator",
        xy=(0.74, 0.74), xytext=(1.05, 0.76),
        fontsize=7.3, ha="left", va="center", color="#283593",
        arrowprops=dict(arrowstyle="-", color="#283593", lw=0.7, ls="--"),
        zorder=7, linespacing=1.2, annotation_clip=False,
    )

    # --- Mismatch: VLM / learned explanations (L1 model, L3 query) ---
    vlm_x, vlm_y = L1, L3
    ax.plot([vlm_x, vlm_x], [0, vlm_y], color="#e57373", lw=1.0, ls="--",
            alpha=0.7, zorder=2)
    ax.plot([0, vlm_x], [vlm_y, vlm_y], color="#e57373", lw=1.0, ls="--",
            alpha=0.7, zorder=2)
    ax.scatter([vlm_x], [vlm_y], s=180, c="#e53935", edgecolors="#b71c1c",
               linewidths=1.8, zorder=6)
    ax.annotate(
        "Vision-Language Models &\nLearned Explanations\n"
        r"(ask $\mathcal{L}_3$ queries with $\mathcal{L}_1$ models)",
        xy=(vlm_x, vlm_y), xytext=(vlm_x + 0.025, vlm_y - 0.025),
        fontsize=7.8, ha="left", va="center", color="#b71c1c",
        arrowprops=dict(arrowstyle="-", color="#b71c1c", lw=0.8),
        zorder=7, linespacing=1.15,
    )

    # --- L2 → L3 diagnostic gap (horizontal then vertical) ---
    # Horizontal: stronger assumptions needed
    ax.annotate(
        "", xy=(L3, L2), xytext=(L2, L2),
        arrowprops=dict(
            arrowstyle="<->", color="#fbc02d", lw=3.0,
            mutation_scale=14, shrinkA=10, shrinkB=6,
        ),
        zorder=5,
    )
    ax.text(
        (L2 + L3) / 2, L2 - 0.06,
        r"Gap to $\mathcal{L}_3$: Stronger assumptions"
        "\nneeded (SCM / counterfactuals)",
        ha="center", va="top", fontsize=8, color="#f57f17",
        zorder=7, linespacing=1.15,
    )

    # Vertical: cannot reach single-trial 'why' diagnosis
    # Place label LEFT of the arrow so it stays inside the axes
    ax.annotate(
        "", xy=(L3, L3), xytext=(L3, L2),
        arrowprops=dict(
            arrowstyle="<->", color="#fbc02d", lw=3.0,
            mutation_scale=14, shrinkA=10, shrinkB=10,
        ),
        zorder=5,
    )
    ax.text(
        L3 + 0.085, (L2 + L3) / 2,
        "Cannot reach single-trial\n‘why’ diagnosis.",
        ha="left", va="center", fontsize=8.5, color="#b26a00",
        fontweight="bold", zorder=7, linespacing=1.2,
        clip_on=False,
    )

    # --- Axes styling ---
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.margins(x=0, y=0)
    ax.set_xticks([L1, L2, L3])
    ax.set_xticklabels([
        r"$\mathcal{L}_1$", r"$\mathcal{L}_2$", r"$\mathcal{L}_3$",
    ], fontsize=10)
    ax.set_yticks([L1, L2, L3])
    ax.set_yticklabels([
        r"$\mathcal{L}_1$",
        r"$\mathcal{L}_2$",
        r"$\mathcal{L}_3$",
    ], fontsize=10)
    ax.tick_params(axis="both", length=0, pad=6)

    # Reference-style axis descriptors: horizontal above/right of the plot,
    # not conventional rotated/below-axis labels.
    ax.text(
        0.02, 1.025, "Query Expressiveness\n(diagnostic question asked)",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
        fontweight="bold", clip_on=False, linespacing=1.15,
    )
    ax.text(
        1.115, 0.03, "Model Assumptions\n(causal information encoded)",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
        fontweight="bold", clip_on=False, linespacing=1.15,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#546e7a")
    ax.spines["bottom"].set_color("#546e7a")
    ax.annotate(
        "", xy=(0, 1.055), xytext=(0, 0),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.5),
        annotation_clip=False, zorder=8,
    )
    ax.annotate(
        "", xy=(1.07, 0), xytext=(0, 0),
        xycoords=ax.transAxes, textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.5),
        annotation_clip=False, zorder=8,
    )

    # --- Left sidebar: Diagnostic Questions (Queries) ---
    # A single contiguous stacked bar (like the PCH bar below the x-axis),
    # with question text to its left and a base pedestal beneath it.
    ax_q = fig.add_axes([0.015, AX_BOTTOM, 0.205, AX_HEIGHT])
    ax_q.set_xlim(0, 1.62)
    ax_q.set_ylim(-0.10, 1.12)
    ax_q.axis("off")
    ax_q.text(
        1.28, 1.08, "Diagnostic Questions\n(Queries)",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        color="#212121", linespacing=1.2,
    )

    BAR_X0, BAR_W = 1.05, 0.52
    query_bands = [
        (2 / 3, 1 / 3, "#c8e6c9", "#2e7d32",
         "Counterfactual\n(why)\n" + r"$\mathcal{L}_3$",
         "Why did this\ntrial fail?\n" + r"$P(Y_x \mid x')$"),
        (1 / 3, 1 / 3, "#fff9c4", "#f9a825",
         "Interventional\n(what works)\n" + r"$\mathcal{L}_2$",
         "What works?\n" + r"$P(Y \mid do(X))$"),
        (0.0, 1 / 3, "#ffcdd2", "#c62828",
         "Observational\n(what happened)\n" + r"$\mathcal{L}_1$",
         "What happened?\n" + r"$P(Y \mid X)$"),
    ]
    for y0, h, face, edge, title, qtext in query_bands:
        rect = plt.Rectangle(
            (BAR_X0, y0), BAR_W, h,
            facecolor=face, edgecolor="#37474f", linewidth=1.1, zorder=2,
        )
        ax_q.add_patch(rect)
        ax_q.text(
            BAR_X0 + BAR_W / 2, y0 + h / 2, title,
            ha="center", va="center", fontsize=7.0, fontweight="bold",
            color="#212121", zorder=3, linespacing=1.25,
        )
        ax_q.text(
            BAR_X0 - 0.07, y0 + h / 2, qtext,
            ha="right", va="center", fontsize=7.6, color="#212121",
            zorder=3, linespacing=1.3,
        )

    # Divider lines between the three sections (dashed, matching the plot's
    # own L1/L2/L3 gridlines)
    for y in (1 / 3, 2 / 3):
        ax_q.plot([0.60, BAR_X0 + BAR_W], [y, y], color="#90a4ae", lw=0.8,
                  ls="--", alpha=0.7, zorder=1)

    # Base / pedestal beneath the bar
    base = plt.Rectangle(
        (BAR_X0 - 0.05, -0.075), BAR_W + 0.10, 0.045,
        facecolor="#cfd8dc", edgecolor="#90a4ae", linewidth=1.0, zorder=2,
    )
    ax_q.add_patch(base)

    # --- Horizontal PCH bar directly under the main plot's x-axis ---
    ax_pch = fig.add_axes([AX_LEFT, 0.205, AX_WIDTH * 1.13, 0.075])
    ax_pch.set_xlim(0, 1.13)
    ax_pch.set_ylim(0, 1)
    ax_pch.axis("off")

    pch_bands = [
        (0.0, 1 / 3, "#ffcdd2", "#c62828",
         "Observational\n(associations)"),
        (1 / 3, 1 / 3, "#fff9c4", "#f9a825",
         "Interventional\n(do-operator)"),
        (2 / 3, 1 / 3, "#c8e6c9", "#2e7d32",
         "Structural\n(SCM / counterfactuals)"),
    ]
    BAR_Y0, BAR_H = 0.55, 0.42
    for x0, w, face, edge, label in pch_bands:
        rect = plt.Rectangle(
            (x0, BAR_Y0), w, BAR_H,
            facecolor=face, edgecolor="#37474f", linewidth=1.1, zorder=2,
        )
        ax_pch.add_patch(rect)
        ax_pch.text(
            x0 + w / 2, BAR_Y0 - 0.14, label,
            ha="center", va="top", fontsize=8, color="#212121", zorder=3,
            linespacing=1.15,
        )

    # Dark end-cap + "PCH" label, mirroring the sidebar's pedestal
    cap = plt.Rectangle(
        (1.0, BAR_Y0), 0.03, BAR_H,
        facecolor="#37474f", edgecolor="#37474f", linewidth=0, zorder=2,
    )
    ax_pch.add_patch(cap)
    ax_pch.text(
        1.06, BAR_Y0 + BAR_H / 2, "PCH", fontsize=10, fontweight="bold",
        color="#212121", ha="left", va="center", zorder=3,
    )

    fig.text(
        AX_LEFT + AX_WIDTH / 2, 0.285, "", ha="center", va="top", fontsize=1,
    )

    # --- Bottom legend + summary ---
    ax_leg = fig.add_axes([0.04, 0.01, 0.92, 0.18])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")

    legend_items = [
        (0.00, 0.86, "#f48fb1", "#c2185b", "solid",
         r"Matched at $\mathcal{L}_1$ (statistical models, anomaly detection)"),
        (0.00, 0.70, "#ffeb3b", "#f9a825", "solid",
         r"Matched at $\mathcal{L}_2$ (reinforcement learning)"),
        (0.00, 0.54, "#81c784", "#2e7d32", "solid",
         r"Matched at $\mathcal{L}_3$, confirmed vs. ground truth (this thesis)"),
        (0.00, 0.38, "#ede7f6", "#5e35b1", "hollow",
         "Conditionally matched: valid only if the model's causal\n"
         "assumptions happen to be correct (unverified)"),
        (0.00, 0.16, "#e53935", "#b71c1c", "solid",
         "Mismatch: query exceeds model (above Cartwright line)"),
    ]
    for x, y, face, edge, style, text in legend_items:
        if style == "hollow":
            ax_leg.scatter([x + 0.012], [y], s=95, facecolors=face,
                           edgecolors=edge, linewidths=1.4, linestyle="--",
                           zorder=3, clip_on=False)
        else:
            ax_leg.scatter([x + 0.012], [y], s=95, c=face, edgecolors=edge,
                           linewidths=1.4, zorder=3, clip_on=False)
        ax_leg.text(x + 0.035, y, text, ha="left", va="center", fontsize=7.6,
                    color="#37474f", linespacing=1.25)

    # Gap / line legend
    ax_leg.plot([0.46, 0.46], [0.74, 0.90], color="#fbc02d", lw=4.0,
                solid_capstyle="round", clip_on=False)
    ax_leg.text(
        0.485, 0.82,
        "Vertical gap: Cannot reach single-trial ‘why’ diagnosis.\n"
        r"(needs $\mathcal{L}_3$ queries and $\mathcal{L}_3$ assumptions)",
        ha="left", va="center", fontsize=7.8, color="#37474f",
        linespacing=1.15,
    )
    ax_leg.plot([0.46, 0.54], [0.56, 0.56], color="#fbc02d", lw=4.0,
                solid_capstyle="round", clip_on=False)
    ax_leg.text(
        0.555, 0.56,
        "Horizontal gap: model assumptions insufficient\n"
        "(need stronger causal assumptions / structure).",
        ha="left", va="center", fontsize=7.8, color="#37474f",
        linespacing=1.15,
    )
    ax_leg.plot([0.46, 0.54], [0.34, 0.34], color="#2e7d32", lw=2.6,
                solid_capstyle="round", clip_on=False)
    ax_leg.text(
        0.555, 0.34,
        "Cartwright (Matched Expressiveness) line:\n"
        "required condition for valid causal inference.",
        ha="left", va="center", fontsize=7.8, color="#37474f",
        linespacing=1.15,
    )

    # Summary box
    summary = (
        "Valid inference requires being\n"
        "on or below the Cartwright line.\n"
        "Points above the line represent\n"
        "queries that cannot be validly\n"
        "answered with the model's\n"
        "current assumptions."
    )
    summary_box = FancyBboxPatch(
        (0.82, 0.02), 0.16, 0.48,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#fafafa", edgecolor="#90a4ae", linewidth=1.0,
        linestyle="--", transform=ax_leg.transData, clip_on=False, zorder=2,
    )
    ax_leg.add_patch(summary_box)
    ax_leg.text(
        0.90, 0.26, summary,
        ha="center", va="center", fontsize=6.8, color="#455a64",
        zorder=3, linespacing=1.25,
    )

    fig.suptitle(
        "The Diagnostic Gap on the Pearl Causal Hierarchy",
        fontsize=14, fontweight="bold", y=0.97, color="#212121",
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_diagnostic_gap.{ext}")
        fig.savefig(OUT / f"diagnostic_gap.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_diagnostic_gap.png'}")


def _rounded(ax, xy, w, h, text, face, edge, fontsize=9, bold=False, lw=1.4):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=3,
        transform=ax.transData, clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color="#212121", zorder=4, linespacing=1.2,
        transform=ax.transData, clip_on=False,
    )
    return box


def _arrow(ax, start, end, color="#455a64"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.5,
            mutation_scale=12, connectionstyle="arc3,rad=0",
        ),
        zorder=2,
    )


def fig_audit_engine():
    """Architecture-style diagram of the mechanistic causal audit.

    Left-to-right layout in the style of a robotics architecture figure:
    input tensors, dashed module groups, rounded process blocks, thin
    data bars, and a diagnosis grid as the output. Content is unchanged
    from the Yang & Bareinboim causal-inference-engine framing.
    """
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    }):
        FW, FH = 13.0, 5.72
        fig, ax = plt.subplots(figsize=(FW, FH))
        ax.set_xlim(0.0, FW)
        ax.set_ylim(0.0, FH)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        LW = 1.05
        BLACK = "black"
        MODULE_FACE = "#f0f0f0"
        DASH = (0, (4.0, 2.6))

        def sharp(x, y, w, h, lw=LW, face="white", z=3, edge=BLACK):
            ax.add_patch(mpatches.Rectangle(
                (x, y), w, h, facecolor=face, edgecolor=edge,
                linewidth=lw, joinstyle="miter", zorder=z, clip_on=False,
            ))

        def rounded(x, y, w, h, radius=0.14, lw=LW, face="white", edge=BLACK, z=3):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0.0,rounding_size={radius}",
                facecolor=face, edgecolor=edge, linewidth=lw,
                zorder=z, clip_on=False,
            ))

        def module(x, y, w, h):
            sharp(x, y, w, h, lw=0, face=MODULE_FACE, z=0, edge="none")
            ax.add_patch(mpatches.Rectangle(
                (x, y), w, h, facecolor="none", edgecolor=BLACK,
                linewidth=1.15, linestyle=DASH, zorder=1, clip_on=False,
            ))

        def txt(x, y, s, fs=9.0, w="normal", ha="center", va="center", z=5):
            ax.text(
                x, y, s, ha=ha, va=va, fontsize=fs, fontweight=w,
                color=BLACK, zorder=z, linespacing=1.22, clip_on=False,
            )

        def arr(p0, p1, lw=LW, ms=10.0):
            ax.annotate(
                "", xy=p1, xytext=p0,
                arrowprops=dict(
                    arrowstyle="-|>", color=BLACK, lw=lw,
                    mutation_scale=ms, shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0",
                ),
                zorder=2,
            )

        def hline(x0, x1, y, lw=LW):
            ax.plot([x0, x1], [y, y], color=BLACK, lw=lw,
                    solid_capstyle="butt", zorder=2, clip_on=False)

        def vline(x, y0, y1, lw=LW):
            ax.plot([x, x], [y0, y1], color=BLACK, lw=lw,
                    solid_capstyle="butt", zorder=2, clip_on=False)

        # Shared vertical band. xlim/ylim match figsize so 1 unit = 1 inch.
        Y0, Y1 = 1.42, 4.18
        YM = 0.5 * (Y0 + Y1)
        TH = Y1 - Y0
        TALL_PAD = 0.22
        LAB_Y, SUB_Y = 0.72, 0.42

        # ------------------------------------------------------------------
        # Inputs: Query Q, stacked observations D, graphical model G
        # ------------------------------------------------------------------
        q_x, q_w = 0.22, 0.72
        d_x, d_bw = 1.06, 0.30
        d_n = 4
        d_w = d_n * d_bw
        g_x, g_w = 2.38, 0.72
        in_right = g_x + g_w

        # L3 query banner (too long for a thin tensor; kept as a wide label).
        banner_h = 0.40
        banner_y = 4.78
        rounded(q_x, banner_y, in_right - q_x, banner_h, radius=0.08)
        txt(
            0.5 * (q_x + in_right), banner_y + banner_h / 2,
            r"Counterfactual $\mathcal{L}_3$:  ``If $\sigma_d{=}0$, would the grasp succeed?''",
            fs=10.0,
        )

        sharp(q_x, Y0, q_w, TH)
        txt(q_x + q_w / 2, Y0 + TH * 0.62, r"$\mathcal{Q}$", fs=16.0, w="bold")
        txt(q_x + q_w / 2, Y0 + TH * 0.32, r"$\mathcal{L}_3$", fs=10.0)
        txt(q_x + q_w / 2, 4.38, "demands", fs=8.5)

        d_labels = [r"$\sigma_d$", r"$\rho$", r"$\phi$", r"$\theta$"]
        for i, lab in enumerate(d_labels):
            sharp(
                d_x + i * d_bw, Y0, d_bw, TH,
                face="#ffffff" if i % 2 == 0 else "#e9e9e9",
            )
            txt(d_x + (i + 0.5) * d_bw, YM, lab, fs=11.0)
        txt(d_x + d_w / 2, 4.38, "logs", fs=8.5)

        sharp(g_x, Y0, g_w, TH)
        txt(g_x + g_w / 2, Y0 + TH * 0.62, r"$\mathcal{G}$", fs=16.0, w="bold")
        txt(g_x + g_w / 2, Y0 + TH * 0.32, "SCM", fs=10.0)
        txt(g_x + g_w / 2, 4.38, "architecture", fs=8.5)

        br = in_right + 0.11
        vline(br, Y0, Y1)
        hline(br - 0.08, br, Y1)
        hline(br - 0.08, br, Y0)

        txt(
            0.5 * (q_x + in_right), LAB_Y,
            r"Physical Reality $\mathcal{M}^{*}$",
            fs=10.5, w="bold",
        )
        txt(
            0.5 * (q_x + in_right), SUB_Y,
            "(unobserved grasp attempt)",
            fs=8.0,
        )

        # ------------------------------------------------------------------
        # Module 1: MuJoCo testbed
        # ------------------------------------------------------------------
        m1_x, m1_w = 3.42, 2.72
        m1_y, m1_h = 1.18, 3.32
        module(m1_x, m1_y, m1_w, m1_h)

        muj_w, muj_h = 1.52, 1.92
        muj_x = m1_x + 0.16
        muj_y = YM - muj_h / 2
        rounded(muj_x, muj_y, muj_w, muj_h, radius=0.18)
        txt(muj_x + muj_w / 2, muj_y + muj_h * 0.64, "MuJoCo", fs=12.0, w="bold")
        txt(muj_x + muj_w / 2, muj_y + muj_h * 0.42, "testbed", fs=10.5)
        txt(
            muj_x + muj_w / 2, muj_y + muj_h * 0.18,
            r"direct $\mathrm{do}(\cdot)$",
            fs=8.0,
        )

        do_w = 0.68
        do_x = m1_x + m1_w - do_w - 0.16
        sharp(do_x, Y0 + TALL_PAD, do_w, TH - 2 * TALL_PAD)
        txt(do_x + do_w / 2, YM + 0.16, r"$\mathrm{do}(\cdot)$", fs=9.0)
        txt(do_x + do_w / 2, YM - 0.22, "trial", fs=8.0)

        txt(m1_x + m1_w / 2, LAB_Y, "MuJoCo Testbed", fs=10.5, w="bold")
        txt(
            m1_x + m1_w / 2, SUB_Y,
            r"direct $\mathrm{do}(\cdot)$ intervention",
            fs=8.0,
        )

        arr((br, YM), (muj_x, YM))
        arr((muj_x + muj_w, YM), (do_x, YM))

        # ------------------------------------------------------------------
        # Module 2: Pearl counterfactual procedure
        # ------------------------------------------------------------------
        m2_x, m2_w = 6.32, 4.28
        module(m2_x, m1_y, m2_w, m1_h)

        proc_h = 0.82
        proc_y = YM - proc_h / 2
        gap = 0.11
        abd_w, u_w, act_w, pred_w, yhat_w = 1.02, 0.40, 0.86, 1.02, 0.38
        abd_x = m2_x + 0.16
        u_x = abd_x + abd_w + gap
        act_x = u_x + u_w + gap
        pred_x = act_x + act_w + gap
        yhat_x = pred_x + pred_w + gap

        rounded(abd_x, proc_y, abd_w, proc_h, radius=0.14)
        txt(abd_x + abd_w / 2, YM + 0.12, "Abduction", fs=10.0, w="bold")
        txt(abd_x + abd_w / 2, YM - 0.16, r"recover $U$", fs=8.0)

        sharp(u_x, Y0 + TALL_PAD, u_w, TH - 2 * TALL_PAD)
        txt(u_x + u_w / 2, YM, r"$U$", fs=13.0, w="bold")

        rounded(act_x, proc_y, act_w, proc_h, radius=0.14)
        txt(act_x + act_w / 2, YM + 0.12, "Action", fs=10.0, w="bold")
        txt(act_x + act_w / 2, YM - 0.16, r"$\mathrm{do}(x')$", fs=8.0)

        rounded(pred_x, proc_y, pred_w, proc_h, radius=0.14)
        txt(pred_x + pred_w / 2, YM + 0.12, "Prediction", fs=10.0, w="bold")
        txt(pred_x + pred_w / 2, YM - 0.16, r"$\hat{Y}_{x'}$", fs=8.5)

        sharp(yhat_x, Y0 + TALL_PAD, yhat_w, TH - 2 * TALL_PAD)
        txt(yhat_x + yhat_w / 2, YM, r"$\hat{Y}$", fs=10.5, w="bold")

        txt(
            m2_x + m2_w / 2, LAB_Y,
            "Causal Inference Engine", fs=10.5, w="bold",
        )
        txt(
            m2_x + m2_w / 2, SUB_Y,
            "abduction, action, prediction",
            fs=8.0,
        )

        arr((do_x + do_w, YM), (abd_x, YM))
        arr((abd_x + abd_w, YM), (u_x, YM))
        arr((u_x + u_w, YM), (act_x, YM))
        arr((act_x + act_w, YM), (pred_x, YM))
        arr((pred_x + pred_w, YM), (yhat_x, YM))

        # ------------------------------------------------------------------
        # Outputs: diagnosis grid + counterfactual buffer
        # ------------------------------------------------------------------
        out_x = 10.82
        grid_w = 1.92
        grid_h = 1.92
        grid_y = 2.42
        sharp(out_x, grid_y, grid_w, grid_h, lw=1.15)
        n_grid = 8
        for i in range(1, n_grid):
            gx = out_x + i * grid_w / n_grid
            gy = grid_y + i * grid_h / n_grid
            ax.plot(
                [gx, gx], [grid_y, grid_y + grid_h],
                color="#c8c8c8", lw=0.5, zorder=4, clip_on=False,
            )
            ax.plot(
                [out_x, out_x + grid_w], [gy, gy],
                color="#c8c8c8", lw=0.5, zorder=4, clip_on=False,
            )
        # White knockout so the labels stay readable on the grid.
        panel_w, panel_h = 1.52, 0.78
        rounded(
            out_x + (grid_w - panel_w) / 2,
            grid_y + (grid_h - panel_h) / 2,
            panel_w, panel_h, radius=0.06, lw=0.0, edge="none", z=4,
        )
        txt(
            out_x + grid_w / 2, grid_y + grid_h / 2 + 0.14,
            "Diagnosis", fs=11.0, w="bold",
        )
        txt(
            out_x + grid_w / 2, grid_y + grid_h / 2 - 0.16,
            "geometry or\nperception?", fs=8.5,
        )

        buf_h = 0.82
        buf_y = 1.28
        sharp(out_x, buf_y, grid_w, buf_h)
        txt(
            out_x + grid_w / 2, buf_y + buf_h * 0.64,
            r"Counterfactual $\hat{Y}$", fs=8.5, w="bold",
        )
        txt(
            out_x + grid_w / 2, buf_y + buf_h * 0.28,
            r"$Y_{x'}\in\{0,1\}$", fs=8.0,
        )

        split_x = yhat_x + yhat_w + 0.10
        hline(yhat_x + yhat_w, split_x, YM)
        vline(split_x, buf_y + buf_h / 2, grid_y + grid_h / 2)
        arr((split_x, grid_y + grid_h / 2), (out_x, grid_y + grid_h / 2))
        arr((split_x, buf_y + buf_h / 2), (out_x, buf_y + buf_h / 2))

        for dest in (OUT, fig_dir):
            for ext in ("png", "pdf"):
                fig.savefig(
                    dest / f"fig_mechanistic_audit_engine.{ext}",
                    facecolor="white", bbox_inches="tight", pad_inches=0.12,
                )
        plt.close(fig)
        print(f"Wrote {OUT / 'fig_mechanistic_audit_engine.png'}")


if __name__ == "__main__":
    fig_diagnostic_gap()
    fig_audit_engine()
