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
    """Causal inference engine adapted to the mechanistic grasp audit."""
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    # --- Unobserved SCM (left) ---
    _rounded(
        ax, (0.25, 2.0), 2.0, 1.8,
        "Physical Reality of\nthe Grasp Attempt\n(Unobserved $\\mathcal{M}^*$)",
        "#f3e5f5", "#6a1b9a", fontsize=9, bold=True, lw=1.8,
    )

    # --- Three inputs ---
    _rounded(
        ax, (3.1, 4.35), 2.35, 1.15,
        "1. Query $\\mathcal{Q}$\n"
        "Counterfactual $\\mathcal{L}_3$:\n"
        r"``If $\sigma_d{=}0$, would" + "\nthis grasp have succeeded?''",
        "#e3f2fd", "#1565c0", fontsize=8,
    )
    _rounded(
        ax, (3.1, 2.35), 2.35, 1.15,
        "2. Observed Data $\\mathcal{D}$\n"
        "Failed-trial logs:\n"
        r"$\sigma_d$, $\rho$, $\phi$, $\theta$",
        "#e8f5e9", "#2e7d32", fontsize=8.5,
    )
    _rounded(
        ax, (3.1, 0.35), 2.35, 1.15,
        "3. Graphical Model $\\mathcal{G}$\n"
        "Pre-registered SCM from\n"
        "robot architecture",
        "#fff8e1", "#f9a825", fontsize=8.5,
    )

    # Arrows from reality to inputs
    _arrow(ax, (2.25, 3.55), (3.1, 4.7), "#6a1b9a")
    ax.text(2.55, 4.25, "demands", fontsize=7, color="#6a1b9a", style="italic", rotation=25)

    _arrow(ax, (2.25, 2.9), (3.1, 2.9), "#6a1b9a")
    ax.text(2.45, 3.05, "logs", fontsize=7, color="#6a1b9a", style="italic")

    _arrow(ax, (2.25, 2.25), (3.1, 1.1), "#6a1b9a")
    ax.text(2.4, 1.45, "architecture", fontsize=7, color="#6a1b9a", style="italic", rotation=-30)

    # --- Causal Inference Engine ---
    engine = FancyBboxPatch(
        (5.85, 0.9), 2.55, 4.0,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#e1f5fe", edgecolor="#01579b", linewidth=2.0, zorder=1,
    )
    ax.add_patch(engine)
    ax.text(
        7.125, 4.6, "Causal Inference Engine",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#01579b", zorder=4,
    )

    _rounded(
        ax, (6.05, 3.35), 2.15, 0.85,
        "MuJoCo Testbed\n(direct $do(\\cdot)$ operator)",
        "#b3e5fc", "#0277bd", fontsize=8.5, bold=True,
    )
    _rounded(
        ax, (6.05, 2.35), 2.15, 0.7,
        "Abduction",
        "#ffffff", "#0277bd", fontsize=9,
    )
    _rounded(
        ax, (6.05, 1.5), 2.15, 0.7,
        "Action",
        "#ffffff", "#0277bd", fontsize=9,
    )
    _rounded(
        ax, (6.05, 0.65), 2.15, 0.7,
        "Prediction",
        "#ffffff", "#0277bd", fontsize=9,
    )

    # Vertical flow inside engine
    _arrow(ax, (7.125, 3.35), (7.125, 3.05), "#0277bd")
    _arrow(ax, (7.125, 2.35), (7.125, 2.2), "#0277bd")
    _arrow(ax, (7.125, 1.5), (7.125, 1.35), "#0277bd")

    # Inputs → engine
    _arrow(ax, (5.45, 4.9), (5.85, 3.9), "#455a64")
    _arrow(ax, (5.45, 2.9), (5.85, 2.9), "#455a64")
    _arrow(ax, (5.45, 0.9), (5.85, 1.5), "#455a64")

    # --- Output ---
    _rounded(
        ax, (8.75, 2.15), 1.55, 1.5,
        "Diagnosis\nGeometry\nor\nPerception?",
        "#e8eaf6", "#283593", fontsize=9, bold=True, lw=1.8,
    )
    _arrow(ax, (8.4, 2.9), (8.75, 2.9), "#01579b")

    fig.suptitle(
        "The Mechanistic Causal Audit as a Causal Inference Engine",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_mechanistic_audit_engine.{ext}")
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_mechanistic_audit_engine.png'}")


if __name__ == "__main__":
    fig_diagnostic_gap()
    fig_audit_engine()
