#!/usr/bin/env python3
"""Teaser figure: associative diagnosis vs mechanistic causal audit.

Scenario comparison on top, architecture below (ActiveVLA-style layout).
Uses existing MuJoCo, depth, and viewpoint renders.

Outputs:
  results/figures/fig_teaser_diagnosis.pdf
  results/figures/fig_teaser_diagnosis.png
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Ellipse
import numpy as np
from PIL import Image

ROOT = Path("/Users/bonolomasima/Desktop/Reasoning_via_Inference")
FIG = ROOT / "results" / "figures"

# Physical size chosen so 9–11 pt labels stay readable at A4 textwidth.
FW, FH = 10.50, 7.15

PINK = "#F4C6CE"
PINK_E = "#C97A86"
GOLD = "#F3DC7A"
GOLD_E = "#C9A83A"
PURPLE = "#C9B8EA"
PURPLE_E = "#7E62B8"
CARD = "#F7F7F7"
CARD_E = "#C8C8C8"
NAVY = "#1F3347"
INK = "#222222"
MUTED = "#5A5A5A"
FAIL = "#C0392B"
OK = "#1E7A45"
SOFTRED = "#FDECEC"
SOFTGRN = "#E8F6EE"
WHITE = "#FFFFFF"
CAM_BG = "#FFF8E7"


def rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load(path, crop=None):
    im = Image.open(path).convert("RGB")
    if crop is not None:
        im = im.crop(crop)
    return np.asarray(im)


def crop_table(path, l=0.10, t=0.06, r=0.90, b=0.98):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return np.asarray(im.crop((int(w * l), int(h * t), int(w * r), int(h * b))))


def fig_rect(x, y, w, h):
    return [x / FW, y / FH, w / FW, h / FH]


def rounded(ax, x, y, w, h, fc=CARD, ec=CARD_E, r=0.10, lw=1.05, z=1):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        mutation_aspect=1.0, clip_on=False,
    )
    ax.add_patch(p)
    return p


def add_img(fig, arr, x, y, w, h, z=3, r=0.05):
    ax = fig.add_axes(fig_rect(x, y, w, h), zorder=z)
    ax.imshow(arr, aspect="auto")
    ax.set_axis_off()
    clip = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        transform=ax.transAxes, facecolor="none", edgecolor="none",
    )
    ax.add_patch(clip)
    ax.images[0].set_clip_path(clip)
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        transform=ax.transAxes, facecolor="none",
        edgecolor="#9A9A9A", linewidth=0.65, clip_on=False,
    ))
    return ax


def txt(ax, x, y, s, size=8.0, color=INK, weight="regular", ha="left",
        va="center", z=12, style="normal"):
    return ax.text(
        x, y, s, fontsize=size, color=color, fontweight=weight,
        ha=ha, va=va, zorder=z, clip_on=False, style=style,
    )


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.05, rad=0.0, z=8):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=8.5, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}",
        zorder=z, clip_on=False,
    ))


def tokens(ax, x, y, n, color, size=0.11, gap=0.035, z=9):
    for i in range(n):
        rounded(ax, x + i * (size + gap), y, size, size,
                fc=color, ec="#FFFFFF", r=0.022, lw=0.35, z=z)


def pill(ax, fg, x, y, w, h, s, fc, ec, tc, size=7.2):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, r=h * 0.42, lw=0.85, z=11)
    txt(fg, x + w / 2, y + h / 2, s, size=size, color=tc, weight="bold",
        ha="center", va="center")


def icon_person(ax, x, y, s=0.15, color=NAVY, z=12):
    ax.add_patch(Circle((x, y + s * 0.40), s * 0.20, fc=color, ec="none",
                        zorder=z, clip_on=False))
    ax.add_patch(Wedge((x, y - s * 0.10), s * 0.34, 8, 172,
                       fc=color, ec="none", zorder=z, clip_on=False))


def icon_camera(ax, x, y, s=0.18, z=12):
    rounded(ax, x - s * 0.52, y - s * 0.30, s * 1.05, s * 0.68,
            fc="#2C2C2C", ec="none", r=0.025, z=z)
    ax.add_patch(Circle((x + s * 0.08, y), s * 0.18, fc="#F4F4F4",
                        ec="#2C2C2C", lw=0.5, zorder=z + 1, clip_on=False))
    ax.add_patch(Circle((x + s * 0.08, y), s * 0.09, fc="#4A90C8",
                        ec="none", zorder=z + 2, clip_on=False))


def icon_brain(ax, x, y, s=0.16, color=PINK_E, z=12):
    ax.add_patch(Ellipse((x, y), s * 1.10, s * 0.90, fc=color, ec="none",
                         zorder=z, clip_on=False))
    ax.add_patch(Ellipse((x - s * 0.20, y + s * 0.08), s * 0.52, s * 0.52,
                         fc=color, ec="none", zorder=z, clip_on=False))
    ax.add_patch(Ellipse((x + s * 0.20, y + s * 0.08), s * 0.52, s * 0.52,
                         fc=color, ec="none", zorder=z, clip_on=False))


def load_assets():
    depth = FIG / "thesis_renders" / "fig_depth_degradation.png"
    # Crop past montage title and per-panel σ_d captions.
    clean = load(depth, crop=(28, 228, 632, 660))
    noisy = load(depth, crop=(2644, 228, 3248, 660))
    return dict(
        clean=clean,
        noisy=noisy,
        arm=crop_table(
            FIG / "arm_lift_v4" / "arm_lift_cylinder_thin_friction_03_grasp_open_closeup.png",
            0.06, 0.00, 0.96, 1.00,
        ),
        approach=crop_table(FIG / "pickup_demo" / "cylinder_1_approach.png", 0.14, 0.08, 0.88, 0.99),
        lift=crop_table(FIG / "pickup_demo" / "cylinder_4_full_lift_shake.png", 0.14, 0.08, 0.88, 0.99),
        box=crop_table(FIG / "pickup_demo" / "box_1_approach.png", 0.14, 0.08, 0.88, 0.99),
        mustard=crop_table(FIG / "pickup_demo" / "mustard_1_approach.png", 0.14, 0.08, 0.88, 0.99),
        sphere=load(FIG / "fig_viewing_sphere.png", crop=(110, 70, 1570, 1540)),
    )


def build():
    rc()
    A = load_assets()

    fig = plt.figure(figsize=(FW, FH), facecolor=WHITE, dpi=220)
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.set_xlim(0, FW)
    bg.set_ylim(0, FH)
    bg.axis("off")
    bg.set_facecolor(WHITE)

    fg = fig.add_axes([0, 0, 1, 1], zorder=12)
    fg.set_xlim(0, FW)
    fg.set_ylim(0, FH)
    fg.axis("off")
    fg.patch.set_alpha(0.0)

    m, gap = 0.13, 0.10
    inner = FW - 2 * m

    head_h, top_h, lab_h = 0.26, 3.68, 0.22
    bot_h = FH - m - head_h - top_h - gap - lab_h - m
    bot_y = m
    top_y = bot_y + bot_h + lab_h + gap

    # Give the "this work" panels more width (they carry the extra views).
    wA, wB = 2.58, 2.90
    wC = inner - wA - wB - 2 * gap
    xA, xB, xC = m, m + wA + gap, m + wA + gap + wB + gap

    wD = 3.52
    wE = inner - wD - gap
    xD, xE = m, m + wD + gap

    txt(fg, m, top_y + top_h + 0.08,
        r"(a)  A grasp fails. Which perceptual factor caused it?",
        size=9.7, weight="bold", color=NAVY, va="bottom")
    txt(fg, m, bot_y + bot_h + 0.05,
        r"(b)  Architecture: associative $\mathcal{L}_1$  vs.  structural $\mathcal{L}_3$",
        size=9.7, weight="bold", color=NAVY, va="bottom")

    # Cards
    rounded(bg, xA, top_y, wA, top_h, fc=CARD, ec=CARD_E, r=0.09, z=1)
    rounded(bg, xB, top_y, wB, top_h, fc=CARD, ec=CARD_E, r=0.09, z=1)
    rounded(bg, xC, top_y, wC, top_h, fc="#F4F1FA", ec=PURPLE_E, r=0.09, lw=1.25, z=1)
    rounded(bg, xD, bot_y, wD, bot_h, fc=CARD, ec=CARD_E, r=0.09, z=1)
    rounded(bg, xE, bot_y, wE, bot_h, fc="#F4F1FA", ec=PURPLE_E, r=0.09, lw=1.25, z=1)

    # =====================================================================
    # (a) LEFT — failed attempt
    # =====================================================================
    pad = 0.10
    txt(fg, xA + pad, top_y + top_h - 0.16, "Failed grasp attempt",
        size=8.3, weight="bold", color=NAVY)

    qh = 0.40
    qy = top_y + top_h - 0.62
    rounded(bg, xA + pad, qy, wA - 2 * pad, qh, fc=WHITE, ec="#D0D0D0", r=0.07, z=2)
    icon_person(bg, xA + pad + 0.16, qy + qh * 0.50, s=0.18)
    txt(fg, xA + pad + 0.32, qy + 0.27, "Why did this grasp fail?",
        size=6.5, weight="bold")
    txt(fg, xA + pad + 0.32, qy + 0.12,
        r"Noise $\sigma_d$, sparsity $\rho$, or viewpoint $(\phi,\theta)$?",
        size=5.3, color=MUTED)

    # Main scene
    thumbs_h = 0.62
    scene_y = top_y + 0.18 + thumbs_h + 0.22
    scene_h = qy - 0.10 - scene_y
    scene_x = xA + pad
    scene_w = wA - 2 * pad
    add_img(fig, A["arm"], scene_x, scene_y, scene_w, scene_h, z=3)

    # Degraded-view inset (chrome on fg so it sits above the scene photo)
    iw, ih = 1.02, 0.62
    ix, iy = scene_x + 0.06, scene_y + 0.06
    rounded(fg, ix - 0.04, iy - 0.04, iw + 0.08, ih + 0.32,
            fc=CAM_BG, ec="#C9A227", r=0.05, lw=0.95, z=13)
    ax_in = add_img(fig, A["noisy"], ix, iy, iw, ih, z=14, r=0.04)
    ax_in.text(
        0.50, 0.10, "degraded depth", transform=ax_in.transAxes,
        ha="center", va="bottom", fontsize=6.2, fontweight="bold",
        color=WHITE, zorder=20, clip_on=True,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="#1A1A1A")],
    )
    icon_camera(fg, ix + 0.14, iy + ih + 0.16, s=0.14)
    txt(fg, ix + 0.30, iy + ih + 0.16, "Current view",
        size=6.0, weight="bold", color="#3D2A00")

    # Three objects
    tw = (scene_w - 0.08) / 3
    ty = top_y + 0.20
    for i, (im, lab) in enumerate([
        (A["approach"], "cylinder"),
        (A["box"], "box"),
        (A["mustard"], "mustard"),
    ]):
        add_img(fig, im, scene_x + i * (tw + 0.04), ty, tw, thumbs_h, z=3, r=0.04)
        txt(fg, scene_x + i * (tw + 0.04) + tw / 2, ty - 0.09,
            lab, size=5.2, color=MUTED, ha="center")

    # =====================================================================
    # (a) MIDDLE — previous methods (stacked, not side-by-side)
    # =====================================================================
    txt(fg, xB + pad, top_y + top_h - 0.16, "Previous methods",
        size=8.3, weight="bold", color=NAVY)
    txt(fg, xB + pad, top_y + top_h - 0.34,
        "Same failed trial, observational logs only",
        size=5.5, color=MUTED)

    rounded(bg, xB + pad, top_y + top_h - 0.70, wB - 2 * pad, 0.26,
            fc=WHITE, ec="#D0D0D0", r=0.06, z=2)
    txt(fg, xB + pad + 0.08, top_y + top_h - 0.57,
        r"Instruction:  “Diagnose this failure.”", size=5.9)

    # Depth observation (full card width)
    dw = wB - 2 * pad
    dh = 0.88
    dy = top_y + 1.72
    add_img(fig, A["noisy"], xB + pad, dy, dw, dh, z=3)
    txt(fg, xB + wB / 2, dy - 0.09, r"Observation  ($\sigma_d{=}0.04$, occluded / noisy)",
        size=5.2, color=MUTED, ha="center")

    # LLM bubble
    bw = wB - 2 * pad
    bh = 0.78
    by = top_y + 0.78
    rounded(bg, xB + pad, by, bw, bh, fc=WHITE, ec=PINK_E, r=0.08, lw=1.0, z=2)
    icon_brain(bg, xB + pad + 0.18, by + bh - 0.18, s=0.14, color=PINK_E)
    txt(fg, xB + pad + 0.34, by + bh - 0.18, "LLM  (associative,  $\\mathcal{L}_1$)",
        size=6.0, weight="bold", color=PINK_E)
    txt(fg, xB + pad + 0.10, by + 0.42, r"“Cause: depth noise $\sigma_d$”", size=6.0)
    txt(fg, xB + pad + 0.10, by + 0.26, r"No $\mathrm{do}(\cdot)$ rewind. Cannot test whether",
        size=5.3, color=MUTED)
    txt(fg, xB + pad + 0.10, by + 0.12, "fixing that factor would have saved the grasp.",
        size=5.3, color=MUTED)

    pill(bg, fg, xB + pad, top_y + 0.14, wB - 2 * pad, 0.36,
         "Unverified attribution   ✗", SOFTRED, FAIL, FAIL, size=7.3)

    # =====================================================================
    # (a) RIGHT — this work
    # =====================================================================
    txt(fg, xC + pad, top_y + top_h - 0.16, "This work  (mechanistic causal audit)",
        size=8.3, weight="bold", color=PURPLE_E)
    txt(fg, xC + pad, top_y + top_h - 0.34,
        "Same failed trial  →  rewind in MuJoCo under one intervention",
        size=5.5, color=MUTED)

    rounded(bg, xC + pad, top_y + top_h - 0.70, wC - 2 * pad, 0.26,
            fc=WHITE, ec="#D0D0D0", r=0.06, z=2)
    txt(fg, xC + pad + 0.08, top_y + top_h - 0.57,
        r"The failure may be noise, viewpoint, or irreducibly both.",
        size=5.9)

    # Two counterfactual views
    cw = (wC - 2 * pad - 0.10) / 2
    ch = 1.12
    cy = top_y + 1.58
    add_img(fig, A["clean"], xC + pad, cy, cw, ch, z=3)
    add_img(fig, A["sphere"], xC + pad + cw + 0.10, cy, cw, ch, z=3)
    txt(fg, xC + pad + cw / 2, cy + ch + 0.09,
        r"Actively rewind:  $\mathrm{do}(\sigma_d{=}0)$",
        size=5.7, weight="bold", color=PURPLE_E, ha="center")
    txt(fg, xC + pad + cw + 0.10 + cw / 2, cy + ch + 0.09,
        r"Actively rewind:  $\mathrm{do}(\phi{=}55^\circ)$",
        size=5.7, weight="bold", color=PURPLE_E, ha="center")
    txt(fg, xC + pad + cw / 2, cy - 0.09, "clean depth",
        size=5.2, color=MUTED, ha="center")
    txt(fg, xC + pad + cw + 0.10 + cw / 2, cy - 0.09, r"viewpoint $(\phi,\theta)$",
        size=5.2, color=MUTED, ha="center")

    # Lift + Pearl steps
    ly = top_y + 0.52
    lh, lw = 0.88, 1.52
    add_img(fig, A["lift"], xC + pad, ly, lw, lh, z=3)
    txt(fg, xC + pad + lw / 2, ly - 0.09, r"Counterfactual world  ($Y{=}1$)",
        size=5.1, color=MUTED, ha="center")

    px = xC + pad + lw + 0.10
    pw = wC - 2 * pad - lw - 0.10
    rounded(bg, px, ly, pw, lh, fc=WHITE, ec=PURPLE_E, r=0.07, lw=0.9, z=2)
    txt(fg, px + 0.08, ly + lh - 0.14, "Pearl  (abduction → action → prediction)",
        size=5.7, weight="bold", color=PURPLE_E)
    txt(fg, px + 0.08, ly + 0.56, r"1.  Abduct $U$ from the failed trial", size=5.4)
    txt(fg, px + 0.08, ly + 0.40, r"2.  $\mathrm{do}(X{=}x')$ on one exogenous factor", size=5.4)
    txt(fg, px + 0.08, ly + 0.24, r"3.  Predict $Y_{x'}$ vs simulated ground truth", size=5.4)
    txt(fg, px + 0.08, ly + 0.10, "Single cause, joint causes, or irreducible.",
        size=5.1, style="italic", color=MUTED)

    pill(bg, fg, xC + pad, top_y + 0.10, wC - 2 * pad, 0.32,
         "Diagnosed vs simulated ground truth   ✓", SOFTGRN, OK, OK, size=7.3)

    # =====================================================================
    # (b) LEFT — previous architecture
    # =====================================================================
    txt(fg, xD + pad, bot_y + bot_h - 0.16, "Previous methods",
        size=8.3, weight="bold", color=NAVY)

    ih, iw = 0.68, 1.38
    iy = bot_y + bot_h - 1.02
    gap_in = 0.10
    rounded(bg, xD + pad, iy, iw, ih, fc=WHITE, ec="#CFCFCF", r=0.06, z=2)
    txt(fg, xD + pad + iw / 2, iy + ih - 0.12, "Observation",
        size=5.7, weight="bold", color=NAVY, ha="center")
    txt(fg, xD + pad + 0.08, iy + 0.38, r"$\sigma_d{=}0.04$", size=5.4)
    txt(fg, xD + pad + 0.08, iy + 0.24, r"$\phi{=}45^\circ$", size=5.4)
    txt(fg, xD + pad + 0.08, iy + 0.10, r"$Y{=}0$", size=5.4, color=FAIL, weight="bold")

    rounded(bg, xD + pad + iw + gap_in, iy, iw, ih, fc=WHITE, ec="#CFCFCF", r=0.06, z=2)
    txt(fg, xD + pad + iw + gap_in + iw / 2, iy + ih - 0.12, "Instruction",
        size=5.7, weight="bold", color=NAVY, ha="center")
    txt(fg, xD + pad + iw + gap_in + 0.08, iy + 0.36, "Diagnose this", size=5.3)
    txt(fg, xD + pad + iw + gap_in + 0.08, iy + 0.20, "grasp failure.", size=5.3)

    llm_w, llm_h = wD - 2 * pad - 0.20, 0.50
    llm_x = xD + pad + 0.10
    llm_y = bot_y + 0.88
    arrow(bg, xD + pad + iw / 2, iy, llm_x + llm_w * 0.35, llm_y + llm_h)
    arrow(bg, xD + pad + iw + gap_in + iw / 2, iy, llm_x + llm_w * 0.65, llm_y + llm_h)

    rounded(bg, llm_x, llm_y, llm_w, llm_h, fc=PINK, ec=PINK_E, r=0.08, lw=1.1, z=2)
    txt(fg, llm_x + llm_w / 2, llm_y + llm_h / 2 + 0.08, "Pre-trained LLM",
        size=7.3, weight="bold", color=NAVY, ha="center")
    txt(fg, llm_x + llm_w / 2, llm_y + llm_h / 2 - 0.10, r"(associative, $\mathcal{L}_1$)",
        size=5.4, color=MUTED, ha="center")

    arrow(bg, llm_x + llm_w / 2, llm_y, llm_x + llm_w / 2, bot_y + 0.58)
    tokens(bg, llm_x + llm_w / 2 - 0.58, bot_y + 0.38, 8, PINK_E)
    txt(fg, llm_x + llm_w / 2, bot_y + 0.24, "Language tokens  (unverified cause)",
        size=5.5, color=MUTED, ha="center")

    # =====================================================================
    # (b) RIGHT — this work architecture
    # =====================================================================
    txt(fg, xE + pad, bot_y + bot_h - 0.16, "This work",
        size=8.3, weight="bold", color=PURPLE_E)

    # Three inputs
    n_in = 3
    in_gap = 0.08
    in_w = (wE - 2 * pad - (n_in - 1) * in_gap) / n_in
    in_h = 0.70
    in_y = bot_y + bot_h - 1.04
    in_labels = [
        ("Observation", r"$\sigma_d,\rho,\phi,\theta$", r"$Y{=}0$  (failed)", INK, FAIL),
        ("Query", r"If $\mathrm{do}(X{=}x')$,", r"would $Y{=}1$?", INK, INK),
        ("3D / pipeline", r"camera $(\phi,\theta)$", "point cloud", INK, INK),
    ]
    xs = [xE + pad + i * (in_w + in_gap) for i in range(3)]
    for i, (title, l1, l2, c1, c2) in enumerate(in_labels):
        rounded(bg, xs[i], in_y, in_w, in_h, fc=WHITE, ec="#CFCFCF", r=0.06, z=2)
        txt(fg, xs[i] + in_w / 2, in_y + in_h - 0.12, title,
            size=5.6, weight="bold", color=NAVY, ha="center")
        txt(fg, xs[i] + 0.08, in_y + 0.36, l1, size=5.2, color=c1)
        txt(fg, xs[i] + 0.08, in_y + 0.20, l2, size=5.2, color=c2)

    add_img(fig, A["sphere"], xs[2] + in_w - 0.72, in_y + 0.07, 0.64, 0.56, z=5, r=0.04)

    # Row 2: SCM → CF → intervened → diagnosis
    row2_y = bot_y + 0.72
    row2_h = 0.58
    boxes = 4
    bgap = 0.10
    # widths: SCM, CF, view, diagnosis
    bws = np.array([1.70, 1.72, 1.18, 1.42])
    bws = bws * (wE - 2 * pad - (boxes - 1) * bgap) / bws.sum()
    bx = [xE + pad]
    for i in range(3):
        bx.append(bx[-1] + bws[i] + bgap)

    # arrows from inputs into SCM
    arrow(bg, xs[0] + in_w / 2, in_y, bx[0] + bws[0] * 0.35, row2_y + row2_h, lw=0.9)
    arrow(bg, xs[1] + in_w / 2, in_y, bx[0] + bws[0] * 0.55, row2_y + row2_h, lw=0.9)
    arrow(bg, xs[2] + in_w / 2, in_y, bx[0] + bws[0] * 0.80, row2_y + row2_h, lw=0.9)

    rounded(bg, bx[0], row2_y, bws[0], row2_h, fc=PINK, ec=PINK_E, r=0.07, lw=1.05, z=2)
    txt(fg, bx[0] + bws[0] / 2, row2_y + row2_h / 2 + 0.08, "Pre-registered SCM",
        size=6.4, weight="bold", color=NAVY, ha="center")
    txt(fg, bx[0] + bws[0] / 2, row2_y + row2_h / 2 - 0.10, "from pipeline architecture",
        size=5.0, color=MUTED, ha="center")

    arrow(bg, bx[0] + bws[0], row2_y + row2_h / 2, bx[1], row2_y + row2_h / 2)

    rounded(bg, bx[1], row2_y, bws[1], row2_h, fc=PURPLE, ec=PURPLE_E, r=0.07, lw=1.1, z=2)
    txt(fg, bx[1] + bws[1] / 2, row2_y + row2_h / 2 + 0.12, "Counterfactual module",
        size=6.1, weight="bold", color=NAVY, ha="center")
    txt(fg, bx[1] + bws[1] / 2, row2_y + row2_h / 2 - 0.04,
        r"Abduct · $\mathrm{do}(\cdot)$ · predict",
        size=5.1, color=INK, ha="center")
    txt(fg, bx[1] + bws[1] / 2, row2_y + row2_h / 2 - 0.18, "MuJoCo implements do",
        size=4.8, color=MUTED, ha="center")

    arrow(bg, bx[1] + bws[1], row2_y + row2_h / 2, bx[2], row2_y + row2_h / 2)

    add_img(fig, A["clean"], bx[2], row2_y, bws[2], row2_h, z=4, r=0.04)
    txt(fg, bx[2] + bws[2] / 2, row2_y + row2_h + 0.08,
        "Intervened view", size=5.3, weight="bold", color=PURPLE_E, ha="center")

    arrow(bg, bx[2] + bws[2], row2_y + row2_h / 2, bx[3], row2_y + row2_h / 2)

    rounded(bg, bx[3], row2_y, bws[3], row2_h, fc=GOLD, ec=GOLD_E, r=0.07, lw=1.05, z=2)
    txt(fg, bx[3] + bws[3] / 2, row2_y + row2_h / 2 + 0.08, "Diagnosis",
        size=6.6, weight="bold", color=NAVY, ha="center")
    txt(fg, bx[3] + bws[3] / 2, row2_y + row2_h / 2 - 0.10, "geometry or perception?",
        size=4.9, color=MUTED, ha="center")

    arrow(bg, bx[3] + bws[3] / 2, row2_y, bx[3] + bws[3] / 2, bot_y + 0.42)

    labels = [r"$\sigma_d$", r"$\phi$", r"$\theta$", "multi"]
    colors = ["#5B8FA8", "#C47A3A", "#5B9A6A", "#8A6BB5"]
    twt = 0.38
    tgap = 0.06
    row_w = 4 * twt + 3 * tgap
    tx0 = bx[3] + bws[3] / 2 - row_w / 2
    for i, (lab, col) in enumerate(zip(labels, colors)):
        rounded(bg, tx0 + i * (twt + tgap), bot_y + 0.22, twt, 0.18,
                fc=col, ec="none", r=0.045, z=8)
        txt(fg, tx0 + i * (twt + tgap) + twt / 2, bot_y + 0.31, lab,
            size=5.5, color=WHITE, weight="bold", ha="center", va="center")
    txt(fg, bx[3] + bws[3] / 2, bot_y + 0.12,
        "Diagnosis tokens  (checked vs ground truth)",
        size=5.2, color=MUTED, ha="center")

    out_pdf = FIG / "fig_teaser_diagnosis.pdf"
    out_png = FIG / "fig_teaser_diagnosis.png"
    fig.savefig(out_pdf, dpi=300, facecolor=WHITE, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_png, dpi=280, facecolor=WHITE, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    build()
