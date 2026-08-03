"""
make_equations_slide.py
========================
Generates a clean white-background figure showing:
  1. The pipeline flow (what each variable IS and WHERE it is measured)
  2. The structural equations written plainly
  3. Why sigma_d/rho cannot affect C_pc
  4. What each regression model means in plain English
Save: results/figures/equations_slide.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT   = Path(__file__).parent
FIGDIR = ROOT / "results" / "figures"
OUT    = FIGDIR / "equations_slide.png"

fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor="white")
ax.set_facecolor("white")
ax.set_xlim(0, 19.2)
ax.set_ylim(0, 10.8)
ax.axis("off")

# ── Colour palette ──────────────────────────────────────────────────────────
C_INPUT   = "#1a3c5e"   # dark navy  — input variables (causes)
C_PIPE    = "#2e7d32"   # dark green — pipeline steps
C_OUTPUT  = "#6a1e14"   # dark red   — outcome
C_NOISE   = "#b34700"   # orange     — where noise enters
C_CPc     = "#3949ab"   # indigo     — C_pc (special — measured before noise)
C_EQ      = "#1a1a1a"   # near-black — equation text
C_GREY    = "#757575"   # grey       — secondary text
C_BOX     = "#f5f5f5"   # light grey — equation box background
C_DIVIDER = "#dddddd"   # divider lines

def box(ax, x, y, w, h, text, col, fontsize=9.5, text_col="white",
        bold=False, subtext=None, subsize=7.5, radius=0.18):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.0,rounding_size={radius}",
                          facecolor=col, edgecolor="none", zorder=3)
    ax.add_patch(rect)
    ty = y + h/2 + (0.12 if subtext else 0)
    ax.text(x + w/2, ty, text,
            ha="center", va="center", fontsize=fontsize,
            color=text_col, fontweight="bold" if bold else "normal",
            zorder=4)
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.22, subtext,
                ha="center", va="center", fontsize=subsize,
                color=text_col, alpha=0.85, zorder=4, style="italic")

def arrow(ax, x1, y1, x2, y2, col="#555555", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=14),
                zorder=5)

def label(ax, x, y, text, col=C_EQ, size=9, bold=False, ha="left", va="top"):
    ax.text(x, y, text, color=col, fontsize=size,
            fontweight="bold" if bold else "normal",
            ha=ha, va=va)

# ════════════════════════════════════════════════════════════════════════════
# TOP HALF — Pipeline flow diagram
# ════════════════════════════════════════════════════════════════════════════

label(ax, 0.3, 10.55,
      "THE PIPELINE: what each variable is and where it is measured",
      col=C_INPUT, size=13, bold=True)

# ── Row 1: The four input variables ─────────────────────────────────────────
BW, BH = 2.9, 0.72
BY1 = 9.55

inputs = [
    (0.25, r"σ_d  (depth noise)", "adds Gaussian error\nto each depth pixel"),
    (3.35, r"ρ  (sparsity)", "randomly removes\npoints from cloud"),
    (6.45, r"φ  (camera elevation)", "angle camera tilts\nup/down"),
    (9.55, r"θ  (camera azimuth)", "angle camera rotates\nleft/right"),
]
for x, name, sub in inputs:
    box(ax, x, BY1, BW, BH, name, C_INPUT,
        fontsize=10, subtext=sub, subsize=7.5, bold=True)

label(ax, 0.25, BY1 - 0.07, "① These are the CAUSES you controlled in the experiment",
      col=C_GREY, size=8)

# ── Segmentation map step ───────────────────────────────────────────────────
BY_SEG = 7.90
box(ax, 6.45, BY_SEG, 5.95, 0.65,
    "Depth camera captures image → segmentation map",
    "#455a64", fontsize=9.5,
    subtext="(which pixels belong to the object — pure geometry, no noise yet)",
    subsize=7.8, text_col="white")

# arrows from phi, theta → segmentation
arrow(ax, 6.45+2.9/2, BY1, 6.45+2.9/2, BY_SEG+0.65, col=C_INPUT)
arrow(ax, 9.55+2.9/2, BY1, 9.55+2.9/2+0.5, BY_SEG+0.65, col=C_INPUT)

# ── C_pc box ────────────────────────────────────────────────────────────────
BY_CPC = 6.85
box(ax, 6.45, BY_CPC, 5.95, 0.72,
    "C_pc  (point cloud completeness)  — computed HERE",
    C_CPc, fontsize=10.5, bold=True,
    subtext="fraction of the scene the object occupies  |  range 0–1",
    subsize=8, text_col="white")

arrow(ax, 9.42, BY_SEG, 9.42, BY_CPC+0.72, col="#455a64")

# Big annotation: C_pc is computed BEFORE noise
ax.annotate("C_pc is computed\nBEFORE noise\nor downsampling\n→ σ_d and ρ\ncannot affect it",
            xy=(6.45, BY_CPC+0.36), xytext=(4.0, BY_CPC+0.5),
            fontsize=8, color=C_CPc, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=C_CPc, lw=1.4),
            ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_CPc, lw=1.2))

# ── Noise injection box ──────────────────────────────────────────────────────
BY_NOISE = 6.02
box(ax, 0.25, BY_NOISE, 5.95, 0.62,
    "NOISE INJECTED HERE: σ_d adds depth error, ρ removes points",
    C_NOISE, fontsize=9, bold=True,
    subtext="the point cloud fed into Contact-GraspNet is now degraded",
    subsize=7.8, text_col="white")

# arrows sigma_d, rho → noise box
arrow(ax, 0.25+2.9/2, BY1, 1.72, BY_NOISE+0.62, col=C_NOISE, lw=2.0)
arrow(ax, 3.35+2.9/2, BY1, 4.82, BY_NOISE+0.62, col=C_NOISE, lw=2.0)
# phi, theta also go to noise box (degraded point cloud)
arrow(ax, 6.45+1.0, BY_CPC, 6.20, BY_NOISE+0.62, col=C_INPUT, lw=1.4)

# ── Contact-GraspNet ────────────────────────────────────────────────────────
BY_CGN = 5.05
box(ax, 0.25, BY_CGN, 11.85, 0.70,
    "Contact-GraspNet (CGN) — proposes grasps from point cloud",
    "#37474f", fontsize=10, bold=True,
    subtext="outputs: n_grasps (how many proposals) and q_grasp (best score)",
    subsize=8, text_col="white")

arrow(ax, 3.22, BY_NOISE, 3.22, BY_CGN+0.70, col=C_NOISE, lw=2.0)

# ── n_grasps, q_grasp, e_pose ────────────────────────────────────────────────
BY_MED = 3.98
metrics = [
    (0.25, "n_grasps", "count of grasp proposals\n(0 = total pipeline collapse)", "#e65100"),
    (4.25, "q_grasp", "confidence of best grasp\n(0–1; higher = better)", "#1b5e20"),
    (8.25, "e_pose", "distance: proposed grasp\nvs true object centre (m)", "#4a148c"),
]
for x, name, sub, col in metrics:
    box(ax, x, BY_MED, 3.65, 0.72, name, col,
        fontsize=11, bold=True, subtext=sub, subsize=7.8)
    arrow(ax, x+1.82, BY_CGN, x+1.82, BY_MED+0.72, col=col)

# arrows n→q, q→e
arrow(ax, 0.25+3.65, BY_MED+0.36, 4.25, BY_MED+0.36, col="#e65100")
arrow(ax, 4.25+3.65, BY_MED+0.36, 8.25, BY_MED+0.36, col="#1b5e20")

# ── Success ──────────────────────────────────────────────────────────────────
BY_Y = 2.92
box(ax, 3.25, BY_Y, 5.65, 0.70,
    "Y = success  (1 if e_pose < 0.065 m, else 0)",
    C_OUTPUT, fontsize=11, bold=True,
    subtext="97.9% agreement between this threshold and the recorded label",
    subsize=8)
arrow(ax, 10.07, BY_MED, 8.90, BY_Y+0.70, col="#4a148c")

# ════════════════════════════════════════════════════════════════════════════
# BOTTOM HALF — equations + what the model types mean
# ════════════════════════════════════════════════════════════════════════════

# Divider
ax.plot([0.2, 19.0], [2.60, 2.60], color=C_DIVIDER, lw=1.0)

label(ax, 0.3, 2.52,
      "THE STRUCTURAL EQUATIONS — one per arrow — and why each model was chosen",
      col=C_INPUT, size=12, bold=True)

EQ_Y = 0.12
EQ_H = 2.15
EQ_GAP = 0.22

eqs = [
    {
        "x": 0.20, "w": 4.55,
        "title": "Eq 1  →  C_pc",
        "eq":    "C_pc  =  α₀ + α_φ·φ + α_θ·θ  + ε",
        "model": "OLS  (Ordinary Least Squares)",
        "why":   "C_pc is a continuous number (0–1).\nOLS = draw the best-fit line/plane\nthrough the data points.\nR² = 0.893  means the model explains\n89% of the variation in C_pc.\nThe remaining 11% is random scatter.",
        "result":"φ coefficient = −0.000259/°\nEvery extra degree of elevation\nreduces coverage by 0.026%",
    },
    {
        "x": 4.97, "w": 4.55,
        "title": "Eq 2A  →  has_grasps  (0 or 1)",
        "eq":    "log( P / 1−P )  =  β₀ + β_σ·σ_d + β_ρ·ρ + β_φ·φ + β_θ·θ",
        "model": "Logistic regression",
        "why":   "has_grasps is binary: either CGN\nfound at least one grasp (1)\nor it found nothing (0).\nYou can't draw a straight line\nthrough a 0/1 outcome.\nLogistic regression predicts a\nprobability between 0 and 1.\nAUC = 0.943 means: pick any two\ntrials, one with grasps and one\nwithout — model ranks them\ncorrectly 94.3% of the time.",
        "result":"σ_d coefficient = −169\n→ tiny noise increase\ncollapses grasp detection",
    },
    {
        "x": 9.74, "w": 4.55,
        "title": "Eq 2B  →  n_grasps  (count: 0,1,2,...)",
        "eq":    "log( E[n] )  =  γ₀ + γ_σ·σ_d + γ_ρ·ρ + γ_φ·φ + γ_θ·θ",
        "model": "Negative Binomial GLM",
        "why":   "n_grasps is a count, not continuous.\nPoisson regression assumes\nvariance = mean.\nHere variance >> mean\n(some trials get 0 grasps,\nothers get 200+) — overdispersed.\nNegative Binomial allows\nextra spread. Better fit.",
        "result":"ρ coefficient = +1.27\n→ denser cloud = more proposals\n(correct, expected sign)",
    },
    {
        "x": 14.51, "w": 4.55,
        "title": "Eq 3 & 4  →  q_grasp and e_pose",
        "eq":    "q_grasp = δ₀ + δ_σ·σ_d + δ_ρ·ρ + δ_φ·φ + δ_θ·θ + δ_n·log(n)\ne_pose  = ζ₀ + ζ_σ·σ_d + ζ_ρ·ρ + ζ_φ·φ + ζ_θ·θ + ζ_q·q_grasp",
        "model": "OLS  (both continuous outcomes)",
        "why":   "Both are continuous numbers,\nso OLS works.\nEq 4 is run TWICE:\n  • once without q_grasp\n    (total effect of each variable)\n  • once with q_grasp\n    (direct effect)\nThe difference = how much each\nvariable acts THROUGH grasp quality.\nσ_d: 10.9% mediated → acts mostly\n  directly (corrupts 3D positions)\nρ: 90.9% mediated → acts through\n  count → quality (fewer options\n  → worse best pick)",
        "result":"R² = 0.699 (Eq3)\nR² = 0.625 (Eq4)",
    },
]

for eq in eqs:
    x, w = eq["x"], eq["w"]
    # outer box
    rect = FancyBboxPatch((x, EQ_Y), w, EQ_H,
                          boxstyle="round,pad=0.0,rounding_size=0.12",
                          facecolor=C_BOX, edgecolor="#cccccc", lw=0.8, zorder=2)
    ax.add_patch(rect)
    # title
    ax.text(x+0.12, EQ_Y+EQ_H-0.08, eq["title"],
            fontsize=9.5, color=C_INPUT, fontweight="bold", va="top")
    # equation
    ax.text(x+0.12, EQ_Y+EQ_H-0.36, eq["eq"],
            fontsize=8, color="#c62828", va="top", family="monospace")
    # model name
    ax.text(x+0.12, EQ_Y+EQ_H-0.62, "Model: " + eq["model"],
            fontsize=7.8, color=C_PIPE, va="top", fontweight="bold")
    # why
    ax.text(x+0.12, EQ_Y+EQ_H-0.84, eq["why"],
            fontsize=7.2, color=C_EQ, va="top", linespacing=1.35)
    # result
    ax.text(x+0.12, EQ_Y+0.06, eq["result"],
            fontsize=7.5, color="#1a237e", va="bottom",
            fontweight="bold", linespacing=1.4)

# ── Title ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.988,
         "Structural Causal Model — Pipeline & Equations",
         ha="center", va="top", fontsize=17, fontweight="bold", color="#1a1a1a")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT}")
plt.close(fig)
