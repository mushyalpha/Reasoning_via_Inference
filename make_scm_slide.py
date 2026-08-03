"""
make_scm_slide.py
=================
Generates a single 16:9 composite slide figure for the SCM supervisor update.
Arranges the six existing SCM figures into a publication-quality panel layout.
Saves to: results/figures/scm_slide.png  (ready to paste into PowerPoint)

Layout:
  ┌─────────────────┬──────────────────────────────┐
  │                 │  Eq2A: has_grasps            │
  │   DAG           │  Calibration  AUC=0.943      │
  │   (causal       ├──────────────────────────────┤
  │    structure)   │  Coefficients across all Eqs │
  │                 │                              │
  ├─────────────────┴──────────────────────────────┤
  │  Heatmap σ_d×ρ  │  Heatmap φ×θ  │  Key numbers│
  └─────────────────┴───────────────┴──────────────┘
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.image import imread
import numpy as np

ROOT    = Path(__file__).parent
FIGDIR  = ROOT / "results" / "figures"
OUT     = FIGDIR / "scm_slide.png"

# ---------------------------------------------------------------------------
# Load existing figures
# ---------------------------------------------------------------------------
dag        = imread(FIGDIR / "scm_dag.png")
calibration= imread(FIGDIR / "scm_has_grasps_calibration.png")
coeffs     = imread(FIGDIR / "scm_coefficients.png")
hmap_sr    = imread(FIGDIR / "scm_heatmap_sigma_rho.png")
hmap_pt    = imread(FIGDIR / "scm_heatmap_phi_theta.png")
residuals  = imread(FIGDIR / "scm_binned_residuals.png")

# ---------------------------------------------------------------------------
# Slide canvas — 16:9, high-res
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(19.2, 10.8), facecolor="#1a1a2e")

# Typography
TITLE_COL  = "#e0e0ff"
LABEL_COL  = "#a0c4ff"
NUMBER_COL = "#ffd166"
BODY_COL   = "#c8d6e5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color":  TITLE_COL,
})

# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------
outer = gridspec.GridSpec(
    2, 1,
    figure=fig,
    height_ratios=[1.65, 1],
    hspace=0.04,
    top=0.91, bottom=0.03,
    left=0.01, right=0.99,
)

# Top row: 3 panels
top = gridspec.GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=outer[0],
    wspace=0.025,
    width_ratios=[1, 1, 1.05],
)

# Bottom row: 3 panels
bot = gridspec.GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=outer[1],
    wspace=0.025,
)

ax_dag   = fig.add_subplot(top[0])
ax_cal   = fig.add_subplot(top[1])
ax_coeff = fig.add_subplot(top[2])
ax_hsr   = fig.add_subplot(bot[0])
ax_hpt   = fig.add_subplot(bot[1])
ax_key   = fig.add_subplot(bot[2])

# ---------------------------------------------------------------------------
# Helper: show image in axis, no ticks
# ---------------------------------------------------------------------------
def show(ax, img, label, label_col=LABEL_COL, bg="#12122a"):
    ax.set_facecolor(bg)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a6a")
        spine.set_linewidth(0.8)
    ax.set_title(label, fontsize=10.5, color=label_col,
                 fontweight="bold", pad=5, loc="left")

show(ax_dag,   dag,         "① Causal graph (DAG)")
show(ax_cal,   calibration, "② Eq2A — grasp detection  (AUC = 0.943)")
show(ax_coeff, coeffs,      "③ Structural equation coefficients")
show(ax_hsr,   hmap_sr,     "④ Grasp count: σ_d × ρ interaction")
show(ax_hpt,   hmap_pt,     "⑤ Scene coverage: φ × θ interaction")

# ---------------------------------------------------------------------------
# Panel ⑥ — Key numbers summary (text panel, no image)
# ---------------------------------------------------------------------------
ax_key.set_facecolor("#0d1b2a")
ax_key.set_xticks([]); ax_key.set_yticks([])
for spine in ax_key.spines.values():
    spine.set_edgecolor("#3a3a6a"); spine.set_linewidth(0.8)
ax_key.set_title("⑥ Fit quality & mediation", fontsize=10.5,
                 color=LABEL_COL, fontweight="bold", pad=5, loc="left")

# Section: fit quality
lines = [
    ("Fit quality",          None,       "#7ecfff", 14.5),
    ("Eq1  C_pc ~ φ+θ",     "R² = 0.893",  NUMBER_COL, 11.5),
    ("Eq2A has_grasps",      "AUC = 0.943", NUMBER_COL, 11.5),
    ("Eq3  q_grasp",         "R² = 0.699",  NUMBER_COL, 11.5),
    ("Eq4  e_pose",          "R² = 0.625",  NUMBER_COL, 11.5),
    ("",                     None,       None,       6),
    ("Mediation via q_grasp",None,       "#7ecfff", 14.5),
    ("σ_d effect mediated",  "10.9%",    NUMBER_COL, 11.5),
    ("ρ effect mediated",    "90.9%",    NUMBER_COL, 11.5),
    ("",                     None,       None,       4),
    ("Implication",          None,       "#7ecfff", 12.5),
    ("σ_d acts directly on", None,       BODY_COL,  10),
    ("grasp geometry",       None,       BODY_COL,  10),
    ("ρ acts through grasp", None,       BODY_COL,  10),
    ("quality (q_grasp)",    None,       BODY_COL,  10),
]

y = 0.95
for label, value, col, size in lines:
    if col is None:
        y -= 0.04
        continue
    if value is None:
        ax_key.text(0.05, y, label, transform=ax_key.transAxes,
                    fontsize=size, color=col, fontweight="bold", va="top")
    else:
        ax_key.text(0.05, y, label, transform=ax_key.transAxes,
                    fontsize=size, color=BODY_COL, va="top")
        ax_key.text(0.95, y, value, transform=ax_key.transAxes,
                    fontsize=size, color=col, fontweight="bold",
                    va="top", ha="right")
    y -= size * 0.0072 + 0.025

# ---------------------------------------------------------------------------
# Slide title and subtitle
# ---------------------------------------------------------------------------
fig.text(0.5, 0.965,
         "Structural Causal Model — Fitting Results",
         ha="center", va="top",
         fontsize=20, fontweight="bold", color=TITLE_COL)

fig.text(0.5, 0.935,
         "432 trials  ·  4 structural equations  ·  OLS / Logistic / Negative Binomial",
         ha="center", va="top",
         fontsize=11.5, color=BODY_COL)

# Thin divider under title
fig.add_artist(plt.Line2D(
    [0.01, 0.99], [0.925, 0.925],
    transform=fig.transFigure,
    color="#3a3a6a", linewidth=0.9,
))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Slide saved: {OUT}")
plt.close(fig)
