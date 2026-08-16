#!/usr/bin/env python3
"""Compose a few extra thesis figures from existing renders + the v2 CSV."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

ROOT = Path("/Users/bonolomasima/Desktop/Reasoning_via_Inference")
FIG = ROOT / "results" / "figures"
OUT = FIG

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 11,
})


def make_viewing_sphere():
    """Camera on a sphere: φ elevation, θ azimuth, with the actual grid marked."""
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    r = 1.0
    phis_deg = [30, 45, 50, 55, 60, 65]
    thetas_deg = [0, 45, 90]

    # Sphere wire (upper hemisphere only)
    u = np.linspace(0, np.pi / 2, 24)
    v = np.linspace(0, np.pi / 2, 24)
    xs = r * np.outer(np.cos(v), np.cos(u))
    ys = r * np.outer(np.sin(v), np.cos(u))
    zs = r * np.outer(np.ones_like(v), np.sin(u))
    ax.plot_wireframe(xs, ys, zs, color="0.75", linewidth=0.4, alpha=0.7)

    # Table (thin box) and cylinder
    tw, td, th = 0.28, 0.22, 0.04
    ax.bar3d(-tw / 2, -td / 2, -th, tw, td, th, color="#8B6914", alpha=0.85, shade=True)
    cyl_r, cyl_h = 0.07, 0.16
    th_c = np.linspace(0, 2 * np.pi, 40)
    zc = np.linspace(0, cyl_h, 8)
    ax.plot(cyl_r * np.cos(th_c), cyl_r * np.sin(th_c), 0, color="#2a6db5", lw=1.4)
    ax.plot(cyl_r * np.cos(th_c), cyl_r * np.sin(th_c), cyl_h, color="#2a6db5", lw=1.4)
    for z in (0, cyl_h):
        pass
    ax.plot([cyl_r, cyl_r], [0, 0], [0, cyl_h], color="#2a6db5", lw=1.2)

    # Grid cameras
    for phi_d in phis_deg:
        for th_d in thetas_deg:
            phi = np.deg2rad(phi_d)
            th = np.deg2rad(th_d)
            x = r * np.cos(phi) * np.cos(th)
            y = r * np.cos(phi) * np.sin(th)
            z = r * np.sin(phi)
            ax.scatter([x], [y], [z], c="#1f4e79", s=22, depthshade=False, zorder=5)

    # Example camera at φ=55°, θ=45° (the clean-cell viewpoint)
    phi_ex, th_ex = np.deg2rad(55), np.deg2rad(45)
    xe = r * np.cos(phi_ex) * np.cos(th_ex)
    ye = r * np.cos(phi_ex) * np.sin(th_ex)
    ze = r * np.sin(phi_ex)
    ax.scatter([xe], [ye], [ze], c="#c0392b", s=70, marker="s", depthshade=False, zorder=6)
    ax.plot([0, xe], [0, ye], [cyl_h / 2, ze], "k--", lw=0.9, alpha=0.7)
    ax.plot([0, xe], [0, ye], [0, 0], "k:", lw=0.8, alpha=0.6)
    ax.text(xe + 0.05, ye + 0.05, ze + 0.08, r"camera $(\phi,\theta)$", color="#c0392b", fontsize=9)

    # Azimuth arc on XY
    ths = np.linspace(0, th_ex, 40)
    ax.plot(0.45 * np.cos(ths), 0.45 * np.sin(ths), 0, color="#c0392b", lw=2.0)
    ax.text(0.52 * np.cos(th_ex / 2), 0.52 * np.sin(th_ex / 2), 0.02,
            r"$\theta$  azimuth", color="#c0392b", fontsize=10)

    # Elevation arc in the plane of the camera
    phs = np.linspace(0, phi_ex, 40)
    ax.plot(
        0.45 * np.cos(phs) * np.cos(th_ex),
        0.45 * np.cos(phs) * np.sin(th_ex),
        0.45 * np.sin(phs),
        color="#1e8449", lw=2.0,
    )
    mid = phi_ex / 2
    ax.text(
        0.58 * np.cos(mid) * np.cos(th_ex),
        0.58 * np.cos(mid) * np.sin(th_ex),
        0.58 * np.sin(mid),
        r"$\phi$  elevation", color="#1e8449", fontsize=10,
    )

    # Level vs overhead callouts
    phi_lo, th0 = np.deg2rad(30), 0.0
    xl = r * np.cos(phi_lo) * np.cos(th0)
    yl = r * np.cos(phi_lo) * np.sin(th0)
    zl = r * np.sin(phi_lo)
    ax.text(xl + 0.08, yl - 0.15, zl - 0.02, "level\n($\\phi=30^\\circ$)", color="#1f4e79", fontsize=8)

    phi_hi = np.deg2rad(65)
    xh = r * np.cos(phi_hi) * np.cos(th0)
    yh = r * np.cos(phi_hi) * np.sin(th0)
    zh = r * np.sin(phi_hi)
    ax.text(xh - 0.02, yh + 0.18, zh + 0.12, "overhead\n($\\phi=65^\\circ$)",
            color="#1f4e79", fontsize=8, ha="center")

    ax.text(0.02, 0.02, cyl_h + 0.04, "object", fontsize=8, color="#2a6db5")
    ax.text(-0.22, -0.18, -th - 0.02, "table", fontsize=8, color="#5a3d0a")

    ax.set_xlim(-0.2, 1.15)
    ax.set_ylim(-0.2, 1.15)
    ax.set_zlim(-0.15, 1.15)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect((1.2, 1.2, 1.1))
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    # Inset schematics: what each elevation actually sees.

    def _cyl_side(ax_in):
        thc = np.linspace(0, 2 * np.pi, 80)
        ax_in.fill(0.35 * np.cos(thc), 0.55 + 0.55 * np.sin(thc) * 0.15,
                   color="#2a6db5", alpha=0.35)
        ax_in.plot([-0.35, -0.35], [0.0, 1.0], color="#2a6db5", lw=1.4)
        ax_in.plot([0.35, 0.35], [0.0, 1.0], color="#2a6db5", lw=1.4)
        ax_in.plot(0.35 * np.cos(thc), 1.0 + 0.12 * np.sin(thc), color="#2a6db5", lw=1.2)
        ax_in.plot(0.35 * np.cos(thc), 0.12 * np.sin(thc), color="#2a6db5", lw=1.2)
        ax_in.set_xlim(-1.1, 1.1)
        ax_in.set_ylim(-0.2, 1.35)
        ax_in.set_aspect("equal")
        ax_in.axis("off")
        ax_in.set_title("side-on ($\\phi=30^\\circ$):\nsees vertical surfaces",
                        fontsize=7, color="#1f4e79", pad=2)

    def _cyl_top(ax_in):
        thc = np.linspace(0, 2 * np.pi, 80)
        ax_in.fill(0.55 * np.cos(thc), 0.55 * np.sin(thc),
                   color="#2a6db5", alpha=0.35)
        ax_in.plot(0.55 * np.cos(thc), 0.55 * np.sin(thc), color="#2a6db5", lw=1.4)
        ax_in.plot(0.18 * np.cos(thc), 0.18 * np.sin(thc), color="#2a6db5", lw=0.8, ls="--")
        ax_in.set_xlim(-1.1, 1.1)
        ax_in.set_ylim(-1.1, 1.1)
        ax_in.set_aspect("equal")
        ax_in.axis("off")
        ax_in.set_title("overhead ($\\phi=65^\\circ$):\nsees top only",
                        fontsize=7, color="#1f4e79", pad=2)

    ax_lo = fig.add_axes([0.02, 0.08, 0.22, 0.28])
    _cyl_side(ax_lo)
    ax_hi = fig.add_axes([0.02, 0.42, 0.22, 0.28])
    _cyl_top(ax_hi)

    fig.savefig(OUT / "fig_viewing_sphere.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_viewing_sphere.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_viewing_sphere")


def _crop_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = max(0, (h - side) // 2 - int(0.04 * h))
    return im.crop((left, top, left + side, top + side))


def make_three_objects():
    paths = [
        (FIG / "pickup_demo" / "cylinder_1_approach.png", "Cylinder\n(curved, rotationally symmetric)"),
        (FIG / "pickup_demo" / "box_1_approach.png", "Sugar box\n(flat faces, thin short axis)"),
        (FIG / "pickup_demo" / "mustard_1_approach.png", "Mustard bottle\n(irregular, asymmetric)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), facecolor="white")
    for ax, (p, title) in zip(axes, paths):
        im = _crop_square(Image.open(p).convert("RGB"))
        ax.imshow(im)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
        ax.set_facecolor("white")
    fig.suptitle("Floating-gripper evaluation scene (arm omitted)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_three_objects_fg.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / "fig_three_objects_fg.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_three_objects_fg")


def make_depth_degradation_rgb():
    """Rebuild fig 4.2 on a white canvas, matching the fig 4.4 chrome."""
    panel_dir = FIG / "depth_degradation_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_paths = [panel_dir / f"sigma_{i}.png" for i in range(5)]

    if not all(p.exists() for p in panel_paths):
        src_candidates = [
            ROOT / "figures" / "fig_depth_degradation_rgb.png",
            FIG / "fig_depth_degradation_rgb.png",
        ]
        src = next((p for p in src_candidates if p.exists()), None)
        if src is None:
            print("  skip fig_depth_degradation_rgb (source missing)")
            return
        arr = np.array(Image.open(src).convert("RGB"))
        photo = arr[196:646]
        dark = photo.mean(axis=(0, 2)) < 20
        bounds = []
        in_run = False
        for i, is_dark in enumerate(dark):
            if (not is_dark) and not in_run:
                start = i
                in_run = True
            elif is_dark and in_run:
                if i - start > 40:
                    bounds.append((start, i))
                in_run = False
        if in_run and photo.shape[1] - start > 40:
            bounds.append((start, photo.shape[1]))
        if len(bounds) != 5:
            print(f"  skip fig_depth_degradation_rgb (found {len(bounds)} panels)")
            return
        min_w = min(b - a for a, b in bounds)
        for i, (a, b) in enumerate(bounds):
            crop = photo[:, a:b]
            extra = crop.shape[1] - min_w
            left = extra // 2
            Image.fromarray(crop[:, left:left + min_w]).save(panel_paths[i])

    crops = [np.array(Image.open(p).convert("RGB")) for p in panel_paths]
    labels = [
        r"Clean ($\sigma_d = 0$)",
        r"$\sigma_d = 0.005$",
        r"$\sigma_d = 0.01$",
        r"$\sigma_d = 0.02$",
        r"$\sigma_d = 0.04$",
    ]
    fig, axes = plt.subplots(1, 5, figsize=(11.4, 3.15), facecolor="white")
    for ax, crop, title in zip(axes, crops, labels):
        ax.imshow(crop)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
        ax.set_facecolor("white")
    fig.suptitle(
        "Perception degradation: additive Gaussian noise on the camera view",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()

    dests = [OUT, ROOT / "figures"]
    for dest in dests:
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / "fig_depth_degradation_rgb.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(dest / "fig_depth_degradation_rgb.pdf",
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote fig_depth_degradation_rgb")


def _filled_contour(ax, x, y, Z, levels, cmap, vmin, vmax):
    """Smooth filled contours over a discrete factor grid (index coordinates)."""
    from scipy.interpolate import RegularGridInterpolator

    yi = np.linspace(y[0], y[-1], 80)
    xi = np.linspace(x[0], x[-1], 80)
    interp = RegularGridInterpolator(
        (y, x), Z, method="linear", bounds_error=False, fill_value=None,
    )
    YY, XX = np.meshgrid(yi, xi, indexing="ij")
    Zi = interp(np.stack([YY, XX], axis=-1))
    cf = ax.contourf(
        xi, yi, Zi, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax,
        extend="max", antialiased=True,
    )
    ax.contour(
        xi, yi, Zi, levels=levels, colors="0.35", linewidths=0.35, alpha=0.65,
    )
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[-1], y[0])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return cf


def make_heatmap_by_object():
    df = pd.read_csv(ROOT / "results" / "experiment_results_v2.csv")
    sigma = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.04]
    phis = [30.0, 45.0, 50.0, 55.0, 60.0, 65.0]
    objects = [("cylinder", "Cylinder"), ("box", "Sugar box"), ("mustard", "Mustard bottle")]
    x = np.arange(len(phis), dtype=float)
    y = np.arange(len(sigma), dtype=float)
    levels = np.linspace(0, 30, 13)

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0), sharey=True)
    cf = None
    for ax, (key, title) in zip(axes, objects):
        sub = df[df["object"] == key]
        piv = (
            sub.groupby(["sigma_d", "phi"])["success"]
            .mean()
            .unstack("phi")
            .reindex(index=sigma, columns=phis)
        )
        Z = 100.0 * piv.values
        cf = _filled_contour(ax, x, y, Z, levels, "Blues", 0, 30)
        ax.set_xticklabels([f"{int(p)}" for p in phis])
        ax.set_xlabel(r"elevation $\phi$ (deg)")
        ax.set_title(title)
    axes[0].set_yticklabels([f"{s:g}" for s in sigma])
    axes[0].set_ylabel(r"depth noise $\sigma_d$ (m), top $=0$")
    cbar = fig.colorbar(cf, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Success rate")
    cbar.set_ticks([0, 10, 20, 30])
    cbar.set_ticklabels(["0%", "10%", "20%", "30%+"])
    fig.suptitle(
        r"Marginal success (\%) by object, depth noise, and elevation"
        "\n(7,560-trial floating-gripper grid, unfiltered top-1, shake)",
        fontsize=11, y=1.04,
    )
    fig.savefig(OUT / "fig_heatmap_success_by_object.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_heatmap_success_by_object.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_heatmap_success_by_object")

def make_grasp_representation():
    """2D schematic of CGN grasp representation (c, a, b, w, d), in the style of the original paper."""
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    # ── Object: a simple mug silhouette ──────────────────────
    # Body (rectangle with rounded bottom)
    from matplotlib.patches import FancyBboxPatch, Circle, Arc
    mug_x, mug_y, mug_w, mug_h = 3.0, 1.0, 2.0, 3.0
    mug = FancyBboxPatch((mug_x, mug_y), mug_w, mug_h,
                         boxstyle="round,pad=0.15", fc="#c0392b", ec="#7b241c",
                         lw=1.8, alpha=0.85, zorder=2)
    ax.add_patch(mug)
    # Handle
    handle = Arc((mug_x + mug_w, mug_y + mug_h * 0.5), 1.0, 1.6,
                 angle=0, theta1=-90, theta2=90, lw=2.5, color="#7b241c", zorder=1)
    ax.add_patch(handle)

    # ── Contact point c ──────────────────────────────────────
    cx, cy = mug_x, mug_y + mug_h * 0.55  # left surface of the mug
    ax.plot(cx, cy, "o", color="#1abc9c", ms=10, zorder=6, mec="black", mew=0.8)
    ax.annotate(r"$c$", (cx, cy), xytext=(cx - 0.55, cy + 0.45),
                fontsize=15, fontweight="bold", color="#117864",
                arrowprops=dict(arrowstyle="-", color="#117864", lw=0.8))

    # ── Approach vector a (blue, pointing into the surface) ──
    a_len = 1.8
    ax.annotate("", xy=(cx, cy), xytext=(cx - a_len, cy),
                arrowprops=dict(arrowstyle="-|>", color="#2471a3", lw=2.8,
                                mutation_scale=18))
    ax.text(cx - a_len * 0.55, cy + 0.25, r"$\vec{a}$",
            fontsize=15, color="#2471a3", fontweight="bold")

    # ── Baseline vector b (red, perpendicular to a) ──────────
    b_len = 1.5
    ax.annotate("", xy=(cx, cy + b_len), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=2.8,
                                mutation_scale=18))
    ax.text(cx + 0.15, cy + b_len * 0.55, r"$\vec{b}$",
            fontsize=15, color="#c0392b", fontweight="bold")

    # ── Gripper (schematic parallel-jaw) ─────────────────────
    grip_base_x = cx - 1.3   # base of gripper (away from object)
    grip_w = 2.2              # finger opening (width)
    finger_len = 1.0          # finger length toward object
    bar_lw = 5

    # Base bar (horizontal connecting the two fingers)
    top_y = cy + grip_w / 2
    bot_y = cy - grip_w / 2
    ax.plot([grip_base_x, grip_base_x], [bot_y, top_y],
            color="#2c3e50", lw=bar_lw, solid_capstyle="round", zorder=3)
    # Top finger
    ax.plot([grip_base_x, grip_base_x + finger_len], [top_y, top_y],
            color="#2c3e50", lw=bar_lw, solid_capstyle="round", zorder=3)
    # Bottom finger
    ax.plot([grip_base_x, grip_base_x + finger_len], [bot_y, bot_y],
            color="#2c3e50", lw=bar_lw, solid_capstyle="round", zorder=3)
    # Stem / wrist
    ax.plot([grip_base_x - 0.8, grip_base_x], [cy, cy],
            color="#2c3e50", lw=bar_lw - 1, solid_capstyle="round", zorder=3)

    # ── Label w (grasp width) ────────────────────────────────
    wx = grip_base_x + finger_len + 0.25
    ax.annotate("", xy=(wx, top_y), xytext=(wx, bot_y),
                arrowprops=dict(arrowstyle="<->", color="#d4ac0d", lw=1.6))
    ax.text(wx + 0.15, cy, r"$w$", fontsize=14, color="#d4ac0d",
            fontweight="bold", va="center")

    # ── Label d (distance from baseline to base frame) ───────
    dy = bot_y - 0.5
    ax.annotate("", xy=(cx, dy), xytext=(grip_base_x, dy),
                arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=1.6))
    ax.text((cx + grip_base_x) / 2, dy - 0.3, r"$d$", fontsize=14,
            color="#8e44ad", fontweight="bold", ha="center")

    # ── Formatting ───────────────────────────────────────────
    ax.set_xlim(-3.5, 6.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(OUT / "fig_grasp_representation.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_grasp_representation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_grasp_representation")


if __name__ == "__main__":
    make_grasp_representation()
    make_three_objects()
    make_depth_degradation_rgb()
    make_heatmap_by_object()
