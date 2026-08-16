"""
render_thesis_screenshots.py
============================
Generates thesis-quality MuJoCo screenshots for:
  1. All three isolated-object scenes  (cylinder, box, mustard)
     - overhead "perception camera" view at phi=45°
     - isometric "overview" view showing the arm + object together
  2. The clutter scene  (all three objects together)
     - same two viewpoints
  3. A point-cloud degradation strip  (clean → σ_d=0.04)
     using the cylinder scene as the reference object

All images are written to  results/figures/thesis_renders/
"""

import os, math, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "results", "figures", "thesis_renders")
os.makedirs(OUT, exist_ok=True)

SCENES = {
    "cylinder": os.path.join(ROOT, "generated_scenes", "scene_cylinder.xml"),
    "box":      os.path.join(ROOT, "generated_scenes", "scene_box.xml"),
    "mustard":  os.path.join(ROOT, "generated_scenes", "scene_mustard.xml"),
    "clutter":  os.path.join(ROOT, "generated_scenes", "scene_clutter.xml"),
}

LABELS = {
    "cylinder": "Cylinder\n(rotationally symmetric, curved surface)",
    "box":      "Box / Cuboid\n(flat faces, sharp edges)",
    "mustard":  "Mustard Bottle\n(irregular, asymmetric)",
    "clutter":  "Cluttered Scene\n(all three objects together)",
}

W, H = 1280, 720

# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

def load_scene(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    return model, data


def set_arm_home(model, data):
    """Place the Panda arm at a neutral home pose without any settling."""
    home_angles = [-0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    joint_names = [
        "panda_joint1","panda_joint2","panda_joint3","panda_joint4",
        "panda_joint5","panda_joint6","panda_joint7",
    ]
    for name, angle in zip(joint_names, home_angles):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    # open fingers
    for fname in ("panda_finger_joint1", "panda_finger_joint2"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = 0.04
    mujoco.mj_forward(model, data)


def settle(model, data, steps=400):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def place_overview_cam(model, data, renderer,
                       azimuth=135.0, elevation=-25.0,
                       distance=1.8, lookat=(0.5, 0.0, 0.45)):
    """Use a free 'overview' camera by temporarily moving the renderer camera."""
    renderer.update_scene(data, camera=mujoco.MjvCamera())
    cam = mujoco.MjvCamera()
    cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]  = lookat
    cam.distance   = distance
    cam.azimuth    = azimuth
    cam.elevation  = elevation
    renderer.update_scene(data, camera=cam)


def render_overview(model, data, azimuth=135, elevation=-25,
                    distance=1.8, lookat=(0.5, 0.0, 0.45)):
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        cam = mujoco.MjvCamera()
        cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]  = lookat
        cam.distance   = distance
        cam.azimuth    = float(azimuth)
        cam.elevation  = float(elevation)
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()


def render_perception_cam(model, data, phi_deg=45.0, dist=0.7):
    """Render through the named 'perception_camera' after moving it to phi."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "perception_camera")
    if cam_id < 0:
        return render_overview(model, data)
    # Place camera at azimuth=0, elevation=phi above target object
    phi  = math.radians(phi_deg)
    cx   = 0.5 - dist * math.cos(phi)
    cz   = 0.45 + dist * math.sin(phi)
    model.cam_pos[cam_id] = [cx, 0.0, cz]
    # Point it at the table surface (object centroid)
    dx, dz = 0.5 - cx, 0.45 - cz
    norm   = math.sqrt(dx**2 + dz**2)
    # quat: rotation from world-z to (dx,0,dz) direction — use MjvCamera instead
    with mujoco.Renderer(model, height=H, width=W) as renderer:
        cam = mujoco.MjvCamera()
        cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]  = [0.5, 0.0, 0.45]
        cam.distance   = dist
        cam.azimuth    = 0.0
        cam.elevation  = -(90.0 - phi_deg)  # mujoco elevation: 0=horizon, -90=top-down
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()


def add_noise(depth, sigma):
    if sigma <= 0:
        return depth
    noise = np.random.default_rng(42).normal(0, sigma, depth.shape)
    return np.clip(depth + noise, 0, None)


def render_depth_strip(model, data, sigmas=(0.0, 0.005, 0.01, 0.02, 0.04)):
    """Return list of (sigma, depth_image) tuples for thesis degradation figure."""
    frames = []
    phi_deg = 45.0
    dist    = 0.7
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.enable_depth_rendering()
        cam = mujoco.MjvCamera()
        cam.type       = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:]  = [0.5, 0.0, 0.45]
        cam.distance   = dist
        cam.azimuth    = 0.0
        cam.elevation  = -(90.0 - phi_deg)
        renderer.update_scene(data, camera=cam)
        clean_depth = renderer.render().copy()
    for sigma in sigmas:
        frames.append((sigma, add_noise(clean_depth, sigma)))
    return frames


# ──────────────────────────────────────────────────────────────────
# FIGURE 1: three isolated objects — 2-row, 3-col panel
# ──────────────────────────────────────────────────────────────────

def figure_isolated_objects():
    print("Rendering isolated-object panels...")
    objects = ["cylinder", "box", "mustard"]
    rgb_overview = {}
    rgb_percam   = {}

    for obj in objects:
        print(f"  loading {obj} ...")
        model, data = load_scene(SCENES[obj])
        set_arm_home(model, data)
        settle(model, data, steps=500)
        rgb_overview[obj] = render_overview(model, data, azimuth=130, elevation=-22,
                                            distance=1.6, lookat=(0.5, 0.0, 0.48))
        rgb_percam[obj]   = render_perception_cam(model, data, phi_deg=45.0, dist=0.7)
        print(f"  {obj} done.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor("#1a1a2e")

    row_labels = ["Overview (arm + object)", "Perception camera (φ = 45°)"]
    for row, (label, renders) in enumerate(zip(row_labels, [rgb_overview, rgb_percam])):
        for col, obj in enumerate(objects):
            ax = axes[row][col]
            ax.imshow(renders[obj])
            ax.axis("off")
            if row == 0:
                ax.set_title(LABELS[obj], color="white", fontsize=13,
                             fontweight="bold", pad=10)
        # row label on the left
        axes[row][0].set_ylabel(label, color="#aaaacc", fontsize=11,
                                 rotation=90, labelpad=10)
        axes[row][0].yaxis.label.set_visible(True)

    fig.suptitle("Experiment A — Three Isolated Object Geometries\n"
                 "(MuJoCo + Franka Panda, YCB meshes)",
                 color="white", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(pad=2.0)
    out = os.path.join(OUT, "fig_isolated_objects.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved {out}")
    return rgb_overview, rgb_percam


# ──────────────────────────────────────────────────────────────────
# FIGURE 2: clutter scene
# ──────────────────────────────────────────────────────────────────

def figure_clutter():
    print("Rendering clutter scene...")
    model, data = load_scene(SCENES["clutter"])
    set_arm_home(model, data)
    settle(model, data, steps=600)

    # Three viewpoints for the clutter scene
    views = [
        dict(azimuth=130, elevation=-20, distance=1.6, lookat=(0.5, 0.0, 0.46)),
        dict(azimuth=0,   elevation=-35, distance=1.4, lookat=(0.5, 0.0, 0.46)),
        dict(azimuth=90,  elevation=-25, distance=1.5, lookat=(0.5, 0.0, 0.46)),
    ]
    vtitles = ["Isometric view", "Front view", "Side view"]

    imgs = [render_overview(model, data, **v) for v in views]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#1a1a2e")
    for ax, img, title in zip(axes, imgs, vtitles):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")

    fig.suptitle("Experiment B — Cluttered Scene (all three objects together)\n"
                 "New causal failure mode: inter-object collision during degraded perception",
                 color="white", fontsize=15, fontweight="bold")
    fig.tight_layout(pad=2.0)
    out = os.path.join(OUT, "fig_clutter_scene.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved {out}")


# ──────────────────────────────────────────────────────────────────
# FIGURE 3: depth-noise degradation strip (cylinder reference)
# ──────────────────────────────────────────────────────────────────

def figure_depth_strip():
    print("Rendering depth degradation strip...")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })
    model, data = load_scene(SCENES["cylinder"])
    set_arm_home(model, data)
    settle(model, data, steps=400)

    sigmas = [0.0, 0.005, 0.01, 0.02, 0.04]
    frames = render_depth_strip(model, data, sigmas)

    fig, axes = plt.subplots(1, len(sigmas), figsize=(22, 5))
    fig.patch.set_facecolor("white")

    for ax, (sigma, depth) in zip(axes, frames):
        ax.set_facecolor("white")
        valid = depth[depth > 0]
        vmin  = float(valid.min()) if valid.size else 0
        vmax  = float(np.percentile(valid, 98)) if valid.size else 1
        ax.imshow(depth, cmap="plasma", vmin=vmin, vmax=vmax)
        ax.axis("off")
        label = "Clean\n(σ_d = 0)" if sigma == 0 else f"σ_d = {sigma}"
        ax.set_title(label, color="black", fontsize=11)

    fig.suptitle(
        "Depth-noise intervention on the cylinder ($\\phi=45^\\circ$)",
        color="black", fontsize=14,
    )
    fig.tight_layout(pad=1.5)
    out = os.path.join(OUT, "fig_depth_degradation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → saved {out}")


# ──────────────────────────────────────────────────────────────────
# FIGURE 4: experiment-loop diagram (matplotlib schematic)
# ──────────────────────────────────────────────────────────────────

def figure_experiment_loop():
    print("Drawing experiment loop schematic...")

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#0f0f1a")

    colours = {
        "A_header":  "#2d6a9f",
        "B_header":  "#6a2d9f",
        "step":      "#1e1e3a",
        "arrow":     "#aaaacc",
        "text":      "white",
        "subtext":   "#9999bb",
        "new":       "#f0a500",
        "highlight": "#2daf6a",
    }

    def draw_loop(ax, title, title_color, steps, width=0.82, start_y=0.93):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor("#0f0f1a")

        # title box
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (0.09, start_y - 0.04), width, 0.08,
            boxstyle="round,pad=0.01", linewidth=2,
            edgecolor=title_color, facecolor=title_color + "44",
            transform=ax.transAxes))
        ax.text(0.5, start_y, title, color="white",
                fontsize=13, fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)

        gap   = (start_y - 0.04 - 0.06) / max(len(steps), 1)
        box_h = min(gap * 0.72, 0.085)
        x0, x1 = 0.09, 0.09 + width

        for i, (label, sub, colour_key) in enumerate(steps):
            cy = start_y - 0.04 - 0.03 - gap * (i + 0.5)
            bcolour = colours.get(colour_key, colours["step"])
            ecolour = "#f0a500" if colour_key == "new" else "#3a3a6a"
            lw      = 2.5 if colour_key == "new" else 1.2
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (x0, cy - box_h / 2), x1 - x0, box_h,
                boxstyle="round,pad=0.01", linewidth=lw,
                edgecolor=ecolour, facecolor=bcolour + "cc",
                transform=ax.transAxes))
            ax.text(0.5, cy + box_h * 0.15, label,
                    color=colours["text"], fontsize=10, fontweight="bold",
                    ha="center", va="center", transform=ax.transAxes)
            if sub:
                ax.text(0.5, cy - box_h * 0.22, sub,
                        color=colours["subtext"], fontsize=8.5,
                        ha="center", va="center", transform=ax.transAxes,
                        style="italic")
            # arrow down
            if i < len(steps) - 1:
                arrow_y_top = cy - box_h / 2
                arrow_y_bot = arrow_y_top - (gap - box_h) * 0.9
                ax.annotate("", xy=(0.5, arrow_y_bot), xytext=(0.5, arrow_y_top),
                            xycoords="axes fraction", textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=colours["arrow"],
                                            lw=1.8))

    # ── Experiment A ──────────────────────────────────────────────
    exp_a_steps = [
        ("For each object  ∈  {cylinder, box, mustard}",
         "",                                               "A_header"),
        ("For each (σ_d, ρ, φ, θ, seed) in expanded grid",
         "7 × 2 × 6 × 1 × 5  =  7 560 trials",           "step"),
        ("Load MuJoCo scene  +  settle physics",
         "arm parks at safe position; object freejoint settles", "step"),
        ("Place perception camera at (φ, θ)",
         "",                                               "step"),
        ("Render depth image  →  add Gaussian noise σ_d",
         "mujoco.Renderer offscreen, CGL/EGL backend",    "step"),
        ("Downsample point cloud to density ρ",
         "deterministic_downsample_idx() — same subset every seed", "highlight"),
        ("Seed np.random + torch RNG  →  run CGN",
         "bit-exact inference: set_num_threads(1), cuDNN det.", "highlight"),
        ("Select best grasp  →  IK move to pre-grasp",
         "",                                               "step"),
        ("Descend → close fingers → check proximity",
         "success = gripper centre within GRASP_RADIUS of object centroid", "step"),
        ("Append row to experiment_results_v2.csv",
         "object, σ_d, ρ, φ, θ, seed, q_grasp, success, …", "step"),
    ]
    draw_loop(axes[0], "Experiment A — Isolated Objects  (×3 objects)",
              colours["A_header"], exp_a_steps)

    # ── Experiment B ──────────────────────────────────────────────
    exp_b_steps = [
        ("For each target_object  ∈  {cylinder, box, mustard}",
         "other two objects remain as distractors",         "B_header"),
        ("For each (σ_d, ρ, φ, seed) in targeted clutter grid",
         "4 × 2 × 3 × 3  ×  3 objects  =  504 trials",    "step"),
        ("Load clutter MuJoCo scene  +  settle all objects",
         "triangular layout, 0.11 m spacing",              "step"),
        ("Place perception camera  →  render cluttered depth",
         "all three objects visible; overlapping boundaries", "step"),
        ("Add Gaussian noise σ_d  →  segment by instance ID",
         "target gets label 1, distractors get label 2/3",  "step"),
        ("Downsample + seed CGN  →  run on segmented cloud",
         "same determinism as Experiment A",               "step"),
        ("Execute grasp  →  proximity check on TARGET only",
         "",                                               "step"),
        ("Check finger contacts for non-target collision",
         "MuJoCo contact array: finger geom ↔ distractor geom", "new"),
        ("Append row with collision_with_neighbor flag",
         "NEW outcome variable — the Experiment B signal",  "new"),
    ]
    draw_loop(axes[1], "Experiment B — Cluttered Scene  (inter-object collisions)",
              colours["B_header"], exp_b_steps)

    fig.suptitle("Experimental Pipeline — Reasoning via Inference (MSc Thesis)",
                 color="white", fontsize=16, fontweight="bold", y=1.01)

    legend_elements = [
        matplotlib.patches.Patch(facecolor=colours["highlight"] + "cc",
                                  edgecolor="#3a3a6a", label="New determinism fix"),
        matplotlib.patches.Patch(facecolor=colours["step"] + "cc",
                                  edgecolor="#3a3a6a", label="Standard pipeline step"),
        matplotlib.patches.Patch(facecolor=colours["B_header"] + "44",
                                  edgecolor="#f0a500", lw=2.5,
                                  label="New — Experiment B only"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               framealpha=0.3, labelcolor="white", fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=2.5)
    out = os.path.join(OUT, "fig_experiment_loop.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → saved {out}")


# ──────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== MuJoCo thesis screenshot renderer ===\n")

    # The loop schematic never needs MuJoCo
    figure_experiment_loop()

    # MuJoCo renders
    try:
        figure_isolated_objects()
        figure_clutter()
        figure_depth_strip()
    except Exception as exc:
        print(f"\n[WARN] MuJoCo rendering failed: {exc}")
        print("The experiment-loop schematic (fig_experiment_loop.png) was saved.")
        print("For the MuJoCo renders, run this script on the RunPod instance with")
        print("  MUJOCO_GL=egl python render_thesis_screenshots.py")
        sys.exit(1)

    print("\n=== Done. All figures written to results/figures/thesis_renders/ ===")
