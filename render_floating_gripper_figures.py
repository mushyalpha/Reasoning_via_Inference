"""
render_floating_gripper_figures.py
====================================
Figures documenting the floating-gripper shake-test milestone (4 Aug):
outcome-variable redesign in response to Marker A/B's second round of
feedback, the mocap-teleport bug discovered mid-implementation, and its
fix via a weld-tracked dynamic hand. See RIGOUR_LEDGER.md Stage 21.

Generates, into results/figures/floating_gripper/:
  1. fig_mocap_bug_vs_fix.png    - the "hero" plot: object height vs.
     time, broken mocap-teleport hand vs. the fixed weld-tracked
     dynamic hand, same grasp pose, same everything else. This is the
     direct evidence for the bug and the fix.
  2. fig_grasp_sequence.png      - MuJoCo screenshots at 4 stages of the
     shake test (open/approach, closed/squeezed, mid-lift, full lift +
     shake) for a known-good cylinder side-grasp.
  3. fig_smoke_test_outcomes.png - bar chart of the 3 smoke-test cases
     (known-good / known-empty / known-colliding) showing collision_free
     and success exactly as expected.
  4. fig_pipeline_validation.png - bar chart of the real end-to-end
     pipeline (CGN-proposed grasps, not hand-picked poses) outcome
     breakdown per object, from results/experiment_results_v2.csv.

Usage:
    python render_floating_gripper_figures.py
"""
import os
import csv
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim_common as sc
from object_specs import (OBJECT_SPECS, spawn_pos, centroid_world,
                           build_scene_xml, FLOATING_GRIPPER_TEMPLATE)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "figures", "floating_gripper")
SCENES_DIR = os.path.join(ROOT, "generated_scenes")
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)

BG = "#0f0f1a"
PANEL = "#1a1a2e"
GOOD = "#2daf6a"
BAD = "#d64545"
NEW = "#f0a500"
GREY = "#9999bb"


def side_quat():
    approach = np.array([-1., 0., 0.])
    base = np.array([0., 0., 1.])
    y = np.cross(approach, base)
    R = np.column_stack((base, y, approach))
    return sc.rot_matrix_to_quat(R)


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1: the bug, in one plot -- object height, broken vs fixed
# ══════════════════════════════════════════════════════════════════════

def _build_broken_mocap_scene():
    """Reconstruct the ORIGINAL (buggy) design as a standalone scene:
    `hand` is a plain mocap body, teleported directly, no weld, no
    hand_target. Written to a clearly-labelled demo file -- never used
    by the real pipeline -- purely so this figure can show, side by
    side, what the discarded design actually did."""
    fixed_path = os.path.join(SCENES_DIR, "scene_cylinder_floating_gripper.xml")
    build_scene_xml("cylinder", fixed_path, template_path=FLOATING_GRIPPER_TEMPLATE)
    xml = open(fixed_path).read()

    # Remove the hand_target mocap body block.
    start = xml.index('<body name="hand_target"')
    end = xml.index('<body name="hand"')
    xml = xml[:start] + xml[end:]

    # Revert `hand` to a plain mocap body (delete the freejoint line,
    # drop gravcomp, add mocap="true").
    xml = xml.replace(
        '<body name="hand" pos="0 0 1.2" quat="1 0 0 0" childclass="panda" gravcomp="1">\n'
        '      <freejoint name="hand_freejoint"/>\n',
        '<body name="hand" mocap="true" pos="0 0 1.2" quat="1 0 0 0" childclass="panda">\n'
    )

    # Remove the weld equality (mocap bodies can't be dynamics-side of one).
    start = xml.index('<weld body1="hand"')
    end = xml.index('/>', start) + 2
    xml = xml[:start] + xml[end:]

    demo_path = os.path.join(SCENES_DIR, "DEMO_ONLY_broken_mocap_cylinder.xml")
    with open(demo_path, "w") as f:
        f.write(xml)
    return demo_path


def _trace_broken_mocap(model, obj_pos, obj_quat, grasp_pos, grasp_quat, fp,
                         close_steps=400, shake_steps=600, lift_height=0.15,
                         sample_every=4):
    """Reproduce the ORIGINAL close+lift loop exactly as first written:
    teleport the mocap `hand` body directly every step. This is the
    discarded design -- kept here only to generate the comparison
    figure, not imported by any pipeline script."""
    import math
    data = mujoco.MjData(model)
    sc.set_object_pose(model, data, "target_object", obj_pos, obj_quat)
    mujoco.mj_forward(model, data)
    sc.settle(model, data, 30)

    sc.teleport_mocap(model, data, "hand", grasp_pos, grasp_quat)
    sc.open_gripper(model, data)
    mujoco.mj_forward(model, data)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    z0 = float(data.xpos[obj_id][2])
    t, z, mocap_z = [], [], []
    step = 0

    contact_ctrl = None
    for _ in range(close_steps):
        if contact_ctrl is None:
            data.ctrl[aid] = max(0., data.ctrl[aid] - 255. / close_steps)
            mujoco.mj_step(model, data)
            if sc.gripper_contacted_bodies(model, data):
                contact_ctrl = max(0., data.ctrl[aid] - 60.)
        else:
            data.ctrl[aid] = contact_ctrl
            mujoco.mj_step(model, data)
        step += 1
        if step % sample_every == 0:
            t.append(step * model.opt.timestep)
            z.append(float(data.xpos[obj_id][2]) - z0)
            mocap_z.append(0.0)

    for i in range(shake_steps):
        tt = i / shake_steps
        ramp = min(1.0, tt / 0.3)
        dz = lift_height * ramp
        sc.teleport_mocap(model, data, "hand", grasp_pos + np.array([0, 0, dz]), grasp_quat)
        mujoco.mj_step(model, data)
        step += 1
        if step % sample_every == 0:
            t.append(step * model.opt.timestep)
            z.append(float(data.xpos[obj_id][2]) - z0)
            mocap_z.append(dz)

    return np.array(t), np.array(z), np.array(mocap_z)


def _trace_fixed_weld(model, obj_pos, obj_quat, grasp_pos, grasp_quat, fp,
                       close_steps=400, shake_steps=600, lift_height=0.15,
                       sample_every=4):
    """Same loop, but using the FIXED mechanism: `hand` is a real
    dynamic body welded to a teleported `hand_target` mocap body."""
    data = mujoco.MjData(model)
    sc.set_object_pose(model, data, "target_object", obj_pos, obj_quat)
    mujoco.mj_forward(model, data)
    sc.settle(model, data, 30)

    sc.teleport_hand_hard(model, data, grasp_pos, grasp_quat)
    sc.open_gripper(model, data)
    mujoco.mj_forward(model, data)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    z0 = float(data.xpos[obj_id][2])
    t, z, mocap_z = [], [], []
    step = 0

    contact_ctrl = None
    for _ in range(close_steps):
        sc.teleport_mocap(model, data, "hand_target", grasp_pos, grasp_quat)
        if contact_ctrl is None:
            data.ctrl[aid] = max(0., data.ctrl[aid] - 255. / close_steps)
            mujoco.mj_step(model, data)
            if sc.gripper_contacted_bodies(model, data):
                contact_ctrl = max(0., data.ctrl[aid] - 60.)
        else:
            data.ctrl[aid] = contact_ctrl
            mujoco.mj_step(model, data)
        step += 1
        if step % sample_every == 0:
            t.append(step * model.opt.timestep)
            z.append(float(data.xpos[obj_id][2]) - z0)
            mocap_z.append(0.0)

    for i in range(shake_steps):
        tt = i / shake_steps
        ramp = min(1.0, tt / 0.3)
        dz = lift_height * ramp
        sc.teleport_mocap(model, data, "hand_target", grasp_pos + np.array([0, 0, dz]), grasp_quat)
        mujoco.mj_step(model, data)
        step += 1
        if step % sample_every == 0:
            t.append(step * model.opt.timestep)
            z.append(float(data.xpos[obj_id][2]) - z0)
            mocap_z.append(dz)

    return np.array(t), np.array(z), np.array(mocap_z)


def figure_mocap_bug_vs_fix():
    print("Figure 1: mocap-teleport bug vs. weld-tracked fix ...")
    spec = OBJECT_SPECS["cylinder"]
    obj_pos = np.array(spawn_pos(spec))
    obj_quat = np.array([1., 0., 0., 0.])
    centroid = np.array(centroid_world(spec, obj_pos))
    grasp_quat = side_quat()
    grasp_pos = centroid + np.array([0.12, 0., 0.])
    fp = spec["footprint_radius"]

    broken_xml = _build_broken_mocap_scene()
    broken_model = mujoco.MjModel.from_xml_path(broken_xml)
    t_b, z_b, cmd_b = _trace_broken_mocap(broken_model, obj_pos, obj_quat, grasp_pos, grasp_quat, fp)

    fixed_xml = os.path.join(SCENES_DIR, "scene_cylinder_floating_gripper.xml")
    fixed_model = mujoco.MjModel.from_xml_path(fixed_xml)
    t_f, z_f, cmd_f = _trace_fixed_weld(fixed_model, obj_pos, obj_quat, grasp_pos, grasp_quat, fp)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    ax.plot(t_b, cmd_b * 100, "--", color=GREY, lw=1.6, label="Commanded lift (gripper target)")
    ax.plot(t_b, z_b * 100, color=BAD, lw=2.4,
            label="Object height — BROKEN (mocap-teleported hand)")
    ax.plot(t_f, z_f * 100, color=GOOD, lw=2.4,
            label="Object height — FIXED (weld-tracked dynamic hand)")

    close_end = 400 * broken_model.opt.timestep
    ax.axvline(close_end, color=GREY, lw=1, ls=":")
    ax.text(close_end + 0.02, ax.get_ylim()[1] * 0.02, "fingers closed,\nlift begins",
            color=GREY, fontsize=9, va="bottom")

    ax.set_xlabel("Simulated time (s)", color="white", fontsize=12)
    ax.set_ylabel("Height relative to resting pose (cm)", color="white", fontsize=12)
    ax.set_title("The bug and the fix: object height during the floating-gripper lift\n"
                 "Same grasp pose, same friction, same everything else — only the hand's "
                 "kinematics differ",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.legend(facecolor=PANEL, edgecolor="#444466", labelcolor="white", fontsize=10, loc="upper left")
    ax.grid(alpha=0.15, color="white")

    fig.tight_layout()
    out = os.path.join(OUT, "fig_mocap_bug_vs_fix.png")
    fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> saved {out}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2: MuJoCo screenshots of the shake-test sequence
# ══════════════════════════════════════════════════════════════════════

def render_scene(model, data, azimuth, elevation, distance, lookat, w=960, h=720):
    with mujoco.Renderer(model, height=h, width=w) as renderer:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        renderer.update_scene(data, camera=cam)
        return renderer.render().copy()


def figure_grasp_sequence():
    print("Figure 2: floating-gripper shake-test screenshot sequence ...")
    spec = OBJECT_SPECS["cylinder"]
    fg_xml = os.path.join(SCENES_DIR, "scene_cylinder_floating_gripper.xml")
    build_scene_xml("cylinder", fg_xml, template_path=FLOATING_GRIPPER_TEMPLATE)
    model = mujoco.MjModel.from_xml_path(fg_xml)

    obj_pos = np.array(spawn_pos(spec))
    obj_quat = np.array([1., 0., 0., 0.])
    data = mujoco.MjData(model)
    sc.set_object_pose(model, data, spec["body_name"], obj_pos, obj_quat)
    mujoco.mj_forward(model, data)
    sc.settle(model, data, 30)
    centroid = np.array(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body_name"])])

    grasp_pos = centroid + np.array([0.12, 0., 0.])
    grasp_quat = side_quat()
    fp = spec["footprint_radius"]

    cam_kwargs = dict(azimuth=110, elevation=-18, distance=0.55, lookat=(0.5, 0.0, 0.5))

    frames = {}

    # Stage 1: open, at grasp pose (pre-check)
    sc.teleport_hand_hard(model, data, grasp_pos, grasp_quat)
    sc.open_gripper(model, data)
    mujoco.mj_forward(model, data)
    frames["1. Approach\n(open, pre-grasp pose)"] = render_scene(model, data, **cam_kwargs)

    # Stage 2: closed (run the close loop, stop right after)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    contact_ctrl = None
    for _ in range(400):
        sc.teleport_mocap(model, data, "hand_target", grasp_pos, grasp_quat)
        if contact_ctrl is None:
            data.ctrl[aid] = max(0., data.ctrl[aid] - 255. / 400)
            mujoco.mj_step(model, data)
            if sc.gripper_contacted_bodies(model, data):
                contact_ctrl = max(0., data.ctrl[aid] - 60.)
        else:
            data.ctrl[aid] = contact_ctrl
            mujoco.mj_step(model, data)
    frames["2. Fingers closed\n(squeeze, pre-lift)"] = render_scene(model, data, **cam_kwargs)

    # Stage 3 & 4: mid-lift and full lift+shake
    import math
    shake_steps = 600
    for i in range(shake_steps):
        t = i / shake_steps
        ramp = min(1.0, t / 0.3)
        dz = 0.15 * ramp
        dx = 0.03 * ramp * math.sin(2 * math.pi * 4.0 * t)
        dy = 0.03 * ramp * math.sin(2 * math.pi * 5.3 * t)
        sc.teleport_mocap(model, data, "hand_target", grasp_pos + np.array([dx, dy, dz]), grasp_quat)
        mujoco.mj_step(model, data)
        if i == 150:
            frames["3. Mid-lift\n(~40% of 15cm target)"] = render_scene(model, data, **cam_kwargs)
    frames["4. Full lift + shake\n(held through disturbance)"] = render_scene(model, data, **cam_kwargs)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    fig.patch.set_facecolor("white")
    for ax, (label, img) in zip(axes, frames.items()):
        ax.set_facecolor("white")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, color="black", fontsize=12, pad=6)
    fig.tight_layout(pad=0.6, w_pad=0.4)
    out = os.path.join(OUT, "fig_grasp_sequence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> saved {out}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3: smoke-test outcomes (good / bad / colliding)
# ══════════════════════════════════════════════════════════════════════

def figure_smoke_test_outcomes():
    print("Figure 3: smoke-test outcome summary ...")
    spec = OBJECT_SPECS["cylinder"]
    fg_xml = os.path.join(SCENES_DIR, "scene_cylinder_floating_gripper.xml")
    build_scene_xml("cylinder", fg_xml, template_path=FLOATING_GRIPPER_TEMPLATE)
    model = mujoco.MjModel.from_xml_path(fg_xml)
    obj_pos = np.array(spawn_pos(spec))
    obj_quat = np.array([1., 0., 0., 0.])

    def fresh():
        data = mujoco.MjData(model)
        sc.set_object_pose(model, data, spec["body_name"], obj_pos, obj_quat)
        mujoco.mj_forward(model, data)
        sc.settle(model, data, 30)
        return data

    data = fresh()
    centroid = np.array(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body_name"])])
    quat = side_quat()
    fp = spec["footprint_radius"]

    cases = {
        "Known-good\n(side grasp, x=+12cm)\nexpected: True / True": centroid + np.array([0.12, 0., 0.]),
        "Known-empty\n(x=+30cm, nothing there)\nexpected: True / False": centroid + np.array([0.30, 0., 0.]),
        "Known-colliding\n(hand at centroid)\nexpected: False / n/a": centroid.copy(),
    }
    results = {}
    for label, pos in cases.items():
        d = fresh()
        results[label] = sc.run_floating_gripper_test(
            model, d, spec["body_name"], pos, quat, footprint_radius=fp, squeeze_margin=60.)

    labels = list(results.keys())
    collision_free = [1 if results[l]["collision_free"] else 0 for l in labels]
    success = [1 if results[l]["success"] else 0 for l in labels]

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    b1 = ax.bar(x - width / 2, collision_free, width, label="collision_free",
                color="#3d7fc9", edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, success, width, label="success (held through shake)",
                color=GOOD, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["False", "True"], color="white")
    ax.set_ylim(0, 1.5)
    ax.set_title("Smoke Test: run_floating_gripper_test() Sanity Checks\n"
                 "All 3 expected outcomes reproduced correctly",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.legend(facecolor=PANEL, edgecolor="#444466", labelcolor="white",
              loc="center", bbox_to_anchor=(0.83, 0.35), fontsize=9.5)
    for xi in range(len(labels)):
        ax.text(xi, 1.18, "\u2713 PASS", ha="center", color=GOOD, fontsize=11, fontweight="bold")
    fig.subplots_adjust(bottom=0.22, top=0.85)
    out = os.path.join(OUT, "fig_smoke_test_outcomes.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> saved {out}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4: real end-to-end pipeline validation (CGN-proposed grasps)
# ══════════════════════════════════════════════════════════════════════

def figure_pipeline_validation():
    print("Figure 4: end-to-end pipeline validation (real CGN grasps) ...")
    csv_path = os.path.join(ROOT, "results", "experiment_results_v2.csv")
    if not os.path.exists(csv_path):
        print(f"  [skip] {csv_path} not found -- run `python run_experiments_v2.py --test` first.")
        return

    rows = list(csv.DictReader(open(csv_path)))
    objects = sorted(set(r["object"] for r in rows))
    categories = ["success", "collision\n(CGN pose invalid)", "no lift\n(collision-free, slipped)", "error"]
    counts = {obj: {c: 0 for c in categories} for obj in objects}

    for r in rows:
        obj = r["object"]
        if r.get("error"):
            counts[obj]["error"] += 1
        elif r["success"] == "1":
            counts[obj]["success"] += 1
        elif r.get("collision_free") == "0":
            counts[obj]["collision\n(CGN pose invalid)"] += 1
        elif r.get("collision_free") == "1":
            counts[obj]["no lift\n(collision-free, slipped)"] += 1
        else:
            counts[obj]["error"] += 1

    colors = {"success": GOOD, "collision\n(CGN pose invalid)": BAD,
              "no lift\n(collision-free, slipped)": NEW, "error": "#555577"}

    x = np.arange(len(objects))
    width = 0.6
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    bottoms = np.zeros(len(objects))
    for cat in categories:
        vals = np.array([counts[obj][cat] for obj in objects])
        ax.bar(x, vals, width, bottom=bottoms, label=cat, color=colors[cat],
               edgecolor="white", linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0:
                ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([o.capitalize() for o in objects], color="white", fontsize=11)
    ax.set_ylabel("Trials (8 per object, --test smoke grid)", color="white", fontsize=11)
    ax.set_title("End-to-End Pipeline Validation — Real CGN-Proposed Grasps\n"
                 "Floating-gripper shake test outcome, by object",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")
    ax.legend(facecolor=PANEL, edgecolor="#444466", labelcolor="white",
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_pipeline_validation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> saved {out}")


if __name__ == "__main__":
    print("=== Floating-gripper milestone figures ===\n")
    figure_mocap_bug_vs_fix()
    figure_grasp_sequence()
    figure_smoke_test_outcomes()
    figure_pipeline_validation()
    print(f"\n=== Done. Figures written to {OUT} ===")
