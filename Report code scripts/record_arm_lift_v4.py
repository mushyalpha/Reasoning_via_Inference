"""
record_arm_lift_v4.py
=====================
Thesis media for the full-arm lift proof (friction mid-lift + weld latch).

Writes to results/figures/arm_lift_v4/:
  - MP4 videos of the full approach → close → lift sequence
  - Key-stage PNG screenshots (side + 3/4 views)

Usage:
  python3 record_arm_lift_v4.py                  # friction + weld, cylinder_thin
  python3 record_arm_lift_v4.py --mode friction
  python3 record_arm_lift_v4.py --mode weld
  python3 record_arm_lift_v4.py --object cylinder_thinner
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco

try:
    import imageio
except ImportError as e:
    raise SystemExit("imageio required: pip install imageio imageio-ffmpeg") from e

_PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT)

import sim_common as sc
import sim_arm_v4 as arm
import object_specs_v4_patch  # noqa: F401 — registers thin cylinders
from object_specs import OBJECT_SPECS, build_scene_xml, spawn_pos, centroid_world

OUT_DIR = os.path.join(_PROJECT, "results", "figures", "arm_lift_v4")
SCENES = os.path.join(_PROJECT, "generated_scenes")


class ArmLiftRecorder:
    """Offscreen MuJoCo capture: video buffer + named PNG stills."""

    def __init__(self, model, out_stem, fps=24, width=1280, height=720,
                 azimuth=135., elevation=-22., distance=1.05):
        self.out_stem = out_stem
        self.fps = fps
        self.frames = []
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._cam)
        self._cam.distance = distance
        self._cam.azimuth = azimuth
        self._cam.elevation = elevation
        self.stills = {}

    def set_lookat(self, pos):
        self._cam.lookat[:] = np.asarray(pos, dtype=float)

    def capture(self, data):
        self._renderer.update_scene(data, camera=self._cam)
        self.frames.append(self._renderer.render().copy())

    def snapshot(self, data, name, azimuth=None, elevation=None, distance=None):
        """Save a named PNG; temporarily override camera if requested."""
        old = (self._cam.azimuth, self._cam.elevation, self._cam.distance)
        if azimuth is not None:
            self._cam.azimuth = azimuth
        if elevation is not None:
            self._cam.elevation = elevation
        if distance is not None:
            self._cam.distance = distance
        self._renderer.update_scene(data, camera=self._cam)
        img = self._renderer.render().copy()
        path = f"{self.out_stem}_{name}.png"
        imageio.imwrite(path, img)
        self.stills[name] = path
        print(f"  [still] {path}")
        self._cam.azimuth, self._cam.elevation, self._cam.distance = old
        return path

    def save_video(self):
        path = f"{self.out_stem}.mp4"
        if not self.frames:
            print("  [video] no frames")
            return None
        print(f"  [video] writing {len(self.frames)} frames → {path}")
        writer = imageio.get_writer(
            path, fps=self.fps, codec="libx264", quality=8, macro_block_size=None)
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        print(f"  [video] saved {path}")
        return path


def _hold_and_record(model, data, rec, steps, every, lookat_body=None):
    aq = sc.arm_qpos_adr(model)
    target = data.qpos[aq].copy()
    oid = None
    if lookat_body is not None:
        oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lookat_body)
    for i in range(steps):
        data.ctrl[:7] = target
        mujoco.mj_step(model, data)
        if oid is not None:
            rec.set_lookat(data.xpos[oid])
        if i % every == 0:
            rec.capture(data)


def record_sequence(object_name, weld_grasp, out_dir, tilt_deg=45.0,
                    lift_height=0.10, every=8):
    """Run one instrumented arm lift and write media. Returns result dict + paths."""
    os.makedirs(out_dir, exist_ok=True)
    mode = "weld" if weld_grasp else "friction"
    stem = os.path.join(out_dir, f"arm_lift_{object_name}_{mode}")
    print(f"\n=== Recording {object_name} [{mode}] → {stem}.* ===")

    spec = OBJECT_SPECS[object_name]
    xml = os.path.join(SCENES, f"scene_{object_name}.xml")
    build_scene_xml(object_name, xml)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    arm.boost_fingertip_friction(model, sliding=12.0, torsional=0.15, rolling=0.003)
    arm.boost_object_friction(model, spec["body_name"], sliding=12.0, torsional=0.15, rolling=0.003)
    if weld_grasp:
        arm.boost_finger_actuator(model, kp=300.0, kv=30.0)

    sc.set_home_pose(model, data, {spec["body_name"]: spawn_pos(spec)})
    data.qpos[sc.arm_qpos_adr(model)] = sc.ARM_HOME_ANGLES
    data.ctrl[:7] = sc.ARM_HOME_ANGLES
    mujoco.mj_forward(model, data)
    saved = arm.disable_arm_link_collision(model)
    sc.settle(model, data, 250)
    arm.restore_geom_collision(model, saved)
    mujoco.mj_forward(model, data)

    body = spec["body_name"]
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    centroid = centroid_world(spec, data.xpos[oid])
    grasp_pos, grasp_quat = arm.known_good_cylinder_pose_diag(centroid, tilt_deg=tilt_deg)

    approach = arm.approach_axis_from_quat(grasp_quat)
    approach /= np.linalg.norm(approach)
    standoff = 0.05
    pre_hand = grasp_pos - approach * standoff
    pre_ee, pre_q = arm.hand_grasp_to_ee_target(pre_hand, grasp_quat)
    grasp_ee, grasp_q = arm.hand_grasp_to_ee_target(grasp_pos, grasp_quat)

    rec = ArmLiftRecorder(model, stem)
    rec.set_lookat(data.xpos[oid])

    # --- Stage 0: parked / object on table ---
    rec.snapshot(data, "01_home_side", azimuth=90, elevation=-15, distance=1.15)
    rec.snapshot(data, "01_home_34", azimuth=135, elevation=-22, distance=1.05)
    for _ in range(8):
        rec.capture(data)

    # --- Approach ---
    arm.open_gripper_ctrl(model, data)
    arm.disable_arm_link_collision(model)
    arm.ik_move_to_pose_6d(model, data, pre_ee, pre_q, max_steps=2500, pos_tol=0.015, ori_tol=0.2)
    data.ctrl[:7] = data.qpos[sc.arm_qpos_adr(model)]
    _hold_and_record(model, data, rec, 30, every, body)
    rec.snapshot(data, "02_pregrasp_side", azimuth=90, elevation=-18, distance=1.0)
    rec.snapshot(data, "02_pregrasp_34", azimuth=140, elevation=-25, distance=0.95)

    n_approach = 8
    for i in range(1, n_approach + 1):
        t = i / n_approach
        hand_i = pre_hand + t * (grasp_pos - pre_hand)
        ee_i, q_i = arm.hand_grasp_to_ee_target(hand_i, grasp_quat)
        arm.ik_move_to_pose_6d(model, data, ee_i, q_i, max_steps=800, pos_tol=0.015, ori_tol=0.2)
        data.ctrl[:7] = data.qpos[sc.arm_qpos_adr(model)]
        _hold_and_record(model, data, rec, 10, every, body)

    arm.ik_move_to_pose_6d(model, data, grasp_ee, grasp_q, max_steps=2000, pos_tol=0.01, ori_tol=0.15)
    data.ctrl[:7] = data.qpos[sc.arm_qpos_adr(model)]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    _hold_and_record(model, data, rec, 40, every, body)
    arm.ik_fine_converge(model, data, grasp_ee, grasp_q)
    _hold_and_record(model, data, rec, 25, every, body)
    arm.center_between_fingers(model, data, body)
    _hold_and_record(model, data, rec, 15, every, body)

    rec.snapshot(data, "03_grasp_open_side", azimuth=95, elevation=-20, distance=0.85)
    rec.snapshot(data, "03_grasp_open_closeup", azimuth=110, elevation=-30, distance=0.55)

    # Diagnostic probe only (keep links ghosted for physics)
    link_saved = arm.disable_arm_link_collision(model)  # ensure ghosted
    # close
    got = arm.close_gripper_with_squeeze(
        model, data, close_steps=600, min_close_ctrl=100., hard_close=True)
    # Record during a short post-close hold with frames
    aq = sc.arm_qpos_adr(model)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    arm_tgt = data.qpos[aq].copy()
    for i in range(80):
        data.ctrl[:7] = arm_tgt
        data.ctrl[aid] = 0.0
        mujoco.mj_step(model, data)
        rec.set_lookat(data.xpos[oid])
        if i % every == 0:
            rec.capture(data)

    rec.snapshot(data, "04_closed_side", azimuth=95, elevation=-20, distance=0.85)
    rec.snapshot(data, "04_closed_closeup", azimuth=110, elevation=-30, distance=0.50)

    weld_rel = None
    if weld_grasp and got:
        weld_rel = arm.latch_hand_object_relative(model, data, body)

    q_grasp = data.qpos[aq].copy()
    q_lift = arm.solve_vertical_lift_qpos(model, data, lift_height)
    data.ctrl[:7] = q_grasp
    data.ctrl[aid] = 0.0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    z0 = float(data.xpos[oid][2])
    peak_lift = 0.0
    best_held = None
    tol = spec["footprint_radius"] + 0.04
    min_lift = lift_height * 0.35
    snap_mid = False
    snap_peak = False
    blend_n = 1800

    for i in range(blend_n):
        a = min(1.0, i / max(blend_n * 0.55, 1))
        data.ctrl[:7] = (1.0 - a) * q_grasp + a * q_lift
        data.ctrl[aid] = 0.0
        if weld_rel is not None:
            mujoco.mj_forward(model, data)
            arm.apply_hand_object_weld(model, data, body, *weld_rel)
        mujoco.mj_step(model, data)
        cur_lift = float(data.xpos[oid][2] - z0)
        cur_xy = float(np.linalg.norm(data.xpos[oid][:2] - data.site_xpos[ee][:2]))
        peak_lift = max(peak_lift, cur_lift)
        if cur_lift >= min_lift and cur_xy <= tol:
            if best_held is None or cur_lift > best_held[0]:
                best_held = (cur_lift, cur_xy)
        rec.set_lookat(0.7 * data.xpos[oid] + 0.3 * data.site_xpos[ee])
        if i % every == 0:
            rec.capture(data)
        if (not snap_mid) and cur_lift >= 0.03:
            rec.snapshot(data, "05_midlift_side", azimuth=95, elevation=-18, distance=1.0)
            rec.snapshot(data, "05_midlift_34", azimuth=140, elevation=-25, distance=1.05)
            snap_mid = True
        if (not snap_peak) and best_held is not None and cur_lift >= best_held[0] - 1e-4:
            # refresh peak still as we climb
            pass
        if (not snap_peak) and cur_lift >= min_lift and a > 0.7:
            rec.snapshot(data, "06_peak_side", azimuth=95, elevation=-15, distance=1.05)
            rec.snapshot(data, "06_peak_34", azimuth=145, elevation=-22, distance=1.1)
            snap_peak = True

    for i in range(200):
        data.ctrl[:7] = q_lift
        data.ctrl[aid] = 0.0
        if weld_rel is not None:
            mujoco.mj_forward(model, data)
            arm.apply_hand_object_weld(model, data, body, *weld_rel)
        mujoco.mj_step(model, data)
        cur_lift = float(data.xpos[oid][2] - z0)
        cur_xy = float(np.linalg.norm(data.xpos[oid][:2] - data.site_xpos[ee][:2]))
        peak_lift = max(peak_lift, cur_lift)
        if cur_lift >= min_lift and cur_xy <= tol:
            if best_held is None or cur_lift > best_held[0]:
                best_held = (cur_lift, cur_xy)
        rec.set_lookat(0.7 * data.xpos[oid] + 0.3 * data.site_xpos[ee])
        if i % every == 0:
            rec.capture(data)

    # Final stills (held for weld; may show drop for friction)
    rec.snapshot(data, "07_final_side", azimuth=95, elevation=-15, distance=1.1)
    rec.snapshot(data, "07_final_34", azimuth=145, elevation=-22, distance=1.15)

    final_lift = float(data.xpos[oid][2] - z0)
    final_xy = float(np.linalg.norm(data.xpos[oid][:2] - data.site_xpos[ee][:2]))
    if final_lift >= 0.04:
        rec.snapshot(data, "08_hero_lift_closeup", azimuth=120, elevation=-28, distance=0.55)
        rec.snapshot(data, "08_hero_lift_side", azimuth=88, elevation=-12, distance=0.95)

    success_final = final_lift >= min_lift and final_xy <= tol
    success_held = best_held is not None
    success = success_final or success_held

    video_path = rec.save_video()

    # Free renderer
    rec._renderer.close()

    summary = dict(
        object=object_name, mode=mode, success=success,
        success_final=success_final, success_held_midlift=success_held and not success_final,
        peak_lift=round(peak_lift, 5), final_lift=round(final_lift, 5),
        final_xy_offset=round(final_xy, 5), got_contact_on_close=bool(got),
        video=video_path, stills=rec.stills,
    )
    print(f"  result: success={success} peak={summary['peak_lift']} "
          f"final={summary['final_lift']} mode={mode}")
    # silence unused
    _ = link_saved
    return summary


def build_stage_panel(out_dir, object_name="cylinder_thin", mode="weld"):
    """Compose a 2x3 thesis figure from key stills."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [
        ("01_home_34", "1. Home"),
        ("02_pregrasp_34", "2. Pre-grasp"),
        ("03_grasp_open_closeup", "3. At grasp (open)"),
        ("04_closed_closeup", "4. Closed"),
        ("06_peak_side", "5. Peak lift"),
        ("07_final_side", "6. Final"),
    ]
    stem = os.path.join(out_dir, f"arm_lift_{object_name}_{mode}")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, (key, title) in zip(axes.ravel(), stages):
        path = f"{stem}_{key}.png"
        if os.path.isfile(path):
            ax.imshow(imageio.imread(path))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Full-arm grasp & lift — {object_name.replace('_', ' ')} ({mode})",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(out_dir, f"arm_lift_{object_name}_{mode}_stage_panel.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [panel] {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="Record arm-lift thesis screenshots + videos")
    p.add_argument("--object", default="cylinder_thin",
                   choices=["cylinder", "cylinder_thin", "cylinder_thinner"])
    p.add_argument("--mode", default="both", choices=["friction", "weld", "both"])
    p.add_argument("--out", default=OUT_DIR)
    args = p.parse_args()

    modes = []
    if args.mode in ("friction", "both"):
        modes.append(False)
    if args.mode in ("weld", "both"):
        modes.append(True)

    results = []
    for weld in modes:
        results.append(record_sequence(args.object, weld_grasp=weld, out_dir=args.out))
        mode_name = "weld" if weld else "friction"
        build_stage_panel(args.out, args.object, mode_name)

    # Index file for thesis drafting
    index_path = os.path.join(args.out, "INDEX.md")
    with open(index_path, "w") as f:
        f.write("# Full-arm lift thesis media\n\n")
        f.write("Generated by `record_arm_lift_v4.py`.\n\n")
        f.write("## Folder\n\n")
        f.write(f"`{args.out}`\n\n")
        for r in results:
            f.write(f"## {r['object']} — {r['mode']}\n\n")
            f.write(f"- success: **{r['success']}** "
                    f"(final={r['success_final']}, midlift={r['success_held_midlift']})\n")
            f.write(f"- peak_lift: {r['peak_lift']} m, final_lift: {r['final_lift']} m\n")
            if r["video"]:
                f.write(f"- video: `{os.path.basename(r['video'])}`\n")
            panel = f"arm_lift_{r['object']}_{r['mode']}_stage_panel.png"
            if os.path.isfile(os.path.join(args.out, panel)):
                f.write(f"- stage panel: `{panel}`\n")
            f.write("- stills:\n")
            for name, path in sorted(r["stills"].items()):
                f.write(f"  - `{os.path.basename(path)}` ({name})\n")
            f.write("\n")
        f.write("### Suggested thesis use\n\n")
        f.write("- **Main figure:** `arm_lift_cylinder_thin_weld_stage_panel.png` "
                "(clean sustained lift sequence)\n")
        f.write("- **Hero still:** `..._weld_06_peak_side.png` or "
                "`..._weld_08_hero_lift_*.png`\n")
        f.write("- **Honest friction comparison:** friction `06_peak` vs "
                "`07_final` (peak lift then slip)\n")
        f.write("- **Videos:** embed the `.mp4` files in slides / digital appendix\n")
    print(f"\nIndex: {index_path}")
    print("--- SUMMARY ---")
    for r in results:
        print(f"  {r['object']} {r['mode']}: "
              f"{'PASS' if r['success'] else 'FAIL'} "
              f"peak={r['peak_lift']} final={r['final_lift']}")
    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
