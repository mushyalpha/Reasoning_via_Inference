"""
smoke_test_arm_grasp_v4.py — force a full-arm lift.

Uses diagonal side-grasp (avoids elbow∩table), hard close, stiff finger
servo, and optional weld latch. Compares original vs thin cylinders.
"""
import os, sys
import numpy as np
import mujoco

_PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT)

import sim_common as sc
import sim_arm_v4 as arm
import object_specs_v4_patch  # noqa: F401
from object_specs import OBJECT_SPECS, build_scene_xml, spawn_pos, centroid_world

SCENES = os.path.join(_PROJECT, "generated_scenes")


def run_one(object_name, weld_grasp=False, tilt_deg=45.0):
    spec = OBJECT_SPECS[object_name]
    xml = os.path.join(SCENES, f"scene_{object_name}.xml")
    build_scene_xml(object_name, xml)

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    arm.boost_fingertip_friction(model, sliding=12.0, torsional=0.15, rolling=0.003)
    arm.boost_object_friction(model, spec["body_name"], sliding=12.0, torsional=0.15, rolling=0.003)
    # Note: raising finger kp above ~200 hurts frictional lift (over-stiff
    # squeeze ejects / fails to seat). Leave Menagerie defaults for friction;
    # weld path does not need stiff fingers.
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

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec["body_name"])
    centroid = centroid_world(spec, data.xpos[obj_id])
    grasp_pos, grasp_quat = arm.known_good_cylinder_pose_diag(centroid, tilt_deg=tilt_deg)

    result = arm.run_arm_approach_grasp_lift(
        model, data, spec["body_name"], grasp_pos, grasp_quat,
        footprint_radius=spec["footprint_radius"],
        standoff=0.05, n_approach=8, lift_height=0.10,
        check_pregrasp_collision=True,
        hard_close=True, weld_grasp=weld_grasp,
        blend_lift_steps=1800)

    clearance_mm = (0.04 - spec["footprint_radius"]) * 1000
    mode = "WELD" if weld_grasp else "FRICTION"
    print(f"\n=== {object_name} [{mode}] r={spec['footprint_radius']*100:.1f}cm "
          f"clear={clearance_mm:.1f}mm/side tilt={tilt_deg} ===")
    for k in ("ik_grasp_ok", "fine_converge_ok", "got_contact_on_close",
              "weld_grasp", "peak_lift", "held_lift", "final_lift",
              "final_xy_offset", "success_final", "success_held_midlift",
              "failure_mode", "success"):
        print(f"  {k}: {result.get(k)}")
    return result["success"]


def main():
    args = sys.argv[1:]
    weld = "--weld" in args
    args = [a for a in args if a != "--weld"]
    objects = args or ["cylinder_thin", "cylinder_thinner", "cylinder"]
    results = {}
    for name in objects:
        key = f"{name}{'+weld' if weld else ''}"
        try:
            results[key] = run_one(name, weld_grasp=weld)
        except Exception as e:
            print(f"\n=== {name}: ERROR {e} ===")
            results[key] = False

    # If friction failed, also try weld on the best candidate
    if not weld and not any(results.values()):
        print("\n--- friction failed; trying weld latch on cylinder_thin ---")
        results["cylinder_thin+weld"] = run_one("cylinder_thin", weld_grasp=True)

    print("\n--- SUMMARY ---")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
