"""
demo_floating_gripper.py
========================
Slow, interactive MuJoCo viewer demo of the full grasp-and-lift routine.

Shows every frame — approach (fingers open) -> close fingers -> lift +
shake under gravity — for each of the three experiment objects.

By default uses validated hand-tuned poses (guaranteed success, no CGN
wait). Add --cgn to use Contact-GraspNet instead.

Usage (macOS):
    mjpython demo_floating_gripper.py                     # all 3 objects, slow
    mjpython demo_floating_gripper.py --object cylinder   # one object
    mjpython demo_floating_gripper.py --cgn               # use CGN grasps
    mjpython demo_floating_gripper.py --speed 0.5         # half speed (even slower)

Linux:
    python3 demo_floating_gripper.py
"""

import os, sys, platform, shutil


def _ensure_mjpython():
    """Re-exec under mjpython on macOS — required for the passive viewer."""
    if platform.system() != 'Darwin':
        return
    if os.environ.get('_MJPY_REEXEC') == '1':
        return
    exe = sys.executable
    if os.path.basename(exe) in ('mjpython',) or 'mjpython' in exe:
        return
    mjpy = shutil.which('mjpython')
    if not mjpy:
        sys.exit('ERROR: mjpython not found. Install mujoco and try:\n'
                 '  pip install mujoco\nThen: mjpython demo_floating_gripper.py')
    os.environ['_MJPY_REEXEC'] = '1'
    os.execv(mjpy, [mjpy, os.path.abspath(__file__)] + sys.argv[1:])


_ensure_mjpython()

import time, math, argparse
import numpy as np
import mujoco
import mujoco.viewer

_PROJECT = os.path.dirname(os.path.abspath(__file__))
_CGN_REPO = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
_CGN_SRC  = os.path.join(_CGN_REPO, 'contact_graspnet_pytorch')
for _p in [_CGN_REPO, _CGN_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc
from object_specs import (OBJECT_SPECS, OBJECT_NAMES, build_scene_xml,
                           spawn_pos, centroid_world, FLOATING_GRIPPER_TEMPLATE)

SCENES_DIR      = os.path.join(_PROJECT, 'generated_scenes')
CKPT_DIR        = os.path.join(_CGN_REPO, 'checkpoints', 'contact_graspnet')
IMG_W, IMG_H    = 640, 480
CAM_RADIUS      = 0.8
PARK_POS        = np.array([0.5, -0.35, 0.75])
CLOSE_STEPS     = 400
SHAKE_STEPS     = 600
LIFT_HEIGHT     = 0.15       # metres
SHAKE_AMPLITUDE = 0.03
SQUEEZE_MARGIN  = 60.
# Render every Nth sim step (1 = every step, smoothest but slower on CPU).
# Keep at 1 — the sleep_per_step controls wall-clock speed.
RENDER_EVERY    = 1
# Wall-clock sleep per rendered frame at speed=1.0.
# MuJoCo dt ~ 0.002 s, so 1 sim step = 0.002 s real time.
# 0.010 s sleep/step → ~5× slower than real time (easy to follow visually).
SLEEP_PER_STEP  = 0.010

os.makedirs(SCENES_DIR, exist_ok=True)


# ─── known-good hand-tuned poses (validated in smoke_test_floating_gripper.py) ─

def _known_good_pose(object_name, centroid):
    c = np.asarray(centroid, dtype=float)
    if object_name == 'cylinder':
        approach = np.array([-1., 0., 0.])
        base     = np.array([ 0., 0., 1.])
        y = np.cross(approach, base)
        R = np.column_stack((base, y, approach))
        return c + np.array([0.12, 0., 0.]), sc.rot_matrix_to_quat(R)
    if object_name == 'box':
        approach = np.array([ 0., -1., 0.])
        base     = np.array([ 0.,  0., 1.])
        y = np.cross(approach, base)
        R = np.column_stack((base, y, approach))
        return c + np.array([0., 0.12, 0.]), sc.rot_matrix_to_quat(R)
    if object_name == 'mustard':
        # Side approach from +x (mirrors cylinder; mustard has similar diameter)
        approach = np.array([-1., 0., 0.])
        base     = np.array([ 0., 0., 1.])
        y = np.cross(approach, base)
        R = np.column_stack((base, y, approach))
        return c + np.array([0.12, 0., 0.]), sc.rot_matrix_to_quat(R)
    raise ValueError(f'No hand-tuned pose for {object_name}')


def get_pose_known_good(object_name, spec):
    """Settle the object, return its resting pos/quat + hand-tuned grasp pose."""
    scene_xml = os.path.join(SCENES_DIR, f'scene_{object_name}.xml')
    build_scene_xml(object_name, scene_xml)

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)
    sc.set_home_pose(model, data, {spec['body_name']: spawn_pos(spec)})
    mujoco.mj_forward(model, data)

    arm_saved = sc.disable_arm_collision(model)
    sc.ik_move_to(model, data, PARK_POS, max_steps=1000, tol=0.02)
    sc.settle(model, data, 200)
    sc.restore_arm_collision(model, arm_saved)
    mujoco.mj_forward(model, data)

    obj_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
    centroid = centroid_world(spec, data.xpos[obj_id])
    grasp_pos, grasp_quat = _known_good_pose(object_name, centroid)
    return data.xpos[obj_id].copy(), data.xquat[obj_id].copy(), grasp_pos, grasp_quat


def get_pose_cgn(object_name, spec, phi, theta, sigma_d, rho, seed):
    from contact_grasp_estimator import GraspEstimator
    import config_utils
    from checkpoints import CheckpointIO
    from data import load_available_input_data
    import torch

    scene_xml = os.path.join(SCENES_DIR, f'scene_{object_name}.xml')
    build_scene_xml(object_name, scene_xml)

    rng = np.random.default_rng(seed)
    sc.seed_cgn_global_random(seed)

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data  = mujoco.MjData(model)
    sc.set_home_pose(model, data, {spec['body_name']: spawn_pos(spec)})
    mujoco.mj_forward(model, data)

    arm_saved = sc.disable_arm_collision(model)
    sc.ik_move_to(model, data, PARK_POS, max_steps=1000, tol=0.02)
    sc.settle(model, data, 200)
    sc.restore_arm_collision(model, arm_saved)
    mujoco.mj_forward(model, data)

    obj_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
    centroid = centroid_world(spec, data.xpos[obj_id])
    sc.set_camera(model, phi, theta, CAM_RADIUS, np.array(centroid))
    mujoco.mj_forward(model, data)

    depth, K, seg_map, _seg_empty = sc.render_depth_seg(
        model, data, {spec['body_name']: 1}, sigma_d=sigma_d, rng=rng,
        img_w=IMG_W, img_h=IMG_H)

    cfg = config_utils.load_config(CKPT_DIR, batch_size=1, arg_configs=[])
    est = GraspEstimator(cfg)
    CheckpointIO(checkpoint_dir=os.path.join(CKPT_DIR, 'checkpoints'),
                 model=est.model).load('model.pt')
    est.model.eval()

    tmp = os.path.join(_PROJECT, '_tmp_demo.npz')
    np.savez(tmp, depth=depth, K=K, seg=seg_map)
    from data import load_available_input_data
    segmap, rgb, depth_in, cam_K, _, _ = load_available_input_data(tmp, K=None)
    pc_full, pc_segs, _ = est.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb,
        skip_border_objects=False, z_range=[0.1, 2.0])
    if rho < 1. and len(pc_full) > 0:
        idx = sc.deterministic_downsample_idx(len(pc_full), rho)
        pc_full = pc_full[idx]
        for k in pc_segs:
            s = pc_segs[k]
            if len(s) > 0:
                pc_segs[k] = s[sc.deterministic_downsample_idx(len(s), rho)]
    with torch.no_grad():
        pred_grasps, scores, _, _ = est.predict_scene_grasps(
            pc_full, pc_segments=pc_segs,
            local_regions=True, filter_grasps=True, forward_passes=1)
    os.path.exists(tmp) and os.remove(tmp)

    n_grasps = sum(len(scores[k]) for k in scores)
    if n_grasps == 0:
        print(f'[CGN] No grasps found — falling back to hand-tuned pose for {object_name}')
        centroid = centroid_world(spec, data.xpos[obj_id])
        gp, gq = _known_good_pose(object_name, centroid)
        return data.xpos[obj_id].copy(), data.xquat[obj_id].copy(), gp, gq, None

    pose_cam, q_grasp, _ = sc.best_grasp_overall(pred_grasps, scores)
    pose_world = sc.cam_to_world(pose_cam, model, data)
    grasp_pos  = pose_world[:3, 3]
    grasp_quat = sc.rot_matrix_to_quat(pose_world[:3, :3])
    return data.xpos[obj_id].copy(), data.xquat[obj_id].copy(), grasp_pos, grasp_quat, q_grasp


# ─── core animated demo ──────────────────────────────────────────────────────

def _sync(viewer, model, data, sleep_s):
    """Sync viewer and sleep; returns False if viewer was closed."""
    if not viewer.is_running():
        return False
    viewer.sync()
    time.sleep(sleep_s)
    return True


def _hold_still(viewer, model, data, n_steps, sleep_s):
    """Run physics + render n_steps times without moving the gripper."""
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        if not _sync(viewer, model, data, sleep_s):
            return False
    return True


def _update_cam_lookat(viewer, pos):
    """Smoothly keep the viewer lookat centred on pos (world coords)."""
    viewer.cam.lookat[:] = pos


def run_demo_animated(model, data, spec, grasp_pos, grasp_quat, viewer, sleep_s):
    """
    Full animated grasp routine, rendered every sim step:
      1. Approach  — teleport gripper open to grasp pose, hold 1 s
      2. Close     — close fingers slowly, camera follows
      3. Lift+shake — rise 15 cm while shaking, camera follows object up
      4. Hold      — freeze at top for 3 s so you can see the result
    """
    gp   = np.asarray(grasp_pos, dtype=float)
    gq   = np.asarray(grasp_quat, dtype=float)
    tb   = spec['body_name']
    fp   = spec['footprint_radius']
    aid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
    ees  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
    oid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tb)
    tol  = fp + 0.03

    # ── Phase 1: Approach (fingers open) ───────────────────────────────────
    print('[Demo] Phase 1 — Approach (fingers open)')
    sc.teleport_hand_hard(model, data, gp, gq)
    sc.open_gripper(model, data)
    mujoco.mj_forward(model, data)

    contacted = sorted(sc.gripper_contacted_bodies(model, data, sc.GRIPPER_BODIES))
    collision_free = len(contacted) == 0
    print(f'         collision_free={collision_free}  contacted={contacted}')

    # Hold at approach for ~1.5 s so viewer can take in the scene
    approach_frames = int(1.5 / sleep_s)
    for _ in range(approach_frames):
        mujoco.mj_step(model, data)
        _update_cam_lookat(viewer, data.xpos[oid].copy())
        if not _sync(viewer, model, data, sleep_s):
            return None

    if not collision_free:
        print('[Demo] Collision — showing for 2 s then exiting this object')
        _hold_still(viewer, model, data, int(2.0 / sleep_s), sleep_s)
        return dict(collision_free=False, contacted_bodies=contacted, success=False)

    z0 = float(data.xpos[oid][2])

    # ── Phase 2: Close fingers ─────────────────────────────────────────────
    print('[Demo] Phase 2 — Closing fingers')
    contact_ctrl = None
    for step in range(CLOSE_STEPS):
        sc.teleport_mocap(model, data, 'hand_target', gp, gq)
        if contact_ctrl is None:
            data.ctrl[aid] = max(0., data.ctrl[aid] - 255. / CLOSE_STEPS)
            mujoco.mj_step(model, data)
            if sc.gripper_contacted_bodies(model, data, sc.GRIPPER_BODIES):
                contact_ctrl = max(0., data.ctrl[aid] - SQUEEZE_MARGIN)
        else:
            data.ctrl[aid] = contact_ctrl
            mujoco.mj_step(model, data)
        if step % RENDER_EVERY == 0:
            _update_cam_lookat(viewer, data.xpos[oid].copy())
            if not _sync(viewer, model, data, sleep_s):
                return None

    if contact_ctrl is not None:
        data.ctrl[aid] = contact_ctrl

    # Pause with fingers closed for 1 s
    for _ in range(int(1.0 / sleep_s)):
        mujoco.mj_step(model, data)
        _update_cam_lookat(viewer, data.xpos[oid].copy())
        if not _sync(viewer, model, data, sleep_s):
            return None

    # ── Phase 3: Lift + shake ──────────────────────────────────────────────
    print('[Demo] Phase 3 — Lift + shake')
    held_throughout = True
    for i in range(SHAKE_STEPS):
        t    = i / SHAKE_STEPS
        ramp = min(1.0, t / 0.3)
        dz   = LIFT_HEIGHT * ramp
        dx   = SHAKE_AMPLITUDE * ramp * math.sin(2 * math.pi * 4.0 * t)
        dy   = SHAKE_AMPLITUDE * ramp * math.sin(2 * math.pi * 5.3 * t)
        sc.teleport_mocap(model, data, 'hand_target', gp + np.array([dx, dy, dz]), gq)
        mujoco.mj_step(model, data)

        if i % 25 == 0 and ramp >= 1.0:
            obj_xy = data.xpos[oid][:2]
            ee_xy  = data.site_xpos[ees][:2]
            xy_off = float(np.linalg.norm(obj_xy - ee_xy))
            if xy_off > tol or (data.xpos[oid][2] - z0) < LIFT_HEIGHT * 0.4:
                held_throughout = False

        if i % RENDER_EVERY == 0:
            # Camera lookat follows the object as it rises
            _update_cam_lookat(viewer, data.xpos[oid].copy())
            if not _sync(viewer, model, data, sleep_s):
                return None

    # ── Phase 4: Final result hold ─────────────────────────────────────────
    obj_final  = data.xpos[oid].copy()
    xy_offset  = float(np.linalg.norm(obj_final[:2] - data.site_xpos[ees][:2]))
    lift       = float(obj_final[2] - z0)
    success    = held_throughout and xy_offset <= tol and lift >= LIFT_HEIGHT * 0.4

    tag = 'SUCCESS ✓' if success else 'FAILED ✗'
    print(f'[Demo] Result: {tag}  lift={lift:.3f} m  xy_offset={xy_offset:.4f} m')
    print('[Demo] Holding at top — close the window or wait 5 s for next object...')

    hold_frames = int(5.0 / sleep_s)
    for _ in range(hold_frames):
        mujoco.mj_step(model, data)
        _update_cam_lookat(viewer, data.xpos[oid].copy())
        if not _sync(viewer, model, data, sleep_s):
            break

    return dict(collision_free=True, contacted_bodies=[], success=bool(success),
                final_xy_offset=round(xy_offset, 5), final_lift=round(lift, 5),
                obj_z_final=round(float(obj_final[2]), 4))


# ─── per-object entry point ───────────────────────────────────────────────────

def demo_object(object_name, use_cgn, phi, theta, sigma_d, rho, seed, speed):
    spec = OBJECT_SPECS[object_name]
    sleep_s = SLEEP_PER_STEP / max(speed, 0.05)

    print(f'\n{"=" * 70}')
    print(f'  OBJECT: {object_name} — {spec["label"]}')
    print(f'  Pose source: {"CGN" if use_cgn else "hand-tuned (guaranteed success)"}')
    print(f'{"=" * 70}')

    fg_xml = os.path.join(SCENES_DIR, f'scene_{object_name}_floating_gripper.xml')
    build_scene_xml(object_name, fg_xml, template_path=FLOATING_GRIPPER_TEMPLATE)

    print('[Setup] Settling object in perception scene...')
    if use_cgn:
        obj_pos, obj_quat, grasp_pos, grasp_quat, q = get_pose_cgn(
            object_name, spec, phi, theta, sigma_d, rho, seed)
        if q is not None:
            print(f'[CGN]  top-1 score={q:.4f}  grasp_pos={grasp_pos.round(3)}')
    else:
        obj_pos, obj_quat, grasp_pos, grasp_quat = get_pose_known_good(object_name, spec)
        print(f'[Pose] Hand-tuned side grasp  pos={grasp_pos.round(3)}')

    model = mujoco.MjModel.from_xml_path(fg_xml)
    data  = mujoco.MjData(model)
    sc.set_object_pose(model, data, spec['body_name'], obj_pos, obj_quat)
    mujoco.mj_forward(model, data)
    sc.settle(model, data, 30)

    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
    start_lookat = data.xpos[oid].copy()

    print(f'[Viewer] Opening window (speed={speed:.2f}× — sleep {sleep_s*1000:.0f} ms/step)')
    print('         Close the window to skip to the next object.')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Slightly elevated angle so lift is clearly visible
        viewer.cam.distance  = 0.80
        viewer.cam.azimuth   = 130.
        viewer.cam.elevation = -22.
        viewer.cam.lookat[:] = start_lookat
        return run_demo_animated(model, data, spec, grasp_pos, grasp_quat,
                                 viewer, sleep_s)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Slow MuJoCo viewer: watch gripper grasp & lift each object')
    p.add_argument('--object', nargs='+', choices=OBJECT_NAMES, default=OBJECT_NAMES,
                   metavar='OBJ', help='Objects to demo (default: all three)')
    p.add_argument('--cgn', action='store_true',
                   help='Use Contact-GraspNet to pick the pose (loads CGN, ~30 s)')
    p.add_argument('--speed', type=float, default=1.0,
                   help='Speed multiplier — 1.0 = normal slow, 0.5 = half speed, 2.0 = faster')
    p.add_argument('--phi',     type=float, default=45., help='CGN camera elevation (deg)')
    p.add_argument('--theta',   type=float, default=0.,  help='CGN camera azimuth (deg)')
    p.add_argument('--sigma_d', type=float, default=0.,  help='Depth noise std (m)')
    p.add_argument('--rho',     type=float, default=1.0, help='Point-cloud keep fraction')
    p.add_argument('--seed',    type=int,   default=42)
    args = p.parse_args()

    sc.configure_determinism()
    print('\nDemo: floating-gripper grasp + lift routine')
    print(f'Objects : {args.object}')
    print(f'Poses   : {"CGN" if args.cgn else "hand-tuned (guaranteed success)"}')
    print(f'Speed   : {args.speed}×  (~{SLEEP_PER_STEP/args.speed*1000:.0f} ms/step)\n')

    results = {}
    for obj in args.object:
        try:
            results[obj] = demo_object(obj, args.cgn, args.phi, args.theta,
                                       args.sigma_d, args.rho, args.seed, args.speed)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results[obj] = {'error': str(exc)}

    print(f'\n{"=" * 70}\n  SUMMARY\n{"=" * 70}')
    for obj, res in results.items():
        if res is None:
            print(f'  {obj}: viewer closed early')
        elif 'error' in res:
            print(f'  {obj}: ERROR — {res["error"]}')
        else:
            tag = 'SUCCESS' if res.get('success') else 'FAIL'
            print(f'  {obj}: {tag}  lift={res.get("final_lift")} m  '
                  f'collision_free={res.get("collision_free")}')
    print()


if __name__ == '__main__':
    main()
