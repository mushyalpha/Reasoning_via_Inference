"""
run_clutter_experiments.py
===========================
Experiment B (clutter) -- response to Marker B's central point:

  "Because your thesis focuses on degrading the perception system to
  extract the causal structure of the process, the number of objects in
  the scene matters immensely. [...] Show how perception degradation
  introduces unexpected causal failures -- specifically proving that
  visual noise in clutter leads to inter-object collisions rather than
  just simple missing/slipping errors."

All 3 objects (cylinder, box, mustard -- see object_specs.py) are placed
together in a fixed triangular arrangement (object_specs.clutter_layout).
On each trial, ONE object is the designated grasp target; the other two
are visual/physical clutter. The new outcome variable is
`collision_with_neighbor`: whether the gripper contacts a NON-target
body at any point during the approach/descend/close sequence (via
MuJoCo's live contact array -- sim_common.finger_nontarget_collision).

This directly tests the mechanism Marker B describes: degraded
perception can make a target grasp look geometrically valid while the
execution clips a neighbouring, visually-blurred object -- a distinct
failure mode from Experiment A's missed/slipped grasps.

Outcome-variable fix (same second-round marking feedback as
run_experiments_v2.py -- see that module's docstring, RIGOUR_LEDGER.md,
MARKER_FEEDBACK.md): `success` is now the floating-gripper shake-test
result (sim_common.run_floating_gripper_test), not a proximity
threshold on e_pose. `collision_with_neighbor` is now read directly off
the SAME floating-gripper collision check used to determine
`collision_free` -- at the predicted (open-finger) pose, any contact
with a body that is neither the target nor a static scene fixture is a
neighbour collision. This is a cleaner, more direct test of Marker B's
"finger clips a blurred neighbour" mechanism than the old descend-and-
watch-for-contact approach, because it is evaluated at the exact
predicted 6-DoF pose rather than along an IK-planned approach path.

Grid (targeted, not a full factorial replication of Experiment A --
this is a mechanism/robustness check, not the primary causal grid):
  sigma_d : [0.000, 0.0025, 0.005, 0.010, 0.015, 0.020, 0.040]  (7, dense)
  rho     : [1.0, 0.75, 0.5, 0.25]                              (4, deterministic)
  phi     : [45, 60]   -- favourable vs. the pathological viewpoint
                           (RIGOUR_LEDGER Stage 8: phi=60 dominates the
                           "no single-variable fix" failures in the
                           single-object data; testing whether clutter
                           compounds or is orthogonal to this regime)
  theta   : [0]         -- fixed, single azimuth (keeps the grid tractable;
                           azimuth was the weaker main effect in the
                           original single-object fit)
  target  : [cylinder, box, mustard]  -- rotate which object is graspd
  seeds   : 3
Total: 7 x 4 x 2 x 1 x 3 x 3 = 504 trials.

Usage:
    python run_clutter_experiments.py            # full 504-trial grid
    python run_clutter_experiments.py --test      # tiny smoke test
    python run_clutter_experiments.py --resume
"""

import os
import sys
import time
import argparse
import itertools
import csv
import traceback
import numpy as np
import mujoco

_PROJECT = os.path.dirname(os.path.abspath(__file__))
_CGN_REPO = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
_CGN_SRC = os.path.join(_CGN_REPO, 'contact_graspnet_pytorch')
for _p in [_CGN_REPO, _CGN_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contact_grasp_estimator import GraspEstimator
import config_utils
from checkpoints import CheckpointIO
from data import load_available_input_data
import torch

import sim_common as sc
from object_specs import (OBJECT_SPECS, CLUTTER_BODY_NAMES, clutter_layout,
                           clutter_spawn_positions, centroid_world, build_scene_xml,
                           FLOATING_GRIPPER_TEMPLATE)

RESULTS_DIR = os.path.join(_PROJECT, 'results')
SCENES_DIR = os.path.join(_PROJECT, 'generated_scenes')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'clutter_results.csv')
CKPT_DIR = os.path.join(_CGN_REPO, 'checkpoints', 'contact_graspnet')
SCENE_XML = os.path.join(SCENES_DIR, 'scene_clutter.xml')
FG_SCENE_XML = os.path.join(SCENES_DIR, 'scene_clutter_floating_gripper.xml')

IMG_W, IMG_H = 640, 480
CAM_RADIUS = 0.8
PARK_POS = np.array([0.5, -0.35, 0.75])

# Floating-gripper shake-test parameters -- identical to run_experiments_v2.py
# so Experiment A and B outcomes are directly comparable.
CLOSE_STEPS = 400
SHAKE_STEPS = 600
LIFT_HEIGHT = 0.15
SHAKE_AMPLITUDE = 0.03
STATIC_SCENE_BODIES = ('table', 'world')

SIGMA_D_VALS = [0.000, 0.0025, 0.005, 0.010, 0.015, 0.020, 0.040]
RHO_VALS = [1.00, 0.75, 0.50, 0.25]
PHI_VALS = [45., 60.]
THETA_VALS = [0.]
TARGETS = ['cylinder', 'box', 'mustard']
N_REPEATS = 3

CSV_FIELDS = [
    'trial_id', 'target_object', 'sigma_d', 'rho', 'phi', 'theta', 'seed',
    'C_pc', 'q_grasp', 'e_pose', 'n_grasps',
    'collision_free', 'success', 'collision_with_neighbor',
    'final_xy_offset', 'final_lift', 'obj_z_final', 'error'
]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)


def load_cgn():
    cfg = config_utils.load_config(CKPT_DIR, batch_size=1, arg_configs=[])
    est = GraspEstimator(cfg)
    CheckpointIO(checkpoint_dir=os.path.join(CKPT_DIR, 'checkpoints'),
                 model=est.model).load('model.pt')
    est.model.eval()
    return est


def run_cgn(depth, K, seg_map, estimator, rho=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    tmp = os.path.join(_PROJECT, f'_tmp_clutter_{os.getpid()}.npz')
    np.savez(tmp, depth=depth, K=K, seg=seg_map)
    segmap, rgb, depth_in, cam_K, _, _ = load_available_input_data(tmp, K=None)

    pc_full, pc_segs, _ = estimator.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb,
        skip_border_objects=False, z_range=[0.1, 2.0])

    if rho < 1. and len(pc_full) > 0:
        idx = sc.deterministic_downsample_idx(len(pc_full), rho)
        pc_full = pc_full[idx]
        for k in pc_segs:
            s = pc_segs[k]
            if len(s) > 0:
                idx_s = sc.deterministic_downsample_idx(len(s), rho)
                pc_segs[k] = s[idx_s]

    with torch.no_grad():
        pred_grasps, scores, _, _ = estimator.predict_scene_grasps(
            pc_full, pc_segments=pc_segs,
            local_regions=True, filter_grasps=True, forward_passes=1)

    os.path.exists(tmp) and os.remove(tmp)
    return pred_grasps, scores


def execute_grasp_clutter_floating(fg_scene_xml, target_spec, target_body,
                                    all_obj_poses, grasp_pos, grasp_quat):
    """
    Floating-gripper shake test for the clutter scene (Marker A's fix,
    same protocol as run_experiments_v2.py). All three objects are
    carried over into the arm-free scene at their settled poses so the
    gripper can still clip a neighbour at the exact predicted pose --
    `collision_with_neighbor` is read directly off that check, rather
    than sampled along an IK-planned descent path as in the old version.

    all_obj_poses : dict {body_name: (pos, quat)} for ALL clutter bodies
                    (target + distractors), from the perception scene.
    """
    fg_model = mujoco.MjModel.from_xml_path(fg_scene_xml)
    fg_data = mujoco.MjData(fg_model)
    for body_name, (pos, quat) in all_obj_poses.items():
        sc.set_object_pose(fg_model, fg_data, body_name, pos, quat)
    mujoco.mj_forward(fg_model, fg_data)
    sc.settle(fg_model, fg_data, 30)

    result = sc.run_floating_gripper_test(
        fg_model, fg_data, target_body, grasp_pos, grasp_quat,
        footprint_radius=target_spec['footprint_radius'],
        close_steps=CLOSE_STEPS, shake_steps=SHAKE_STEPS,
        lift_height=LIFT_HEIGHT, shake_amplitude=SHAKE_AMPLITUDE)

    collided_neighbor = any(b not in (target_body,) + STATIC_SCENE_BODIES
                             for b in result['contacted_bodies'])
    return result, collided_neighbor


def cluster_centroid(model, data):
    pts = []
    for key, bname in CLUTTER_BODY_NAMES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        pts.append(centroid_world(OBJECT_SPECS[key], data.xpos[bid]))
    return np.mean(np.array(pts), axis=0)


def run_trial(trial_id, target_name, sigma_d, rho, phi, theta, seed, estimator):
    target_body = CLUTTER_BODY_NAMES[target_name]
    target_spec = OBJECT_SPECS[target_name]
    rng = np.random.default_rng(seed)
    sc.seed_cgn_global_random(seed)

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    spawn_positions = clutter_spawn_positions()
    sc.set_home_pose(model, data, spawn_positions)
    mujoco.mj_forward(model, data)

    arm_saved = sc.disable_arm_collision(model)
    sc.ik_move_to(model, data, PARK_POS, max_steps=1000, tol=0.02)
    sc.settle(model, data, 300)
    sc.restore_arm_collision(model, arm_saved)
    mujoco.mj_forward(model, data)

    look_at = cluster_centroid(model, data)
    sc.set_camera(model, phi, theta, CAM_RADIUS, look_at)
    mujoco.mj_forward(model, data)

    depth, K, seg_map, _seg_empty = sc.render_depth_seg(
        model, data, {target_body: 1}, sigma_d=sigma_d, rng=rng,
        img_w=IMG_W, img_h=IMG_H)
    C_pc = float(seg_map.sum()) / (IMG_W * IMG_H)

    pred_grasps, scores = run_cgn(depth, K, seg_map, estimator, rho=rho, rng=rng)
    n_grasps = sum(len(scores[k]) for k in scores)

    row_base = dict(trial_id=trial_id, target_object=target_name, sigma_d=sigma_d,
                     rho=rho, phi=phi, theta=theta, seed=seed)

    if n_grasps == 0:
        return {**row_base, 'C_pc': C_pc, 'q_grasp': None, 'e_pose': None,
                'n_grasps': 0, 'collision_free': None, 'success': 0,
                'collision_with_neighbor': None, 'final_xy_offset': None,
                'final_lift': None, 'obj_z_final': None, 'error': 'no_grasps'}

    pose_cam, q_grasp, _ = sc.best_grasp_overall(pred_grasps, scores)
    pose_world = sc.cam_to_world(pose_cam, model, data)
    grasp_pos = pose_world[:3, 3]
    grasp_quat = sc.rot_matrix_to_quat(pose_world[:3, :3])

    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    centroid_now = centroid_world(target_spec, data.xpos[tid].copy())
    e_pose = float(np.linalg.norm(np.array(centroid_now[:2]) - grasp_pos[:2]))

    all_obj_poses = {}
    for key, bname in CLUTTER_BODY_NAMES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        all_obj_poses[bname] = (data.xpos[bid].copy(), data.xquat[bid].copy())

    result, collided_neighbor = execute_grasp_clutter_floating(
        FG_SCENE_XML, target_spec, target_body, all_obj_poses, grasp_pos, grasp_quat)

    return {**row_base, 'C_pc': round(C_pc, 5), 'q_grasp': round(q_grasp, 5),
            'e_pose': round(e_pose, 5), 'n_grasps': n_grasps,
            'collision_free': int(result['collision_free']),
            'success': int(result['success']),
            'collision_with_neighbor': int(collided_neighbor),
            'final_xy_offset': result['final_xy_offset'],
            'final_lift': result['final_lift'],
            'obj_z_final': result['obj_z_final'], 'error': ''}


def load_completed(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((row['target_object'], int(row['trial_id'])))
    return done


def build_grid(sigma_d_vals, rho_vals, phi_vals, theta_vals, targets, n_repeats):
    trials = []
    for i, (sigma_d, rho, phi, theta, target, rep) in enumerate(
            itertools.product(sigma_d_vals, rho_vals, phi_vals, theta_vals, targets, range(n_repeats))):
        seed = rep * 10000 + i
        trials.append((i, target, sigma_d, rho, phi, theta, seed))
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.test:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, targets, n_repeats = \
            [0.0, 0.02], [1.0, 0.5], [45.], [0.], ['cylinder', 'box'], 1
    else:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, targets, n_repeats = \
            SIGMA_D_VALS, RHO_VALS, PHI_VALS, THETA_VALS, TARGETS, N_REPEATS

    trials = build_grid(sigma_d_vals, rho_vals, phi_vals, theta_vals, targets, n_repeats)
    completed = load_completed(OUTPUT_CSV) if args.resume else set()
    remaining = [t for t in trials if (t[1], t[0]) not in completed]

    print(f'\n{"="*70}\n  Experiment B (clutter) -- inter-object collision outcome\n'
          f'  Trials: {len(trials)}   Remaining: {len(remaining)}\n  Output: {OUTPUT_CSV}\n{"="*70}\n')

    objects, _ = clutter_layout()
    build_scene_xml(objects, SCENE_XML)
    build_scene_xml(objects, FG_SCENE_XML, template_path=FLOATING_GRIPPER_TEMPLATE)
    print(f'Clutter perception scene: {SCENE_XML}')
    print(f'Clutter floating-gripper scene: {FG_SCENE_XML}')

    sc.configure_determinism()
    print('Loading Contact-GraspNet...')
    estimator = load_cgn()
    print('CGN ready.\n')

    mode = 'a' if args.resume and os.path.exists(OUTPUT_CSV) else 'w'
    csv_file = open(OUTPUT_CSV, mode, newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if mode == 'w':
        writer.writeheader()
    csv_file.flush()

    t0 = time.time()
    for done_count, (trial_id, target, sigma_d, rho, phi, theta, seed) in enumerate(remaining, 1):
        t_start = time.time()
        print(f'[{done_count:>4}/{len(remaining)}] target={target:8s} trial={trial_id:>4} '
              f'sigma_d={sigma_d:.4f} rho={rho:.2f} phi={phi:.0f} theta={theta:.0f} seed={seed}',
              end='  ... ', flush=True)
        try:
            row = run_trial(trial_id, target, sigma_d, rho, phi, theta, seed, estimator)
        except Exception as e:
            row = {f: '' for f in CSV_FIELDS}
            row.update(trial_id=trial_id, target_object=target, sigma_d=sigma_d, rho=rho,
                       phi=phi, theta=theta, seed=seed, error=str(e)[:120])
            print(f'ERROR: {e}')
            traceback.print_exc()
        else:
            s = 'SUCCESS' if row['success'] else 'FAILURE'
            col = ' [COLLISION]' if row.get('collision_with_neighbor') else ''
            dt = time.time() - t_start
            eta = (time.time() - t0) / done_count * (len(remaining) - done_count)
            print(f'{s}{col}  score={row.get("q_grasp","?")}  {dt:.1f}s  ETA={eta/60:.1f}min')

        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    elapsed = (time.time() - t0) / 60.
    print(f'\n{"="*70}\n  Done. {len(remaining)} trials in {elapsed:.1f} minutes.\n'
          f'  Results: {OUTPUT_CSV}\n{"="*70}\n')


if __name__ == '__main__':
    main()
