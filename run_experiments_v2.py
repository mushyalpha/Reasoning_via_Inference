"""
run_experiments_v2.py
======================
Experiment A (isolated objects) -- redesign in response to preliminary
marking feedback:

  Marker A: a single (rotationally-symmetric) cylinder cannot support
  general causal claims about perception degradation -> grasp success;
  geometry must vary. Minimum defensible set: 3-5 objects spanning
  box/cuboid, curved/cylinder, and irregular/asymmetric shapes.

  Marker B: object geometry is a potential *moderator* of the
  degradation -> failure relationship. Testing >=1 additional object
  lets the thesis show either (a) the causal structure is geometry-
  invariant, or (b) it interacts with geometry -- both are informative;
  a single object cannot distinguish either.

Three objects (see object_specs.py): cylinder (kept, primitive),
box (YCB GelatinBox mesh), mustard bottle (YCB MustardBottle mesh,
irregular/asymmetric).

Grid changes vs. the original run_experiments.py (agreed redesign,
see RIGOUR_LEDGER.md Stage 8/11 and this session's design notes):
  - sigma_d densified around the 0.005->0.02 collapse:
      [0.000, 0.0025, 0.005, 0.010, 0.015, 0.020, 0.040]   (was 4 points)
  - phi densified around the 60 deg irreducible-failure boundary:
      [30, 45, 50, 55, 60, 65]                             (was 3 points)
  - theta unchanged: [0, 45, 90]
  - rho unchanged: [1.0, 0.75, 0.5, 0.25], but now DETERMINISTIC
    (sim_common.deterministic_downsample_idx) instead of rng.choice --
    removes the second stochastic process flagged in RIGOUR_LEDGER
    Stage 7.
  - seeds per condition: 5 (was 3) for narrower CIs (RIGOUR_LEDGER
    Stage 11).
  - CGN's own internal randomness is seeded per trial
    (sim_common.seed_cgn_global_random) for full trial reproducibility.

Full grid per object: 7 x 4 x 6 x 3 x 5 = 2520 trials.
Full Experiment A (3 objects):                     7560 trials.

Outcome-variable fix (response to a second round of preliminary marking
feedback, 4 Aug -- see RIGOUR_LEDGER.md and MARKER_FEEDBACK.md):

  Marker B pointed out that `success = 1 if e_pose < GRASP_RADIUS`
  (i) is not grasp success (it measures localisation accuracy only),
  (ii) is circular w.r.t. the SCM (success is a deterministic function
  of a mediator -- e_pose -- already in the model, so q_grasp/n_grasps
  become decorative and any SCM-vs-LLM comparison is rigged in the
  SCM's favour), and (iii) the 6.5cm threshold was tuned for a nice
  success-rate spread, not physics (real gripper tolerance ~4mm).

  Marker A independently recommended the fix: the ACRONYM / 6-DOF-
  GraspNet floating-gripper protocol. Teleport a free Panda gripper
  (no arm) directly to CGN's predicted 6-DoF pose (position AND
  orientation -- the orientation was previously being discarded after
  cam_to_world, despite being available), check collision-free with
  fingers open, close the fingers, then apply a fixed lift+shake
  disturbance under gravity. Success = the object is still held
  (within footprint_radius of the gripper) after the disturbance.

  This is implemented in sim_common.run_floating_gripper_test(), run in
  a separate arm-free scene (floating_gripper_template.xml, built via
  object_specs.FLOATING_GRIPPER_TEMPLATE) -- the object's settled pose
  from the perception scene is carried over via sim_common.set_object_pose(),
  and the gripper is a MuJoCo *mocap* body (kinematically teleported,
  immune to contact reaction forces) so "close fingers while held in
  place" and "shake while gravity + friction act on the object" both
  fall out of the same primitive with no custom PD controller needed.
  `success` is now a genuinely physical, independent outcome -- no
  longer a function of e_pose/q_grasp/n_grasps by construction.

Usage:
    python run_experiments_v2.py --object cylinder box mustard   # full grid, all objects (default)
    python run_experiments_v2.py --object box                    # one object only
    python run_experiments_v2.py --lean                           # reduced grid, faster first pass
    python run_experiments_v2.py --test                           # tiny smoke test
    python run_experiments_v2.py --resume                         # skip completed (object, trial_id) rows
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
from object_specs import (OBJECT_SPECS, OBJECT_NAMES, build_scene_xml, spawn_pos,
                           centroid_world, FLOATING_GRIPPER_TEMPLATE)

RESULTS_DIR = os.path.join(_PROJECT, 'results')
SCENES_DIR = os.path.join(_PROJECT, 'generated_scenes')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'experiment_results_v2.csv')
CKPT_DIR = os.path.join(_CGN_REPO, 'checkpoints', 'contact_graspnet')

IMG_W, IMG_H = 640, 480
CAM_RADIUS = 0.8
# Parked arm position used while the object settles / camera captures the
# scene. The original fixed ARM_HOME_ANGLES pose was calibrated against the
# short cylinder and physically overlaps taller objects (e.g. the ~19cm
# mustard bottle) -- see sim_common.disable_arm_collision docstring. Instead
# of re-deriving per-object home angles, the arm is IK-moved to a safe
# parked Cartesian position (clear of every object's height) before the
# object is allowed to settle, with arm collision disabled during that
# transient move so it doesn't fight residual overlap from the raw home pose.
PARK_POS = np.array([0.5, -0.35, 0.75])

# Floating-gripper shake-test parameters (Marker A's fix; see module
# docstring). Fixed, deterministic -- no RNG, so this introduces no new
# stochastic dimension into the grid.
CLOSE_STEPS = 400
SHAKE_STEPS = 600
LIFT_HEIGHT = 0.15
SHAKE_AMPLITUDE = 0.03

# ── Expanded experimental grid ──────────────────────────────────────────
SIGMA_D_VALS = [0.000, 0.0025, 0.005, 0.010, 0.015, 0.020, 0.040]
RHO_VALS = [1.00, 0.75, 0.50, 0.25]
PHI_VALS = [30., 45., 50., 55., 60., 65.]
THETA_VALS = [0., 45., 90.]
N_REPEATS = 5

LEAN_SIGMA_D_VALS = [0.000, 0.005, 0.010, 0.015, 0.020, 0.040]
LEAN_PHI_VALS = [30., 45., 50., 55., 60., 65.]
LEAN_N_REPEATS = 3

CSV_FIELDS = [
    'trial_id', 'object', 'sigma_d', 'rho', 'phi', 'theta', 'seed',
    'C_pc', 'q_grasp', 'e_pose', 'n_grasps',
    'collision_free', 'success', 'final_xy_offset', 'final_lift',
    'obj_z_final', 'error'
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
    tmp = os.path.join(_PROJECT, f'_tmp_batch_{os.getpid()}.npz')
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


def execute_grasp_floating(fg_scene_xml, spec, obj_pos, obj_quat, grasp_pos, grasp_quat):
    """
    Marker A's fix: floating-gripper shake test instead of full-arm IK.

    Loads a fresh arm-free scene, re-seeds the target object at the
    exact pose it settled to in the perception scene (obj_pos/obj_quat),
    teleports the mocap gripper to CGN's full 6-DoF predicted pose, and
    runs sim_common.run_floating_gripper_test() (collision check ->
    close fingers -> lift+shake -> check still held).
    """
    fg_model = mujoco.MjModel.from_xml_path(fg_scene_xml)
    fg_data = mujoco.MjData(fg_model)
    sc.set_object_pose(fg_model, fg_data, spec['body_name'], obj_pos, obj_quat)
    mujoco.mj_forward(fg_model, fg_data)
    sc.settle(fg_model, fg_data, 30)

    return sc.run_floating_gripper_test(
        fg_model, fg_data, spec['body_name'], grasp_pos, grasp_quat,
        footprint_radius=spec['footprint_radius'],
        close_steps=CLOSE_STEPS, shake_steps=SHAKE_STEPS,
        lift_height=LIFT_HEIGHT, shake_amplitude=SHAKE_AMPLITUDE)


def run_trial(trial_id, object_name, scene_xml, fg_scene_xml, sigma_d, rho, phi, theta, seed, estimator):
    spec = OBJECT_SPECS[object_name]
    rng = np.random.default_rng(seed)
    sc.seed_cgn_global_random(seed)

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    target_pos = spawn_pos(spec)
    sc.set_home_pose(model, data, {spec['body_name']: target_pos})
    mujoco.mj_forward(model, data)

    # Move the arm clear of the object, then let the object settle under
    # gravity/table contact only, with arm collision disabled throughout
    # this transient (see PARK_POS / disable_arm_collision docstrings).
    arm_saved = sc.disable_arm_collision(model)
    sc.ik_move_to(model, data, PARK_POS, max_steps=1000, tol=0.02)
    sc.settle(model, data, 200)
    sc.restore_arm_collision(model, arm_saved)
    mujoco.mj_forward(model, data)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
    centroid = centroid_world(spec, data.xpos[obj_id])

    sc.set_camera(model, phi, theta, CAM_RADIUS, np.array(centroid))
    mujoco.mj_forward(model, data)

    depth, K, seg_map = sc.render_depth_seg(
        model, data, {spec['body_name']: 1}, sigma_d=sigma_d, rng=rng,
        img_w=IMG_W, img_h=IMG_H)
    C_pc = float(seg_map.sum()) / (IMG_W * IMG_H)

    pred_grasps, scores = run_cgn(depth, K, seg_map, estimator, rho=rho, rng=rng)
    n_grasps = sum(len(scores[k]) for k in scores)

    row_base = dict(trial_id=trial_id, object=object_name, sigma_d=sigma_d, rho=rho,
                     phi=phi, theta=theta, seed=seed)

    if n_grasps == 0:
        return {**row_base, 'C_pc': C_pc, 'q_grasp': None, 'e_pose': None,
                'n_grasps': 0, 'collision_free': None, 'success': 0,
                'final_xy_offset': None, 'final_lift': None,
                'obj_z_final': None, 'error': 'no_grasps'}

    pose_cam, q_grasp, _ = sc.best_grasp_overall(pred_grasps, scores)
    pose_world = sc.cam_to_world(pose_cam, model, data)
    grasp_pos = pose_world[:3, 3]
    # Full 6-DoF pose used now, not just position (previously discarded --
    # see module docstring / Marker B's "you'd been dropping it" note).
    grasp_quat = sc.rot_matrix_to_quat(pose_world[:3, :3])

    obj_pos = data.xpos[obj_id].copy()
    obj_quat = data.xquat[obj_id].copy()
    centroid_now = centroid_world(spec, obj_pos)
    e_pose = float(np.linalg.norm(np.array(centroid_now[:2]) - grasp_pos[:2]))

    result = execute_grasp_floating(fg_scene_xml, spec, obj_pos, obj_quat, grasp_pos, grasp_quat)

    return {**row_base, 'C_pc': round(C_pc, 5), 'q_grasp': round(q_grasp, 5),
            'e_pose': round(e_pose, 5), 'n_grasps': n_grasps,
            'collision_free': int(result['collision_free']),
            'success': int(result['success']),
            'final_xy_offset': result['final_xy_offset'],
            'final_lift': result['final_lift'],
            'obj_z_final': result['obj_z_final'], 'error': ''}


def load_completed(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((row['object'], int(row['trial_id'])))
    return done


def build_grid(sigma_d_vals, rho_vals, phi_vals, theta_vals, n_repeats):
    trials = []
    for i, (sigma_d, rho, phi, theta, rep) in enumerate(
            itertools.product(sigma_d_vals, rho_vals, phi_vals, theta_vals, range(n_repeats))):
        seed = rep * 10000 + i
        trials.append((i, sigma_d, rho, phi, theta, seed))
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', nargs='+', choices=OBJECT_NAMES, default=OBJECT_NAMES)
    parser.add_argument('--lean', action='store_true', help='Reduced grid, faster first pass')
    parser.add_argument('--test', action='store_true', help='Tiny smoke test grid')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.test:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_repeats = \
            [0.0, 0.02], [1.0, 0.5], [45.], [0., 45.], 1
    elif args.lean:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_repeats = \
            LEAN_SIGMA_D_VALS, RHO_VALS, LEAN_PHI_VALS, THETA_VALS, LEAN_N_REPEATS
    else:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_repeats = \
            SIGMA_D_VALS, RHO_VALS, PHI_VALS, THETA_VALS, N_REPEATS

    trials = build_grid(sigma_d_vals, rho_vals, phi_vals, theta_vals, n_repeats)
    completed = load_completed(OUTPUT_CSV) if args.resume else set()

    print(f'\n{"="*70}\n  Experiment A (isolated objects) -- v2\n'
          f'  Objects: {args.object}\n  Trials/object: {len(trials)}   Total: {len(trials)*len(args.object)}\n'
          f'  Output: {OUTPUT_CSV}\n{"="*70}\n')

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
    total_done = 0
    grand_total = sum(1 for obj in args.object for (tid, *_r) in trials
                       if (obj, tid) not in completed)

    for object_name in args.object:
        scene_xml = os.path.join(SCENES_DIR, f'scene_{object_name}.xml')
        build_scene_xml(object_name, scene_xml)
        fg_scene_xml = os.path.join(SCENES_DIR, f'scene_{object_name}_floating_gripper.xml')
        build_scene_xml(object_name, fg_scene_xml, template_path=FLOATING_GRIPPER_TEMPLATE)
        print(f'--- Object: {object_name} ({OBJECT_SPECS[object_name]["label"]}) ---')
        print(f'    Perception scene: {scene_xml}')
        print(f'    Floating-gripper scene: {fg_scene_xml}')

        for trial_id, sigma_d, rho, phi, theta, seed in trials:
            if (object_name, trial_id) in completed:
                continue
            t_start = time.time()
            print(f'[{total_done+1:>5}/{grand_total}] obj={object_name:8s} trial={trial_id:>4} '
                  f'sigma_d={sigma_d:.4f} rho={rho:.2f} phi={phi:.0f} theta={theta:.0f} seed={seed}',
                  end='  ... ', flush=True)
            try:
                row = run_trial(trial_id, object_name, scene_xml, fg_scene_xml,
                                 sigma_d, rho, phi, theta, seed, estimator)
            except Exception as e:
                row = {f: '' for f in CSV_FIELDS}
                row.update(trial_id=trial_id, object=object_name, sigma_d=sigma_d, rho=rho,
                           phi=phi, theta=theta, seed=seed, error=str(e)[:120])
                print(f'ERROR: {e}')
                traceback.print_exc()
            else:
                s = 'SUCCESS' if row['success'] else 'FAILURE'
                dt = time.time() - t_start
                total_done += 1
                eta = (time.time() - t0) / total_done * (grand_total - total_done)
                print(f'{s}  z={row.get("obj_z_final","?")}  score={row.get("q_grasp","?")}  '
                      f'{dt:.1f}s  ETA={eta/60:.1f}min')

            writer.writerow(row)
            csv_file.flush()

    csv_file.close()
    elapsed = (time.time() - t0) / 60.
    print(f'\n{"="*70}\n  Done. {total_done} trials in {elapsed:.1f} minutes.\n'
          f'  Results: {OUTPUT_CSV}\n{"="*70}\n')


if __name__ == '__main__':
    main()
