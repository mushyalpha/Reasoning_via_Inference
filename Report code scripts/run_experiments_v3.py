"""
run_experiments_v3.py
======================
v3 copy of run_experiments_v2.py — do NOT edit the original v2 files.

Purpose
-------
Report clean / easy-condition grasp success rates with a high-probability
selection protocol:

  1. Clean perception: σ_d = 0, ρ = 1.0, CGN-friendly camera poses
     (default φ=55°, θ=45° — not φ=30°, which historically yields ~80%
     no_grasps on the cylinder). Optional --clean-grid for φ∈{50,55,60},
     θ∈{0,45}.
  2. Top-k + collision filter: rank CGN proposals by score, reject any
     open-finger pose that contacts object/table, execute the first that
     clears (ACRONYM / CGN-style gate). Default top_k = 10.
  3. Lighter success: clamp + vertical lift only (no XY shake). Success =
     object stays in the gripper footprint and lifts ≥ 40% of the commanded
     15 cm (i.e. clear of the table).

Many seeds (default 100) per clean cell so the reported rate is stable.

Imports sim_common_v3 (not sim_common). Writes to
results/experiment_results_v3_clean.csv by default.

Usage:
    python run_experiments_v3.py --object cylinder
    python run_experiments_v3.py --object cylinder box mustard --seeds 100
    python run_experiments_v3.py --clean-grid --object cylinder   # φ∈{30,45}, θ∈{0,45}
    python run_experiments_v3.py --test
    python run_experiments_v3.py --resume
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

import sim_common_v3 as sc
from object_specs import (OBJECT_SPECS, OBJECT_NAMES, build_scene_xml, spawn_pos,
                           centroid_world, FLOATING_GRIPPER_TEMPLATE)

RESULTS_DIR = os.path.join(_PROJECT, 'results')
SCENES_DIR = os.path.join(_PROJECT, 'generated_scenes')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'experiment_results_v3_clean.csv')
CKPT_DIR = os.path.join(_CGN_REPO, 'checkpoints', 'contact_graspnet')

IMG_W, IMG_H = 640, 480
CAM_RADIUS = 0.8
PARK_POS = np.array([0.5, -0.35, 0.75])

CLOSE_STEPS = 400
LIFT_STEPS = 400
LIFT_HEIGHT = 0.15
XY_TOLERANCE_MARGIN = 0.03
TOP_K_DEFAULT = 20
N_SEEDS_DEFAULT = 25

# Single clean cell (default).
# NOTE: φ=30° looks "easy" for proximity localisation but is a *bad* CGN
# proposal viewpoint for the cylinder (v2: ~80% no_grasps at σ_d=0,ρ=1).
# Prefer mid/high elevation where CGN reliably returns contacts (v2 best
# clean cells: φ=55°/θ=45°, φ=60°/θ=0° — ~80–100% has_grasps, ~40% shake
# success before top-k filtering).
CLEAN_SIGMA_D = [0.0]
CLEAN_RHO = [1.0]
CLEAN_PHI = [55.]
CLEAN_THETA = [45.]

# Small clean camera grid: viewpoints that historically yield proposals
CLEAN_GRID_PHI = [50., 55., 60.]
CLEAN_GRID_THETA = [0., 45.]

# Optional fallback used by --easy-legacy (old mistaken "easy" cell)
LEGACY_EASY_PHI = [30.]
LEGACY_EASY_THETA = [0.]

CSV_FIELDS = [
    'trial_id', 'object', 'sigma_d', 'rho', 'phi', 'theta', 'seed',
    'seg_empty', 'C_pc', 'cube_size', 'q_grasp', 'e_pose', 'n_grasps',
    'top_k', 'selected_rank', 'n_candidates_tried', 'n_rejected_collision',
    'collision_free', 'contacted_bodies', 'success',
    'final_xy_offset', 'final_lift', 'obj_z_final', 'failure_mode', 'error',
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
    tmp = os.path.join(_PROJECT, f'_tmp_batch_v3_{os.getpid()}.npz')
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

    cube_sizes = getattr(estimator, '_last_cube_sizes', {}) or {}
    cube_size = cube_sizes.get(1, next(iter(cube_sizes.values()), None))

    if os.path.exists(tmp):
        os.remove(tmp)
    return pred_grasps, scores, cube_size


def execute_grasp_topk_lift(fg_scene_xml, spec, obj_pos, obj_quat,
                            ranked_world_grasps):
    """Load floating-gripper scene and run top-k collision-filtered lift."""
    fg_model = mujoco.MjModel.from_xml_path(fg_scene_xml)
    fg_data = mujoco.MjData(fg_model)
    return sc.execute_topk_collision_filtered_lift(
        fg_model, fg_data, spec['body_name'], obj_pos, obj_quat,
        ranked_world_grasps,
        footprint_radius=spec['footprint_radius'],
        close_steps=CLOSE_STEPS, lift_steps=LIFT_STEPS,
        lift_height=LIFT_HEIGHT,
        xy_tolerance_margin=XY_TOLERANCE_MARGIN)


def _empty_exec_fields():
    return dict(
        selected_rank=None, n_candidates_tried=None,
        n_rejected_collision=None, collision_free=None, contacted_bodies=None,
        success=0, final_xy_offset=None, final_lift=None, obj_z_final=None)


def run_trial(trial_id, object_name, scene_xml, fg_scene_xml,
              sigma_d, rho, phi, theta, seed, estimator, top_k):
    spec = OBJECT_SPECS[object_name]
    rng = np.random.default_rng(seed)
    sc.seed_cgn_global_random(seed)

    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)

    target_pos = spawn_pos(spec)
    sc.set_home_pose(model, data, {spec['body_name']: target_pos})
    mujoco.mj_forward(model, data)

    arm_saved = sc.disable_arm_collision(model)
    sc.ik_move_to(model, data, PARK_POS, max_steps=1000, tol=0.02)
    sc.settle(model, data, 200)
    sc.restore_arm_collision(model, arm_saved)
    mujoco.mj_forward(model, data)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
    centroid = centroid_world(spec, data.xpos[obj_id])

    sc.set_camera(model, phi, theta, CAM_RADIUS, np.array(centroid))
    mujoco.mj_forward(model, data)

    depth, K, seg_map, seg_empty = sc.render_depth_seg(
        model, data, {spec['body_name']: 1}, sigma_d=sigma_d, rng=rng,
        img_w=IMG_W, img_h=IMG_H)
    C_pc = float(seg_map.sum()) / (IMG_W * IMG_H)

    row_base = dict(trial_id=trial_id, object=object_name, sigma_d=sigma_d, rho=rho,
                    phi=phi, theta=theta, seed=seed, seg_empty=int(seg_empty),
                    top_k=top_k)

    if seg_empty:
        failure_mode = sc.classify_failure_mode(
            seg_empty=True, n_grasps=0, floating_gripper_result=None,
            footprint_radius=spec['footprint_radius'], lift_height=LIFT_HEIGHT,
            xy_tolerance_margin=XY_TOLERANCE_MARGIN)
        return {**row_base, 'C_pc': round(C_pc, 5), 'cube_size': None,
                'q_grasp': None, 'e_pose': None, 'n_grasps': 0,
                **_empty_exec_fields(),
                'failure_mode': failure_mode, 'error': 'no_visible_object'}

    pred_grasps, scores, cube_size = run_cgn(
        depth, K, seg_map, estimator, rho=rho, rng=rng)
    n_grasps = sum(len(scores[k]) for k in scores)

    if n_grasps == 0:
        failure_mode = sc.classify_failure_mode(
            seg_empty=False, n_grasps=0, floating_gripper_result=None,
            footprint_radius=spec['footprint_radius'], lift_height=LIFT_HEIGHT,
            xy_tolerance_margin=XY_TOLERANCE_MARGIN)
        return {**row_base, 'C_pc': round(C_pc, 5), 'cube_size': cube_size,
                'q_grasp': None, 'e_pose': None, 'n_grasps': 0,
                **_empty_exec_fields(),
                'failure_mode': failure_mode, 'error': 'no_grasps'}

    ranked_cam = sc.ranked_grasps_overall(pred_grasps, scores, top_k=top_k)
    ranked_world = []
    for pose_cam, score, _seg in ranked_cam:
        pose_world = sc.cam_to_world(pose_cam, model, data)
        grasp_pos = pose_world[:3, 3].copy()
        grasp_quat = sc.rot_matrix_to_quat(pose_world[:3, :3])
        ranked_world.append((grasp_pos, grasp_quat, score))

    # e_pose / q_grasp reported for the *selected* grasp after filtering;
    # until then use top-1 as the reference localisation error.
    top1_pos = ranked_world[0][0]
    obj_pos = data.xpos[obj_id].copy()
    obj_quat = data.xquat[obj_id].copy()
    centroid_now = centroid_world(spec, obj_pos)
    e_pose_top1 = float(np.linalg.norm(np.array(centroid_now[:2]) - top1_pos[:2]))

    result = execute_grasp_topk_lift(
        fg_scene_xml, spec, obj_pos, obj_quat, ranked_world)

    sel_rank = result.get('selected_rank')
    if sel_rank is not None and 0 <= sel_rank < len(ranked_world):
        sel_pos, _sel_quat, sel_score = ranked_world[sel_rank]
        e_pose = float(np.linalg.norm(np.array(centroid_now[:2]) - sel_pos[:2]))
        q_grasp = float(sel_score)
    else:
        e_pose = e_pose_top1
        q_grasp = float(ranked_world[0][2]) if ranked_world else None

    failure_mode = sc.classify_failure_mode(
        seg_empty=False, n_grasps=n_grasps, floating_gripper_result=result,
        footprint_radius=spec['footprint_radius'], lift_height=LIFT_HEIGHT,
        xy_tolerance_margin=XY_TOLERANCE_MARGIN)

    contacted = result.get('contacted_bodies') or []
    return {**row_base, 'C_pc': round(C_pc, 5), 'cube_size': cube_size,
            'q_grasp': None if q_grasp is None else round(q_grasp, 5),
            'e_pose': round(e_pose, 5), 'n_grasps': n_grasps,
            'selected_rank': sel_rank,
            'n_candidates_tried': result.get('n_candidates_tried'),
            'n_rejected_collision': result.get('n_rejected_collision'),
            'collision_free': int(bool(result.get('collision_free'))),
            'contacted_bodies': ';'.join(contacted) if contacted else None,
            'success': int(bool(result.get('success'))),
            'final_xy_offset': result.get('final_xy_offset'),
            'final_lift': result.get('final_lift'),
            'obj_z_final': result.get('obj_z_final'),
            'failure_mode': failure_mode, 'error': ''}


def load_completed(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((row['object'], int(row['trial_id'])))
    return done


def build_grid(sigma_d_vals, rho_vals, phi_vals, theta_vals, n_seeds):
    trials = []
    for i, (sigma_d, rho, phi, theta, rep) in enumerate(
            itertools.product(sigma_d_vals, rho_vals, phi_vals, theta_vals,
                              range(n_seeds))):
        seed = 300000 + rep * 10000 + i
        trials.append((i, sigma_d, rho, phi, theta, seed))
    return trials


def main():
    parser = argparse.ArgumentParser(
        description='v3 clean-condition top-k lift protocol')
    parser.add_argument('--object', nargs='+', choices=OBJECT_NAMES,
                        default=OBJECT_NAMES)
    parser.add_argument('--seeds', type=int, default=N_SEEDS_DEFAULT,
                        help=f'Repeats (seeds) per clean cell (default {N_SEEDS_DEFAULT})')
    parser.add_argument('--top-k', type=int, default=TOP_K_DEFAULT,
                        help=f'Top-k CGN proposals to collision-filter (default {TOP_K_DEFAULT})')
    parser.add_argument('--clean-grid', action='store_true',
                        help='Use φ∈{50,55,60}, θ∈{0,45} instead of single clean cell')
    parser.add_argument('--easy-legacy', action='store_true',
                        help='Use φ=30,θ=0 (historically high no_grasps — not recommended)')
    parser.add_argument('--test', action='store_true',
                        help='Tiny smoke: 2 seeds, top_k=5, φ=55 θ=45')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', default=OUTPUT_CSV,
                        help='CSV output path')
    args = parser.parse_args()

    if args.test:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_seeds = \
            [0.0], [1.0], [55.], [45.], 2
        top_k = min(5, args.top_k)
    elif args.clean_grid:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_seeds = \
            CLEAN_SIGMA_D, CLEAN_RHO, CLEAN_GRID_PHI, CLEAN_GRID_THETA, args.seeds
        top_k = args.top_k
    elif getattr(args, 'easy_legacy', False):
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_seeds = \
            CLEAN_SIGMA_D, CLEAN_RHO, LEGACY_EASY_PHI, LEGACY_EASY_THETA, args.seeds
        top_k = args.top_k
    else:
        sigma_d_vals, rho_vals, phi_vals, theta_vals, n_seeds = \
            CLEAN_SIGMA_D, CLEAN_RHO, CLEAN_PHI, CLEAN_THETA, args.seeds
        top_k = args.top_k

    trials = build_grid(sigma_d_vals, rho_vals, phi_vals, theta_vals, n_seeds)
    out_csv = args.output
    completed = load_completed(out_csv) if args.resume else set()

    print(f'\n{"=" * 70}\n  Clean-condition Experiment A -- v3\n'
          f'  Protocol: top-{top_k} collision filter + clamp/lift (no shake)\n'
          f'  Objects: {args.object}\n'
          f'  Grid: σ_d={sigma_d_vals} ρ={rho_vals} φ={phi_vals} θ={theta_vals}\n'
          f'  Seeds/cell: {n_seeds}   Trials/object: {len(trials)}   '
          f'Total: {len(trials) * len(args.object)}\n'
          f'  Output: {out_csv}\n{"=" * 70}\n')

    sc.configure_determinism()
    print('Loading Contact-GraspNet...')
    estimator = load_cgn()
    print('CGN ready.\n')

    mode = 'a' if args.resume and os.path.exists(out_csv) else 'w'
    csv_file = open(out_csv, mode, newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if mode == 'w':
        writer.writeheader()
    csv_file.flush()

    t0 = time.time()
    total_done = 0
    grand_total = sum(1 for obj in args.object for (tid, *_r) in trials
                      if (obj, tid) not in completed)
    n_success = 0

    for object_name in args.object:
        scene_xml = os.path.join(SCENES_DIR, f'scene_{object_name}.xml')
        build_scene_xml(object_name, scene_xml)
        fg_scene_xml = os.path.join(
            SCENES_DIR, f'scene_{object_name}_floating_gripper.xml')
        build_scene_xml(object_name, fg_scene_xml,
                        template_path=FLOATING_GRIPPER_TEMPLATE)
        print(f'--- Object: {object_name} ({OBJECT_SPECS[object_name]["label"]}) ---')

        for trial_id, sigma_d, rho, phi, theta, seed in trials:
            if (object_name, trial_id) in completed:
                continue
            t_start = time.time()
            print(f'[{total_done + 1:>5}/{grand_total}] obj={object_name:8s} '
                  f'trial={trial_id:>4} seed={seed}',
                  end='  ... ', flush=True)
            try:
                row = run_trial(
                    trial_id, object_name, scene_xml, fg_scene_xml,
                    sigma_d, rho, phi, theta, seed, estimator, top_k)
            except Exception as e:
                row = {f: '' for f in CSV_FIELDS}
                row.update(trial_id=trial_id, object=object_name, sigma_d=sigma_d,
                           rho=rho, phi=phi, theta=theta, seed=seed,
                           top_k=top_k, error=str(e)[:120])
                print(f'ERROR: {e}')
                traceback.print_exc()
            else:
                s = row.get('failure_mode', '?')
                if row.get('success'):
                    n_success += 1
                total_done += 1
                dt = time.time() - t_start
                eta = (time.time() - t0) / total_done * (grand_total - total_done)
                rate = n_success / total_done
                print(f'{s:<18} rank={row.get("selected_rank")} '
                      f'rej={row.get("n_rejected_collision")} '
                      f'lift={row.get("final_lift")} '
                      f'{dt:.1f}s  rate={rate:.1%}  ETA={eta / 60:.1f}min')

            writer.writerow(row)
            csv_file.flush()

    csv_file.close()
    elapsed = (time.time() - t0) / 60.
    rate = (n_success / total_done) if total_done else 0.0
    print(f'\n{"=" * 70}\n  Done. {total_done} trials in {elapsed:.1f} minutes.\n'
          f'  Successes: {n_success}/{total_done} = {rate:.1%}\n'
          f'  Results: {out_csv}\n{"=" * 70}\n')


if __name__ == '__main__':
    main()
