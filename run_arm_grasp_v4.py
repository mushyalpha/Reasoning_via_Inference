"""
run_arm_grasp_v4.py
===================
Full-arm CGN grasp attempts (NEW file — does not modify v2/v3 pipelines).

Clean-condition cylinder protocol aligned with ARM_IK_ATTEMPT_LOG.md:
  - φ=55°, θ=45°, σ_d=0, ρ=1
  - Top-k CGN proposals + open-finger collision filter (hand/link)
  - 6-DoF IK to grasp (HOME seed; direct grasp fallback)
  - Arm *link* collisions ghosted during execution (fingers active)
  - Clamp + slow ctrl-only lift attempt

Logs both lift success and intermediate reach/contact metrics so the
thesis can report due diligence even when lift rates stay below the
floating-gripper baseline (known contact-mechanics limitation).

Usage:
    python3 run_arm_grasp_v4.py --test
    python3 run_arm_grasp_v4.py --seeds 25
"""

import os
import sys
import time
import argparse
import csv
import traceback
import numpy as np
import mujoco

_PROJECT = os.path.dirname(os.path.abspath(__file__))
_CGN_REPO = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
_CGN_SRC = os.path.join(_CGN_REPO, 'contact_graspnet_pytorch')
for _p in (_CGN_REPO, _CGN_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contact_grasp_estimator import GraspEstimator
import config_utils
from checkpoints import CheckpointIO
from data import load_available_input_data
import torch

import sim_common as sc
import sim_common_v3 as sc3
import sim_arm_v4 as arm
from object_specs import OBJECT_SPECS, build_scene_xml, spawn_pos, centroid_world

RESULTS = os.path.join(_PROJECT, 'results')
SCENES = os.path.join(_PROJECT, 'generated_scenes')
OUT_CSV = os.path.join(RESULTS, 'experiment_results_arm_v4.csv')
CKPT = os.path.join(_CGN_REPO, 'checkpoints', 'contact_graspnet')
LOG = os.path.join(_PROJECT, 'ARM_IK_ATTEMPT_LOG.md')

IMG_W, IMG_H = 640, 480
CAM_R = 0.8
PHI, THETA = 55., 45.
TOP_K = 20

CSV_FIELDS = [
    'trial_id', 'object', 'sigma_d', 'rho', 'phi', 'theta', 'seed',
    'n_grasps', 'selected_rank', 'n_rejected', 'q_grasp',
    'ik_grasp_ok', 'pos_err_grasp', 'ori_err_grasp', 'used_direct_grasp',
    'collision_at_grasp', 'got_contact_on_close',
    'success', 'final_lift', 'final_xy_offset', 'failure_mode', 'error',
]


def load_cgn():
    cfg = config_utils.load_config(CKPT, batch_size=1, arg_configs=[])
    est = GraspEstimator(cfg)
    CheckpointIO(checkpoint_dir=os.path.join(CKPT, 'checkpoints'),
                 model=est.model).load('model.pt')
    est.model.eval()
    return est


def run_cgn(depth, K, seg, estimator, rho=1.0):
    tmp = os.path.join(_PROJECT, f'_tmp_arm_v4_{os.getpid()}.npz')
    np.savez(tmp, depth=depth, K=K, seg=seg)
    segmap, rgb, depth_in, cam_K, _, _ = load_available_input_data(tmp, K=None)
    pc_full, pc_segs, _ = estimator.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb,
        skip_border_objects=False, z_range=[0.1, 2.0])
    if rho < 1.0 and len(pc_full) > 0:
        idx = sc.deterministic_downsample_idx(len(pc_full), rho)
        pc_full = pc_full[idx]
        for k in pc_segs:
            if len(pc_segs[k]) > 0:
                pc_segs[k] = pc_segs[k][sc.deterministic_downsample_idx(len(pc_segs[k]), rho)]
    with torch.no_grad():
        pred, scores, _, _ = estimator.predict_scene_grasps(
            pc_full, pc_segments=pc_segs, local_regions=True,
            filter_grasps=True, forward_passes=1)
    if os.path.exists(tmp):
        os.remove(tmp)
    return pred, scores


def append_log(line):
    with open(LOG, 'a') as f:
        f.write(line.rstrip() + '\n')


def run_trial(trial_id, seed, estimator, top_k):
    spec = OBJECT_SPECS['cylinder']
    rng = np.random.default_rng(seed)
    sc.seed_cgn_global_random(seed)

    xml = os.path.join(SCENES, 'scene_cylinder.xml')
    build_scene_xml('cylinder', xml)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    arm.boost_fingertip_friction(model)

    sc.set_home_pose(model, data, {spec['body_name']: spawn_pos(spec)})
    data.qpos[sc.arm_qpos_adr(model)] = sc.ARM_HOME_ANGLES
    data.ctrl[:7] = sc.ARM_HOME_ANGLES
    mujoco.mj_forward(model, data)

    # Match v3 perception protocol: park arm clear, then render
    full_saved = sc.disable_arm_collision(model)
    sc.ik_move_to(model, data, np.array([0.5, -0.35, 0.75]), max_steps=1500, tol=0.02)
    sc.settle(model, data, 200)
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['body_name'])
    centroid = centroid_world(spec, data.xpos[obj_id])
    sc.set_camera(model, PHI, THETA, CAM_R, np.array(centroid))
    mujoco.mj_forward(model, data)
    depth, K, seg, seg_empty = sc.render_depth_seg(
        model, data, {spec['body_name']: 1}, sigma_d=0.0, rng=rng,
        img_w=IMG_W, img_h=IMG_H)
    sc.restore_arm_collision(model, full_saved)

    base = dict(trial_id=trial_id, object='cylinder', sigma_d=0.0, rho=1.0,
                phi=PHI, theta=THETA, seed=seed)

    if seg_empty:
        return {**base, 'n_grasps': 0, 'selected_rank': None, 'n_rejected': None,
                'q_grasp': None, 'ik_grasp_ok': 0, 'pos_err_grasp': None,
                'ori_err_grasp': None, 'used_direct_grasp': None,
                'collision_at_grasp': None, 'got_contact_on_close': None,
                'success': 0, 'final_lift': None, 'final_xy_offset': None,
                'failure_mode': 'no_visible_object', 'error': ''}

    pred, scores = run_cgn(depth, K, seg, estimator, rho=1.0)
    n_grasps = sum(len(scores[k]) for k in scores)
    if n_grasps == 0:
        return {**base, 'n_grasps': 0, 'selected_rank': None, 'n_rejected': None,
                'q_grasp': None, 'ik_grasp_ok': 0, 'pos_err_grasp': None,
                'ori_err_grasp': None, 'used_direct_grasp': None,
                'collision_at_grasp': None, 'got_contact_on_close': None,
                'success': 0, 'final_lift': None, 'final_xy_offset': None,
                'failure_mode': 'no_grasps', 'error': ''}

    # Reset arm to HOME for execution
    data.qpos[sc.arm_qpos_adr(model)] = sc.ARM_HOME_ANGLES
    data.ctrl[:7] = sc.ARM_HOME_ANGLES
    mujoco.mj_forward(model, data)

    ranked = sc3.ranked_grasps_overall(pred, scores, top_k=top_k)
    n_rejected = 0
    selected = None
    for rank, (pose_cam, score, _seg) in enumerate(ranked):
        pose_w = sc.cam_to_world(pose_cam, model, data)
        gpos, gquat = pose_w[:3, 3].copy(), sc.rot_matrix_to_quat(pose_w[:3, :3])
        # Probe open-finger collision with link-ghosted arm teleported via IK
        # Cheap probe: floating-style contact check after IK to pose
        link_saved = arm.disable_arm_link_collision(model)
        arm.open_gripper_ctrl(model, data)
        ee, eq = arm.hand_grasp_to_ee_target(gpos, gquat)
        ok = arm.ik_move_to_pose_6d(model, data, ee, eq, max_steps=2000, pos_tol=0.012, ori_tol=0.2)
        data.ctrl[:7] = data.qpos[sc.arm_qpos_adr(model)]
        mujoco.mj_forward(model, data)
        contacted = arm.probe_gripper_contacts(model, data)
        bad = [b for b in contacted if b.startswith('link') or b == 'hand']
        arm.restore_geom_collision(model, link_saved)
        # reset to HOME between probes
        data.qpos[sc.arm_qpos_adr(model)] = sc.ARM_HOME_ANGLES
        data.ctrl[:7] = sc.ARM_HOME_ANGLES
        mujoco.mj_forward(model, data)
        if (not ok) or bad:
            n_rejected += 1
            continue
        selected = (rank, gpos, gquat, score)
        break

    if selected is None:
        return {**base, 'n_grasps': n_grasps, 'selected_rank': None,
                'n_rejected': n_rejected, 'q_grasp': None,
                'ik_grasp_ok': 0, 'pos_err_grasp': None, 'ori_err_grasp': None,
                'used_direct_grasp': None, 'collision_at_grasp': 1,
                'got_contact_on_close': None, 'success': 0,
                'final_lift': None, 'final_xy_offset': None,
                'failure_mode': 'pregrasp_collision', 'error': ''}

    rank, gpos, gquat, score = selected
    # Re-settle object pose reference
    obj_pos = data.xpos[obj_id].copy()
    result = arm.run_arm_approach_grasp_lift(
        model, data, spec['body_name'], gpos, gquat,
        footprint_radius=spec['footprint_radius'],
        standoff=0.05, n_approach=6, lift_height=0.12,
        check_pregrasp_collision=False)

    return {**base, 'n_grasps': n_grasps, 'selected_rank': rank,
            'n_rejected': n_rejected, 'q_grasp': round(float(score), 5),
            'ik_grasp_ok': int(bool(result.get('ik_grasp_ok'))),
            'pos_err_grasp': result.get('pos_err_grasp'),
            'ori_err_grasp': result.get('ori_err_grasp'),
            'used_direct_grasp': int(bool(result.get('used_direct_grasp'))),
            'collision_at_grasp': int(bool(result.get('collision_at_grasp'))),
            'got_contact_on_close': int(bool(result.get('got_contact_on_close'))),
            'success': int(bool(result.get('success'))),
            'final_lift': result.get('final_lift'),
            'final_xy_offset': result.get('final_xy_offset'),
            'failure_mode': result.get('failure_mode'), 'error': ''}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=25)
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--output', default=OUT_CSV)
    args = parser.parse_args()
    n = 2 if args.test else args.seeds
    os.makedirs(RESULTS, exist_ok=True)

    print(f'\n{"=" * 70}\n  Arm IK grasp v4 — cylinder clean\n'
          f'  φ={PHI} θ={THETA} σ_d=0 ρ=1 top_k={args.top_k} seeds={n}\n'
          f'  Output: {args.output}\n{"=" * 70}\n')
    append_log(f'\n### Arm batch run — {time.strftime("%Y-%m-%d %H:%M")}')
    append_log(f'- Starting {n} seeds, top_k={args.top_k}, φ={PHI}, θ={THETA}')

    sc.configure_determinism()
    est = load_cgn()
    f = open(args.output, 'w', newline='')
    w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()

    t0 = time.time()
    n_ok = n_reach = 0
    for i in range(n):
        seed = 400000 + i * 10007
        print(f'[{i+1}/{n}] seed={seed} ...', end=' ', flush=True)
        try:
            row = run_trial(i, seed, est, args.top_k)
        except Exception as e:
            row = {k: '' for k in CSV_FIELDS}
            row.update(trial_id=i, object='cylinder', seed=seed, error=str(e)[:120],
                       success=0, failure_mode='error')
            traceback.print_exc()
        w.writerow(row)
        f.flush()
        if row.get('success'):
            n_ok += 1
        if row.get('ik_grasp_ok'):
            n_reach += 1
        print(f'{row.get("failure_mode")} reach={row.get("ik_grasp_ok")} '
              f'lift={row.get("final_lift")} rate_lift={n_ok/(i+1):.0%} rate_reach={n_reach/(i+1):.0%}')

    f.close()
    msg = (f'- Done {n} trials in {(time.time()-t0)/60:.1f} min. '
           f'Lift success {n_ok}/{n}={n_ok/n:.0%}. '
           f'IK reach {n_reach}/{n}={n_reach/n:.0%}. CSV={args.output}')
    print('\n' + msg)
    append_log(msg)


if __name__ == '__main__':
    main()
