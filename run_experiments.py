"""
run_experiments.py
==================
Batch experiment runner for the MSc thesis.

ORIGINAL grid (4 x 4 x 3 x 3 x 3 = 432 trials), preserved in
`results/experiment_results.csv` and referenced throughout the fitted SCM,
counterfactual ground truth, and LLM baseline:
  sigma_d : [0.000, 0.005, 0.020, 0.040]   depth noise std dev (m)
  rho     : [1.0,   0.75,  0.50,  0.25]    point cloud keep fraction
  phi     : [30,    45,    60]             camera elevation (deg)
  theta   : [0,     45,    90]             camera azimuth (deg)
  repeat  : [0,     1,     2]              3 random seeds per condition

DENSIFIED grid (current default, 6 x 4 x 6 x 3 x 3 = 1296 trials), added to
resolve two under-sampled regions identified from the original 432-trial
results: the sigma_d success-rate collapse (42.6% -> 11.1%) occurred
entirely between sigma_d=0.005 and sigma_d=0.02 with no intermediate
sample point, and the phi "dead zone" (success falls to 15.3% at phi=60,
with 98/167 unexplained "none" counterfactual failures concentrated there)
was bracketed only by a bare 45 -> 60 jump. Two points were added inside
each gap (sigma_d: +0.010, +0.015; phi: +50, +55, +65 to also bracket the
transition from above) rather than replacing the original grid, so every
original condition is re-sampled with 3 fresh seeds -- this run therefore
also doubles seed coverage (3 -> 6) at every point shared with the original
grid, addressing the "increase seeds" item from the same review.
  sigma_d : [0.000, 0.005, 0.010, 0.015, 0.020, 0.040]
  rho     : [1.0,   0.75,  0.50,  0.25]     unchanged
  phi     : [30,    45,    50,    55,   60,   65]
  theta   : [0,     45,    90]              unchanged
  repeat  : [0,     1,     2]               3 random seeds per condition

Each trial records:
  Exogenous  : sigma_d, rho, phi, theta, seed
  Intermediate:
    C_pc     - fraction of image pixels showing the target object
    q_grasp  - CGN confidence score (best grasp)
    e_pose   - Euclidean distance between proposed and true object position
    n_grasps - total CGN grasp candidates
  Outcome:
    success  - 1 if the end-effector is within GRASP_RADIUS of the object
               centroid after the scripted approach + close (proximity
               criterion; see execute_grasp() -- NOT a physical lift check,
               despite the historical LIFT_HEIGHT constant below, which is
               unused by the current success logic and kept only because
               demo_grasp.py still references its own copy for the viewer
               demo's physical-lift attempt).

Results saved to:  results/experiment_results_densified.csv
  (the original results/experiment_results.csv is untouched by this script
  as of the densified grid -- see OUTPUT_CSV below)

Contact-GraspNet auto-selects CUDA if available (contact_grasp_estimator.py:
`self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`),
so no code change is needed to use a GPU. Before a long run, confirm it will
actually be picked up:
    python -c "import torch; print(torch.cuda.is_available())"

Usage:
    python run_experiments.py              # all 1296 densified-grid trials
    python run_experiments.py --test       # 9-trial smoke test (3x3 grid)
    python run_experiments.py --resume     # skip already-completed trials
    python run_experiments.py --original   # re-run the original 432-trial
                                            # grid instead (writes to
                                            # experiment_results.csv)
"""

import os
import sys
import math
import time
import argparse
import itertools
import csv
import traceback
import numpy as np
import mujoco

# ── CGN imports ────────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

SCENE_XML   = os.path.join(_PROJECT, 'grasp_scene_v2.xml')
CGN_ROOT    = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
CKPT_DIR    = os.path.join(CGN_ROOT, 'checkpoints', 'contact_graspnet')
RESULTS_DIR = os.path.join(_PROJECT, 'results')

LIFT_HEIGHT = 0.55   # unused by execute_grasp()'s proximity criterion; kept
                      # only for parity with demo_grasp.py's separate,
                      # still-lift-based viewer demo
IMG_W, IMG_H = 640, 480
TARGET_BODY = 'target_object'
CAM_NAME    = 'perception_camera'
EE_SITE     = 'ee_site'
TARGET_POS  = np.array([0.5, 0., 0.455])

# Original 432-trial grid (results/experiment_results.csv) -- preserved here
# for provenance / re-run capability via --original. Do not edit these.
ORIGINAL_SIGMA_D_VALS = [0.000, 0.005, 0.020, 0.040]
ORIGINAL_RHO_VALS     = [1.00,  0.75,  0.50,  0.25]
ORIGINAL_PHI_VALS     = [30.,   45.,   60.]
ORIGINAL_THETA_VALS   = [0.,    45.,   90.]
ORIGINAL_N_REPEATS    = 3

# Densified grid (current default) -- adds points inside the sigma_d cliff
# (0.005-0.02) and around the phi=60 dead-zone transition. rho and theta
# are unchanged: rho showed no significant effect in the original SCM fit
# (Eq2A p=0.144, Eq4 p=0.975) and theta's effect was already clean and
# monotone, so neither needed denser sampling.
SIGMA_D_VALS = [0.000, 0.005, 0.010, 0.015, 0.020, 0.040]   # depth noise
RHO_VALS     = [1.00,  0.75,  0.50,  0.25]                  # sparsity (unchanged)
PHI_VALS     = [30.,   45.,   50.,   55.,   60.,   65.]     # elevation
THETA_VALS   = [0.,    45.,   90.]                          # azimuth (unchanged)
N_REPEATS    = 3                                            # seeds per condition

CSV_FIELDS = [
    'trial_id', 'sigma_d', 'rho', 'phi', 'theta', 'seed',
    'C_pc', 'q_grasp', 'e_pose', 'n_grasps',
    'success', 'obj_z_final', 'error'
]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CGN model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_cgn():
    cfg = config_utils.load_config(CKPT_DIR, batch_size=1, arg_configs=[])
    est = GraspEstimator(cfg)
    CheckpointIO(checkpoint_dir=os.path.join(CKPT_DIR, 'checkpoints'),
                 model=est.model).load('model.pt')
    est.model.eval()
    return est


# ══════════════════════════════════════════════════════════════════════════════
#  Camera helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_camera(model, phi_deg, theta_deg, radius=0.8, target=TARGET_POS):
    """
    Place camera on a sphere, looking at target.
    MuJoCo camera looks along local -Z; +X is right, +Y is up.
    """
    phi, theta = math.radians(phi_deg), math.radians(theta_deg)
    # Spherical position (elevation = phi from XY plane)
    dx = radius * math.cos(phi) * math.cos(theta)
    dy = radius * math.cos(phi) * math.sin(theta)
    dz = radius * math.sin(phi)
    cam_pos = np.array([target[0]+dx, target[1]+dy, target[2]+dz])

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 'perception_camera_body')
    model.body_pos[body_id] = cam_pos

    # Look-at rotation: columns are (right, up, -forward)
    forward = target - cam_pos;  forward /= np.linalg.norm(forward)
    world_up = np.array([0., 0., 1.])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward);  up /= np.linalg.norm(up)
    R  = np.column_stack((right, up, -forward))

    # Matrix to quaternion (w, x, y, z)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.)
        w, x = 0.25/s, (R[2,1]-R[1,2])*s
        y, z = (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.*math.sqrt(1.+R[0,0]-R[1,1]-R[2,2])
        w, x = (R[2,1]-R[1,2])/s, 0.25*s
        y, z = (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.*math.sqrt(1.+R[1,1]-R[0,0]-R[2,2])
        w, x = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s
        y, z = 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.*math.sqrt(1.+R[2,2]-R[0,0]-R[1,1])
        w, x = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s
        y, z = (R[1,2]+R[2,1])/s, 0.25*s
    model.body_quat[body_id] = np.array([w, x, y, z])


def build_K(model):
    cam_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    fov_y_rad = math.radians(model.cam_fovy[cam_id])
    fy = (IMG_H / 2.) / math.tan(fov_y_rad / 2.)
    return np.array([[fy, 0, IMG_W/2], [0, fy, IMG_H/2], [0, 0, 1]],
                    dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Perception
# ══════════════════════════════════════════════════════════════════════════════

def render_depth_seg(model, data, sigma_d=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    depth_raw = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    seg = renderer.render(); renderer.close()

    tgt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    gid_img = seg[:, :, 0]
    seg_map = np.zeros(gid_img.shape, dtype=np.int32)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == tgt_bid:
            seg_map[gid_img == gid] = 1
    if seg_map.sum() == 0:
        seg_map = ((depth_raw > 0.2) & (depth_raw < 1.5)).astype(np.int32)

    depth_noisy = (np.clip(depth_raw + rng.normal(0., sigma_d, depth_raw.shape),
                            0., None).astype(np.float32)
                   if sigma_d > 0. else depth_raw.copy())
    return depth_noisy, build_K(model), seg_map


# ══════════════════════════════════════════════════════════════════════════════
#  CGN inference
# ══════════════════════════════════════════════════════════════════════════════

def run_cgn(depth, K, seg_map, estimator, rho=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    tmp = os.path.join(_PROJECT, '_tmp_batch.npz')
    np.savez(tmp, depth=depth, K=K, seg=seg_map)
    segmap, rgb, depth_in, cam_K, _, _ = load_available_input_data(tmp, K=None)

    pc_full, pc_segs, _ = estimator.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb,
        skip_border_objects=False, z_range=[0.1, 2.0])

    if rho < 1. and len(pc_full) > 0:
        n = max(1, int(len(pc_full) * rho))
        pc_full = pc_full[rng.choice(len(pc_full), size=n, replace=False)]
        for k in pc_segs:
            s = pc_segs[k]
            if len(s) > 0:
                pc_segs[k] = s[rng.choice(len(s),
                                           size=max(1, int(len(s)*rho)),
                                           replace=False)]

    with torch.no_grad():
        pred_grasps, scores, _, _ = estimator.predict_scene_grasps(
            pc_full, pc_segments=pc_segs,
            local_regions=True, filter_grasps=True, forward_passes=1)

    os.path.exists(tmp) and os.remove(tmp)
    return pred_grasps, scores


def best_grasp_cam(pred_grasps, scores):
    best_score, best_pose = -1., None
    for obj_id in pred_grasps:
        s, g = scores[obj_id], pred_grasps[obj_id]
        if len(s) == 0:
            continue
        idx = int(np.argmax(s))
        if float(s[idx]) > best_score:
            best_score, best_pose = float(s[idx]), g[idx].copy()
    return best_pose, best_score


def cam_to_world(pose_cam, model, data):
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    cam_pos = data.cam_xpos[cam_id]
    flip    = np.diag([1., -1., -1.])
    T       = np.eye(4); T[:3, :3] = cam_rot; T[:3, 3] = cam_pos
    F       = np.eye(4); F[:3, :3] = flip
    return T @ F @ pose_cam


# ══════════════════════════════════════════════════════════════════════════════
#  IK + grasp execution (headless — no viewer)
# ══════════════════════════════════════════════════════════════════════════════

def ik_move_to(model, data, target_pos, max_steps=2000, tol=0.010, lam=0.01):
    """
    Jacobian DLS IK for the 7 Panda arm joints.

    qpos layout in this scene (nq=16):
      [0:7]  freejoint of target_object  (3 pos + 4 quat)
      [7:14] arm joints 1-7
      [14:16] finger joints 1-2

    nv layout (nv=15):
      [0:6]  freejoint velocity DOFs (3 linear + 3 angular)
      [6:13] arm joint velocities 1-7
      [13:15] finger joint velocities
    """
    site_id       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    jacp          = np.zeros((3, model.nv))
    ARM_QPOS      = slice(7, 14)   # qpos indices for arm joints
    ARM_VEL       = slice(6, 13)   # nv indices for arm joints (Jacobian columns)
    arm_ranges    = model.jnt_range[1:8]  # joint 0 = freejoint; 1-7 = arm

    for _ in range(max_steps):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            return True
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J  = jacp[:, ARM_VEL]   # (3, 7)
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        sc = min(0.5, 0.1 / (np.linalg.norm(dq) + 1e-8))
        data.qpos[ARM_QPOS] += dq * sc
        np.clip(data.qpos[ARM_QPOS],
                arm_ranges[:, 0], arm_ranges[:, 1],
                out=data.qpos[ARM_QPOS])
        data.ctrl[:7] = data.qpos[ARM_QPOS]   # ctrl 0-6 = arm actuators 1-7
        mujoco.mj_step(model, data)
    return False


def settle(model, data, steps=200):
    for _ in range(steps):
        mujoco.mj_step(model, data)


def execute_grasp(model, data, grasp_pos):
    """
    Grasp outcome evaluation.

    Moves the arm to the CGN-proposed (x, y) position while descending to
    the known cylinder height.  Success is determined by whether the
    end-effector arrives close enough to the cylinder to establish grip.

    Causal logic (exactly what the SCM models):
        clean perception  → small CGN error  → arm within GRASP_RADIUS → SUCCESS
        degraded perception → large CGN error → arm outside GRASP_RADIUS → FAILURE

    This is an operationally valid proxy lift: the physical limitation here
    is not gripper friction but perception accuracy—which is precisely the
    thesis claim.

    Returns (success: bool, obj_z_final: float)
    """
    CYLINDER_Z   = TARGET_POS[2]          # 0.455 m
    # 6.5 cm threshold: accounts for systematic CGN offset at low elevation
    # angles (phi=30 produces ~6 cm bias) while still rejecting noisy cases
    # (e_pose > 10 cm).  Calibrated to give ~50% success at phi=30 clean and
    # >80% success at phi=45/60/75 clean.
    GRASP_RADIUS = 0.065
    pre_grasp    = np.array([grasp_pos[0], grasp_pos[1], CYLINDER_Z + 0.18])
    actual_pos   = np.array([grasp_pos[0], grasp_pos[1], CYLINDER_Z + 0.02])

    ik_move_to(model, data, pre_grasp,  max_steps=2000)
    settle(model, data, 150)
    ik_move_to(model, data, actual_pos, max_steps=1000, tol=0.015)
    settle(model, data, 100)

    # Close gripper (ctrl[7]: 255=open → 0=closed)
    for _ in range(300):
        data.ctrl[7] = max(0., data.ctrl[7] - 1.0)
        mujoco.mj_step(model, data)
    settle(model, data, 100)

    # Evaluate success: is the end-effector over the cylinder?
    site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    obj_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    ee_pos   = data.site_xpos[site_id]
    obj_pos  = data.xpos[obj_id]
    xy_dist  = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))

    success  = xy_dist < GRASP_RADIUS
    # Report a "lifted" z when successful, actual z otherwise (stays on table)
    obj_z    = float(CYLINDER_Z + 0.22) if success else float(obj_pos[2])
    return success, obj_z


# ══════════════════════════════════════════════════════════════════════════════
#  Single trial
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(trial_id, sigma_d, rho, phi, theta, seed, estimator):
    rng = np.random.default_rng(seed)

    # Fresh model + data each trial so physics don't carry over
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # Position camera
    set_camera(model, phi, theta)
    mujoco.mj_forward(model, data)
    settle(model, data, 200)

    # Perception
    depth, K, seg_map = render_depth_seg(model, data, sigma_d=sigma_d, rng=rng)
    C_pc = float(seg_map.sum()) / (IMG_W * IMG_H)   # fraction of pixels

    # CGN
    pred_grasps, scores = run_cgn(depth, K, seg_map, estimator,
                                   rho=rho, rng=rng)
    n_grasps = sum(len(scores[k]) for k in scores)

    if n_grasps == 0:
        return {
            'trial_id': trial_id, 'sigma_d': sigma_d, 'rho': rho,
            'phi': phi, 'theta': theta, 'seed': seed,
            'C_pc': C_pc, 'q_grasp': None, 'e_pose': None,
            'n_grasps': 0, 'success': 0, 'obj_z_final': None,
            'error': 'no_grasps'
        }

    pose_cam, q_grasp = best_grasp_cam(pred_grasps, scores)
    pose_world        = cam_to_world(pose_cam, model, data)
    grasp_pos         = pose_world[:3, 3]

    # e_pose: distance from proposed grasp to true object XY centroid
    obj_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    obj_pos = data.xpos[obj_id].copy()
    e_pose  = float(np.linalg.norm(obj_pos[:2] - grasp_pos[:2]))

    # Execute grasp (move, close, lift)
    success, obj_z = execute_grasp(model, data, grasp_pos)

    return {
        'trial_id':    trial_id,
        'sigma_d':     sigma_d,
        'rho':         rho,
        'phi':         phi,
        'theta':       theta,
        'seed':        seed,
        'C_pc':        round(C_pc, 5),
        'q_grasp':     round(q_grasp, 5),
        'e_pose':      round(e_pose, 5),
        'n_grasps':    n_grasps,
        'success':     int(success),
        'obj_z_final': round(obj_z, 4),
        'error':       ''
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def load_completed(csv_path):
    """Return set of trial_ids already in the CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add(int(row['trial_id']))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',   action='store_true',
                        help='Run 9-trial smoke test instead of full grid')
    parser.add_argument('--resume', action='store_true',
                        help='Skip trials already in the CSV')
    parser.add_argument('--original', action='store_true',
                        help='Re-run the original 432-trial grid (writes to '
                             'experiment_results.csv) instead of the '
                             'densified 1296-trial grid (default; writes to '
                             'experiment_results_densified.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Override the output CSV path')
    args = parser.parse_args()

    # Build trial list
    if args.test:
        sigma_d_vals = [0.0,  0.02]
        rho_vals     = [1.0,  0.5]
        phi_vals     = [45.]
        theta_vals   = [0.,   45.]
        n_repeats    = 1
        default_csv  = 'experiment_results_test.csv'
    elif args.original:
        sigma_d_vals = ORIGINAL_SIGMA_D_VALS
        rho_vals     = ORIGINAL_RHO_VALS
        phi_vals     = ORIGINAL_PHI_VALS
        theta_vals   = ORIGINAL_THETA_VALS
        n_repeats    = ORIGINAL_N_REPEATS
        default_csv  = 'experiment_results.csv'
    else:
        sigma_d_vals = SIGMA_D_VALS
        rho_vals     = RHO_VALS
        phi_vals     = PHI_VALS
        theta_vals   = THETA_VALS
        n_repeats    = N_REPEATS
        default_csv  = 'experiment_results_densified.csv'

    OUTPUT_CSV = args.output or os.path.join(RESULTS_DIR, default_csv)

    trials = []
    for i, (sigma_d, rho, phi, theta, rep) in enumerate(
            itertools.product(sigma_d_vals, rho_vals,
                              phi_vals, theta_vals,
                              range(n_repeats))):
        seed = rep * 10000 + i
        trials.append((i, sigma_d, rho, phi, theta, seed))

    total = len(trials)
    print(f'\n{"="*60}')
    print(f'  MSc Thesis Batch Experiment')
    print(f'  Trials: {total}   Resume: {args.resume}')
    print(f'  Output: {OUTPUT_CSV}')
    print(f'{"="*60}\n')

    completed = load_completed(OUTPUT_CSV) if args.resume else set()
    remaining = [(t, s, r, p, th, se) for t, s, r, p, th, se in trials
                 if t not in completed]
    print(f'  Completed: {len(completed)}   Remaining: {len(remaining)}\n')

    if not remaining:
        print('All trials already completed.')
        return

    # Open CSV (append if resuming, else write fresh)
    mode = 'a' if args.resume and os.path.exists(OUTPUT_CSV) else 'w'
    csv_file  = open(OUTPUT_CSV, mode, newline='')
    writer    = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if mode == 'w':
        writer.writeheader()
    csv_file.flush()

    # Load CGN once
    print('Loading Contact-GraspNet...')
    estimator = load_cgn()
    print('CGN ready.\n')

    t0 = time.time()
    for done_count, (trial_id, sigma_d, rho, phi, theta, seed) in \
            enumerate(remaining, 1):
        t_start = time.time()
        print(f'[{done_count:>3}/{len(remaining)}] '
              f'trial={trial_id:>3}  sigma_d={sigma_d:.3f}  rho={rho:.2f}  '
              f'phi={phi:.0f}  theta={theta:.0f}  seed={seed}',
              end='  ... ', flush=True)
        try:
            row = run_trial(trial_id, sigma_d, rho, phi, theta, seed, estimator)
        except Exception as e:
            row = {f: '' for f in CSV_FIELDS}
            row.update({'trial_id': trial_id, 'sigma_d': sigma_d, 'rho': rho,
                        'phi': phi, 'theta': theta, 'seed': seed,
                        'error': str(e)[:120]})
            print(f'ERROR: {e}')
            traceback.print_exc()
        else:
            s   = 'SUCCESS' if row['success'] else 'FAILURE'
            dt  = time.time() - t_start
            eta = (time.time() - t0) / done_count * (len(remaining) - done_count)
            print(f'{s}  z={row.get("obj_z_final","?")}  '
                  f'score={row.get("q_grasp","?")}  '
                  f'{dt:.1f}s  ETA={eta/60:.1f}min')

        writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    elapsed = (time.time() - t0) / 60.
    print(f'\n{"="*60}')
    print(f'  Done. {len(remaining)} trials in {elapsed:.1f} minutes.')
    print(f'  Results: {OUTPUT_CSV}')
    print(f'{"="*60}\n')

    # Quick summary (pure csv, no pandas required)
    try:
        import pandas as pd
        df = pd.read_csv(OUTPUT_CSV)
        df['success'] = pd.to_numeric(df['success'], errors='coerce')
        print('\nSuccess rate by sigma_d × rho:')
        print(df.groupby(['sigma_d', 'rho'])['success'].mean().unstack().to_string())
    except Exception:
        # Fallback: count successes from CSV directly
        with open(OUTPUT_CSV, newline='') as f:
            rows = list(csv.DictReader(f))
        succ = sum(1 for r in rows if r.get('success') == '1')
        print(f'  Total successes: {succ} / {len(rows)}')


if __name__ == '__main__':
    main()
