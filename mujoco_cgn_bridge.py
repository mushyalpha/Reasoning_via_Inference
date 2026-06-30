"""
mujoco_cgn_bridge.py
====================
Stage 2 of the causal diagnosis pipeline.

Bridges MuJoCo perception (Stage 1) with Contact-GraspNet grasp prediction.

Steps
-----
  1. Load MuJoCo scene (Panda + table + cylinder)
  2. Position camera at (phi, theta) on a fixed-radius sphere
  3. Render depth image + apply Gaussian noise (sigma_d) [causal var]
  4. Render segmentation map to identify the target object
  5. Save as CGN-compatible .npz file
  6. Run Contact-GraspNet inference
  7. Extract best grasp pose, transform camera frame -> world frame
  8. Return world-frame pose ready for inverse kinematics / MuJoCo execution

Causal variables
----------------
  sigma_d : Gaussian depth noise std dev (injected onto depth buffer)
  phi     : Camera elevation angle
  theta   : Camera azimuth angle
  rho     : Point cloud keep fraction (1.0 = full, <1.0 = sparse)

Usage
-----
  python mujoco_cgn_bridge.py          # runs a single demo trial

  Or import for batch experiments:
    from mujoco_cgn_bridge import load_cgn_model, run_trial
"""

import os
import sys
import math
import numpy as np
import mujoco
import torch

# -----------------------------------------------------------------------
# CGN package path
# contact_graspnet_pytorch is installed with pip install -e, but the bare
# `from data import ...` inside CGN's own modules needs the source dir
# to be on sys.path as well.
# -----------------------------------------------------------------------
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CGN_SRC = os.path.join(_PROJECT_DIR, 'contact_graspnet_pytorch',
                         'contact_graspnet_pytorch')
if _CGN_SRC not in sys.path:
    sys.path.insert(0, _CGN_SRC)

from contact_grasp_estimator import GraspEstimator
import config_utils
from checkpoints import CheckpointIO
from data import load_available_input_data

# Stage-1 helper from existing pipeline
from grasp_simulation import set_camera_spherical


# =======================================================================
#  CGN model loading  (call ONCE, reuse across all 432 trials)
# =======================================================================

def load_cgn_model():
    """
    Load the pre-trained CGN checkpoint from the repo checkpoints folder.

    Returns
    -------
    global_config   : dict            -- CGN config.yaml as a dict
    grasp_estimator : GraspEstimator  -- model ready for inference on CPU
    """
    cgn_root  = os.path.join(_PROJECT_DIR, 'contact_graspnet_pytorch')
    ckpt_dir  = os.path.join(cgn_root, 'checkpoints', 'contact_graspnet')

    global_config = config_utils.load_config(ckpt_dir, batch_size=1,
                                              arg_configs=[])

    grasp_estimator = GraspEstimator(global_config)

    ckpt_io = CheckpointIO(
        checkpoint_dir=os.path.join(ckpt_dir, 'checkpoints'),
        model=grasp_estimator.model
    )
    ckpt_io.load('model.pt')
    grasp_estimator.model.eval()
    print('CGN model loaded (CPU, eval mode).')
    return global_config, grasp_estimator


# =======================================================================
#  MuJoCo rendering helpers
# =======================================================================

def _build_K(model, cam_name, img_width, img_height):
    """Build 3x3 camera intrinsic matrix from MuJoCo camera parameters."""
    cam_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fov_y_rad = math.radians(model.cam_fovy[cam_id])
    fy = (img_height / 2.0) / math.tan(fov_y_rad / 2.0)
    fx = fy
    cx = img_width  / 2.0
    cy = img_height / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def render_scene_for_cgn(model, data,
                          cam_name='perception_camera',
                          target_body_name='target_object',
                          img_width=640, img_height=480,
                          sigma_d=0.0,
                          rng=None):
    """
    Render MuJoCo scene and return CGN-compatible (depth, K, seg_map).

    Parameters
    ----------
    model, data        : MuJoCo model and data
    cam_name           : camera name in scene XML
    target_body_name   : body to grasp ('target_object')
    img_width/height   : render resolution
    sigma_d            : Gaussian depth noise std dev in metres
    rng                : numpy Generator

    Returns
    -------
    depth_noisy : (H, W) float32 -- depth + noise, metres
    K           : (3, 3) float32 -- camera intrinsics
    seg_map     : (H, W) int32   -- 1=target object, 0=everything else
    """
    if rng is None:
        rng = np.random.default_rng()

    renderer = mujoco.Renderer(model, height=img_height, width=img_width)

    # --- depth ---
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth_raw = renderer.render().copy()
    renderer.disable_depth_rendering()

    # --- segmentation ---
    # Returns (H, W, 2): channel 0 = geom_id (-1 = background)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_name)
    seg_raw = renderer.render()
    renderer.close()

    # Build binary segmentation map: target object geoms = 1
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        target_body_name)
    geom_ids_image = seg_raw[:, :, 0]
    seg_map = np.zeros(geom_ids_image.shape, dtype=np.int32)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == target_body_id:
            seg_map[geom_ids_image == gid] = 1

    # Fallback: if object not visible use depth-range heuristic
    if seg_map.sum() == 0:
        print('  Warning: target not visible in segmentation. '
              'Using depth-range fallback (0.3m-1.0m).')
        seg_map = ((depth_raw > 0.3) & (depth_raw < 1.0)).astype(np.int32)

    # --- inject noise (causal variable sigma_d) ---
    if sigma_d > 0.0:
        noise = rng.normal(0.0, sigma_d, depth_raw.shape).astype(np.float32)
        depth_noisy = np.clip(depth_raw + noise, 0.0, None)
    else:
        depth_noisy = depth_raw.copy()

    K = _build_K(model, cam_name, img_width, img_height)
    return depth_noisy, K, seg_map


# =======================================================================
#  CGN inference on MuJoCo data
# =======================================================================

def run_cgn_inference(depth, K, seg_map, grasp_estimator,
                       z_range=(0.2, 1.8), rho=1.0, rng=None):
    """
    Run Contact-GraspNet on depth+K+seg from MuJoCo.

    rho < 1.0 randomly drops points from the full point cloud before
    inference, operationalising the sparsity causal variable.

    Returns
    -------
    pred_grasps_cam : dict {obj_id: (N,4,4)}
    scores          : dict {obj_id: (N,)}
    contact_pts     : dict {obj_id: (N,3)}
    pc_full         : (M,3) array
    """
    if rng is None:
        rng = np.random.default_rng()

    # Save to temp file in CGN-expected format
    tmp_path = os.path.join(_PROJECT_DIR, '_tmp_cgn_input.npz')
    np.savez(tmp_path, depth=depth, K=K, seg=seg_map)

    # Load via CGN's data loader
    segmap, rgb, depth_in, cam_K, pc_full, pc_colors = \
        load_available_input_data(tmp_path, K=None)

    # CGN's point cloud extraction (handles depth -> 3D, z-range filter)
    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
        depth_in, cam_K,
        segmap=segmap, rgb=rgb,
        skip_border_objects=False,
        z_range=list(z_range)
    )

    # Apply rho: downsample (causal variable)
    if rho < 1.0 and len(pc_full) > 0:
        n_keep = max(1, int(len(pc_full) * rho))
        idx = rng.choice(len(pc_full), size=n_keep, replace=False)
        pc_full = pc_full[idx]
        for k in pc_segments:
            s = pc_segments[k]
            if len(s) > 0:
                n_k = max(1, int(len(s) * rho))
                pc_segments[k] = s[rng.choice(len(s), size=n_k,
                                               replace=False)]

    # Run neural network
    with torch.no_grad():
        pred_grasps_cam, scores, contact_pts, _ = \
            grasp_estimator.predict_scene_grasps(
                pc_full,
                pc_segments=pc_segments,
                local_regions=True,
                filter_grasps=True,
                forward_passes=1
            )

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return pred_grasps_cam, scores, contact_pts, pc_full


def best_grasp(pred_grasps_cam, scores):
    """Return the single highest-scoring grasp pose (camera frame)."""
    best_score    = -1.0
    best_pose_cam = None
    for obj_id in pred_grasps_cam:
        s = scores[obj_id]
        g = pred_grasps_cam[obj_id]
        if len(s) == 0:
            continue
        idx = int(np.argmax(s))
        if float(s[idx]) > best_score:
            best_score    = float(s[idx])
            best_pose_cam = g[idx]
    return best_pose_cam, best_score


# =======================================================================
#  Coordinate frame transform: camera -> world
# =======================================================================

def camera_to_world(pose_cam, model, data,
                     cam_name='perception_camera'):
    """
    Convert a 4x4 grasp pose from CGN camera frame to MuJoCo world frame.

    CGN uses OpenCV convention: +X right, +Y down,  +Z forward
    MuJoCo camera frame:        +X right, +Y up,    +Z backward

    The flip [1,-1,-1] converts between them. Then cam_rot (from
    data.cam_xmat) rotates from MuJoCo camera frame to world frame.

    Parameters
    ----------
    pose_cam : (4,4) grasp pose in CGN camera frame
    model, data : MuJoCo model + data (after mj_forward)
    cam_name : camera name in scene XML

    Returns
    -------
    pose_world : (4,4) grasp pose in MuJoCo world frame
    """
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    cam_pos = data.cam_xpos[cam_id]

    # OpenCV -> MuJoCo camera: flip Y and Z
    flip = np.diag([1.0, -1.0, -1.0])

    T_cam_world      = np.eye(4)
    T_cam_world[:3, :3] = cam_rot
    T_cam_world[:3,  3] = cam_pos

    T_flip           = np.eye(4)
    T_flip[:3, :3]   = flip

    return T_cam_world @ T_flip @ pose_cam


# =======================================================================
#  Single trial runner  (main entry point for experiments)
# =======================================================================

def run_trial(model, data, grasp_estimator,
              phi=45.0, theta=0.0,
              sigma_d=0.0, rho=1.0,
              radius=0.8, target_pos=None,
              cam_name='perception_camera',
              target_body_name='target_object',
              seed=None):
    """
    Run one complete perception->grasp trial.

    Parameters
    ----------
    model, data         : MuJoCo model and data
    grasp_estimator     : loaded CGN model (from load_cgn_model)
    phi                 : camera elevation in degrees
    theta               : camera azimuth in degrees
    sigma_d             : depth noise std dev in metres (causal var)
    rho                 : point cloud keep fraction (causal var)
    radius              : camera sphere radius in metres
    target_pos          : (3,) array, world pos of grasp target
    cam_name            : camera name in scene XML
    target_body_name    : body to grasp
    seed                : int, for reproducibility

    Returns
    -------
    dict with keys:
      phi, theta, sigma_d, rho : trial parameters
      pose_world               : (4,4) grasp pose in world frame (or None)
      pose_cam                 : (4,4) grasp pose in camera frame (or None)
      score                    : float CGN confidence (or None)
      n_grasps                 : int total grasp candidates generated
      success                  : None (filled by MuJoCo execution step)
    """
    if target_pos is None:
        target_pos = np.array([0.5, 0.0, 0.45])

    rng = np.random.default_rng(seed=seed)

    # 1. Position camera
    set_camera_spherical(model, phi, theta, radius, np.array(target_pos))
    mujoco.mj_forward(model, data)

    # 2. Render depth + segmentation
    depth, K, seg_map = render_scene_for_cgn(
        model, data,
        cam_name=cam_name,
        target_body_name=target_body_name,
        sigma_d=sigma_d,
        rng=rng
    )
    print(f'  [phi={phi:.0f} theta={theta:.0f} sigma_d={sigma_d} rho={rho}] '
          f'target pixels={seg_map.sum()}')

    # 3. Run CGN
    pred_grasps_cam, scores, contact_pts, pc_full = run_cgn_inference(
        depth, K, seg_map, grasp_estimator, rho=rho, rng=rng
    )
    n_grasps = sum(len(scores[k]) for k in scores)
    print(f'  CGN: {n_grasps} grasp candidates')

    # 4. Best grasp
    pose_cam, score = best_grasp(pred_grasps_cam, scores)

    if pose_cam is None:
        print('  No valid grasps found.')
        return dict(phi=phi, theta=theta, sigma_d=sigma_d, rho=rho,
                    pose_world=None, pose_cam=None,
                    score=None, n_grasps=0, success=None)

    # 5. Transform to world frame
    pose_world = camera_to_world(pose_cam, model, data, cam_name)

    print(f'  Score: {score:.4f} | '
          f'pos_world: ({pose_world[0,3]:.3f}, '
          f'{pose_world[1,3]:.3f}, {pose_world[2,3]:.3f})')

    return dict(phi=phi, theta=theta, sigma_d=sigma_d, rho=rho,
                pose_world=pose_world, pose_cam=pose_cam,
                score=score, n_grasps=n_grasps, success=None)


# =======================================================================
#  Demo main
# =======================================================================

def main():
    print('=' * 60)
    print('MuJoCo -> Contact-GraspNet Bridge  (demo)')
    print('=' * 60)

    xml_path = os.path.join(_PROJECT_DIR, 'simple_grasp_scene.xml')
    print(f'Scene: {xml_path}')
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print('Loading CGN...')
    _, grasp_estimator = load_cgn_model()

    # Baseline trial: clean conditions
    print('\n--- Baseline trial (clean, phi=45, theta=0) ---')
    result = run_trial(model, data, grasp_estimator,
                       phi=45.0, theta=0.0,
                       sigma_d=0.0, rho=1.0, seed=42)

    if result['pose_world'] is not None:
        print('\n=== RESULT ===')
        print(f"Score     : {result['score']:.4f}")
        print(f"N grasps  : {result['n_grasps']}")
        print(f"Position  : {result['pose_world'][:3, 3]}")
        print(f"\n4x4 world pose:\n{result['pose_world']}")
        print('\nNext step: pass pose_world to the Panda IK controller in MuJoCo.')
    else:
        print('No grasps found. Adjust camera angle or object position.')


if __name__ == '__main__':
    main()
