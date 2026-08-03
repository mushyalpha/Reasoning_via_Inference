"""
demo_grasp.py
=============
Visual demonstration: MuJoCo + Contact-GraspNet full pipeline.

Opens the MuJoCo viewer so you can WATCH the Panda arm:
  1. Start at home position
  2. Camera captures depth image
  3. Contact-GraspNet proposes a grasp pose
  4. Arm moves to pre-grasp (above object)
  5. Arm descends to grasp position
  6. Gripper closes
  7. Arm lifts  →  success or failure recorded

Run:
    python demo_grasp.py
    python demo_grasp.py --sigma_d 0.02      # with depth noise
    python demo_grasp.py --sigma_d 0.04 --rho 0.5  # noisy + sparse
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer

try:
    import imageio
    _IMAGEIO_OK = True
except ImportError:
    _IMAGEIO_OK = False

# ── CGN imports ────────────────────────────────────────────────────────────────
_PROJECT  = os.path.dirname(os.path.abspath(__file__))
_CGN_REPO = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
_CGN_SRC  = os.path.join(_CGN_REPO, 'contact_graspnet_pytorch')
# _CGN_REPO must come first so the inner folder is found as a proper package
# (needed for `from contact_graspnet_pytorch import config_utils` inside CGN).
# _CGN_SRC provides flat imports like `from contact_grasp_estimator import ...`.
for _p in [_CGN_REPO, _CGN_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contact_grasp_estimator import GraspEstimator
import config_utils
from checkpoints import CheckpointIO
from data import load_available_input_data
import torch

# ── Constants ──────────────────────────────────────────────────────────────────
SCENE_XML   = os.path.join(_PROJECT, 'grasp_scene_v2.xml')
CGN_ROOT    = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
CKPT_DIR    = os.path.join(CGN_ROOT, 'checkpoints', 'contact_graspnet')
EE_SITE     = 'ee_site'          # site name in XML
TARGET_BODY = 'target_object'
CAM_NAME    = 'perception_camera'
LIFT_HEIGHT = 0.55               # object must reach this Z to count as success
TARGET_POS  = np.array([0.5, 0., 0.455])   # cylinder centroid (world frame)
IMG_W, IMG_H = 640, 480
HOME_QPOS   = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])


# ══════════════════════════════════════════════════════════════════════════════
#  CGN model  (loaded once)
# ══════════════════════════════════════════════════════════════════════════════

def load_cgn():
    global_config = config_utils.load_config(CKPT_DIR, batch_size=1,
                                              arg_configs=[])
    estimator = GraspEstimator(global_config)
    ckpt_io   = CheckpointIO(
        checkpoint_dir=os.path.join(CKPT_DIR, 'checkpoints'),
        model=estimator.model)
    ckpt_io.load('model.pt')
    estimator.model.eval()
    print('[CGN] Model loaded (CPU mode).')
    return estimator


# ══════════════════════════════════════════════════════════════════════════════
#  Video recorder (offscreen, fixed viewpoint)
# ══════════════════════════════════════════════════════════════════════════════

class VideoRecorder:
    """Captures MuJoCo frames offscreen and writes them to an MP4."""

    def __init__(self, model, output_path, fps=30, width=1280, height=720):
        self.output_path = output_path if output_path.endswith('.mp4') else output_path + '.mp4'
        self.fps   = fps
        self.frames = []
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self._cam)
        # Match the initial viewer viewpoint
        self._cam.distance  = 2.0
        self._cam.azimuth   = 135.0
        self._cam.elevation = -20.0

    def capture(self, data):
        self._renderer.update_scene(data, camera=self._cam)
        self.frames.append(self._renderer.render().copy())

    def save(self):
        if not self.frames:
            print('[Record] No frames captured.')
            return
        print(f'[Record] Writing {len(self.frames)} frames to {self.output_path} ...')
        writer = imageio.get_writer(self.output_path, fps=self.fps,
                                    codec='libx264', quality=8,
                                    macro_block_size=None)
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        self._renderer.close()
        size_mb = os.path.getsize(self.output_path) / 1e6
        print(f'[Record] Done. {self.output_path}  ({size_mb:.1f} MB)')


# ══════════════════════════════════════════════════════════════════════════════
#  Camera helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_camera(model, phi_deg, theta_deg, radius=0.8,
               target=np.array([0.5, 0., 0.455])):
    """
    Place camera on a sphere, looking at target.
    MuJoCo camera: looks along local -Z, +X right, +Y up.
    """
    phi, theta = math.radians(phi_deg), math.radians(theta_deg)
    dx = radius * math.cos(phi) * math.cos(theta)
    dy = radius * math.cos(phi) * math.sin(theta)
    dz = radius * math.sin(phi)
    cam_pos = np.array([target[0]+dx, target[1]+dy, target[2]+dz])

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                 'perception_camera_body')
    model.body_pos[body_id] = cam_pos

    forward = target - cam_pos;  forward /= np.linalg.norm(forward)
    world_up = np.array([0., 0., 1.])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward);  up /= np.linalg.norm(up)
    R  = np.column_stack((right, up, -forward))

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
    fy = (IMG_H / 2.0) / math.tan(fov_y_rad / 2.0)
    return np.array([[fy, 0, IMG_W/2], [0, fy, IMG_H/2], [0, 0, 1]],
                    dtype=np.float32)


def render_depth_seg(model, data, sigma_d=0.0, rng=None):
    """Render depth + segmentation; inject Gaussian noise (sigma_d)."""
    if rng is None:
        rng = np.random.default_rng()
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    depth_raw = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    seg_raw = renderer.render()
    renderer.close()

    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    gid_img    = seg_raw[:, :, 0]
    seg_map    = np.zeros(gid_img.shape, dtype=np.int32)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == target_bid:
            seg_map[gid_img == gid] = 1
    if seg_map.sum() == 0:
        seg_map = ((depth_raw > 0.2) & (depth_raw < 1.5)).astype(np.int32)

    depth_noisy = depth_raw.copy()
    if sigma_d > 0.0:
        depth_noisy = np.clip(depth_raw +
                              rng.normal(0., sigma_d, depth_raw.shape), 0., None
                              ).astype(np.float32)
    return depth_noisy, build_K(model), seg_map


# ══════════════════════════════════════════════════════════════════════════════
#  CGN inference
# ══════════════════════════════════════════════════════════════════════════════

def run_cgn(depth, K, seg_map, estimator, rho=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    tmp = os.path.join(_PROJECT, '_tmp_demo.npz')
    np.savez(tmp, depth=depth, K=K, seg=seg_map)
    segmap, rgb, depth_in, cam_K, _, _ = load_available_input_data(tmp, K=None)

    pc_full, pc_segs, pc_col = estimator.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb,
        skip_border_objects=False, z_range=[0.1, 2.0])

    if rho < 1.0 and len(pc_full) > 0:
        n = max(1, int(len(pc_full) * rho))
        idx = rng.choice(len(pc_full), size=n, replace=False)
        pc_full = pc_full[idx]
        for k in pc_segs:
            s = pc_segs[k]
            if len(s) > 0:
                pc_segs[k] = s[rng.choice(len(s),
                                           size=max(1, int(len(s)*rho)),
                                           replace=False)]

    with torch.no_grad():
        pred_grasps, scores, contact_pts, _ = estimator.predict_scene_grasps(
            pc_full, pc_segments=pc_segs,
            local_regions=True, filter_grasps=True, forward_passes=1)

    if os.path.exists(tmp):
        os.remove(tmp)
    return pred_grasps, scores


def best_grasp_cam(pred_grasps, scores):
    best_score, best_pose = -1., None
    for obj_id in pred_grasps:
        s = scores[obj_id]
        g = pred_grasps[obj_id]
        if len(s) == 0:
            continue
        idx = int(np.argmax(s))
        if float(s[idx]) > best_score:
            best_score = float(s[idx])
            best_pose  = g[idx].copy()
    return best_pose, best_score


def top_n_grasps_cam(pred_grasps, scores, n=3):
    """Return the top-n grasps across all objects, sorted by score descending."""
    all_poses = []
    for obj_id in pred_grasps:
        s = scores[obj_id]
        g = pred_grasps[obj_id]
        for i in range(len(s)):
            all_poses.append((float(s[i]), g[i].copy()))
    all_poses.sort(key=lambda x: x[0], reverse=True)
    return [(pose, score) for score, pose in all_poses[:n]]


def cam_to_world(pose_cam, model, data):
    cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    cam_pos = data.cam_xpos[cam_id]
    flip    = np.diag([1., -1., -1.])
    T       = np.eye(4)
    T[:3, :3] = cam_rot
    T[:3,  3] = cam_pos
    F         = np.eye(4)
    F[:3, :3] = flip
    return T @ F @ pose_cam


# ══════════════════════════════════════════════════════════════════════════════
#  IK controller  (Jacobian damped least-squares, position only)
# ══════════════════════════════════════════════════════════════════════════════

def ik_move_to(model, data, target_pos, viewer=None,
               max_steps=2000, tol=0.008, lam=0.01, rec=None):
    """
    Jacobian DLS IK for the 7 Panda arm joints.

    qpos layout (nq=16):  [0:7] freejoint, [7:14] arm, [14:16] fingers
    nv layout  (nv=15):   [0:6] freejoint, [6:13] arm, [13:15] fingers
    """
    site_id    = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    jacp       = np.zeros((3, model.nv))
    ARM_QPOS   = slice(7, 14)
    ARM_VEL    = slice(6, 13)
    arm_ranges = model.jnt_range[1:8]

    for step in range(max_steps):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            return True

        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J  = jacp[:, ARM_VEL]
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)

        scale = min(0.5, 0.1 / (np.linalg.norm(dq) + 1e-8))
        data.qpos[ARM_QPOS] += dq * scale
        np.clip(data.qpos[ARM_QPOS],
                arm_ranges[:, 0], arm_ranges[:, 1],
                out=data.qpos[ARM_QPOS])
        data.ctrl[:7] = data.qpos[ARM_QPOS]
        mujoco.mj_step(model, data)

        if viewer is not None and step % 5 == 0:
            viewer.sync()
            time.sleep(0.018)   # ~18ms per rendered frame → visibly slow
            if rec is not None:
                rec.capture(data)

    return np.linalg.norm(target_pos - data.site_xpos[site_id]) < tol * 3


def settle(model, data, viewer=None, steps=300, rec=None):
    """Run simulation to let physics settle."""
    for i in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None and i % 5 == 0:
            viewer.sync()
            time.sleep(0.018)
            if rec is not None:
                rec.capture(data)


# ══════════════════════════════════════════════════════════════════════════════
#  Main demo
# ══════════════════════════════════════════════════════════════════════════════

def _reset_arm(model, data, viewer=None, rec=None):
    """Return arm to home keyframe, keeping object in place."""
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    obj_pos  = data.xpos[obj_id].copy()
    obj_quat = data.xquat[obj_id].copy()
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # restore object position so it doesn't reset too
    data.qpos[0:3] = obj_pos
    data.qpos[3:7] = obj_quat
    mujoco.mj_forward(model, data)
    settle(model, data, viewer, steps=100, rec=rec)


def run_demo(phi=45., theta=0., sigma_d=0., rho=1.0, seed=42, n_poses=3,
             record=None, record_fps=30):
    rng = np.random.default_rng(seed)

    print(f'\n{"="*60}')
    print(f'  GRASP DEMO  phi={phi}  theta={theta}  '
          f'sigma_d={sigma_d}  rho={rho}  n_poses={n_poses}')
    print(f'{"="*60}')

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # ── Set up video recorder ────────────────────────────────────────────────
    rec = None
    if record:
        if not _IMAGEIO_OK:
            print('[Record] ERROR: imageio not installed.')
            print('         Run:  pip install imageio[ffmpeg]')
        else:
            rec = VideoRecorder(model, record, fps=record_fps)
            print(f'[Record] Will save video to: {rec.output_path}  ({record_fps} fps)')

    print('[CGN] Loading model...')
    estimator = load_cgn()

    print('[Viewer] Opening MuJoCo viewer... (close to exit)')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth  = 135.
        viewer.cam.elevation = -20.

        # ── Step 1: position camera ──────────────────────────────────────────
        set_camera(model, phi, theta)
        mujoco.mj_forward(model, data)
        settle(model, data, viewer, steps=200, rec=rec)

        # ── Step 2: capture depth + run CGN (once) ───────────────────────────
        print('[Perception] Rendering depth image...')
        depth, K, seg_map = render_depth_seg(model, data,
                                              sigma_d=sigma_d, rng=rng)
        print(f'  Target pixels visible: {seg_map.sum()}')

        print('[CGN] Running inference...')
        pred_grasps, scores = run_cgn(depth, K, seg_map, estimator,
                                       rho=rho, rng=rng)
        total = sum(len(scores[k]) for k in scores)
        print(f'  Generated {total} grasp candidates')

        grasps = top_n_grasps_cam(pred_grasps, scores, n=n_poses)
        if not grasps:
            print('[RESULT] No grasps found. Exiting.')
            return []

        print(f'  Will attempt top {len(grasps)} poses.\n')

        CYLINDER_Z = TARGET_POS[2]
        results = []

        # ── Loop: try each pose ──────────────────────────────────────────────
        for attempt, (pose_cam, cgn_score) in enumerate(grasps, start=1):
            if not viewer.is_running():
                break

            print(f'\n{"─"*60}')
            print(f'  ATTEMPT {attempt}/{len(grasps)}   score={cgn_score:.4f}')
            print(f'{"─"*60}')

            # Reset arm to home, keep object where it is
            _reset_arm(model, data, viewer, rec=rec)

            pose_world = cam_to_world(pose_cam, model, data)
            grasp_pos  = pose_world[:3, 3]
            pre_grasp  = np.array([grasp_pos[0], grasp_pos[1], CYLINDER_Z + 0.18])
            actual_pos = np.array([grasp_pos[0], grasp_pos[1], CYLINDER_Z + 0.02])
            lift_target = np.array([grasp_pos[0], grasp_pos[1], CYLINDER_Z + 0.25])

            print(f'  Grasp XY: ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f})')

            # Pre-grasp
            print('[Robot] Moving to pre-grasp...')
            ik_move_to(model, data, pre_grasp, viewer=viewer,
                       max_steps=2000, rec=rec)
            settle(model, data, viewer, steps=150, rec=rec)

            # Descend
            print('[Robot] Descending...')
            ik_move_to(model, data, actual_pos, viewer=viewer,
                       max_steps=800, tol=0.015, rec=rec)
            settle(model, data, viewer, steps=120, rec=rec)

            # Close gripper
            print('[Robot] Closing gripper...')
            for t in range(350):
                data.ctrl[7] = max(0., data.ctrl[7] - 0.73)
                mujoco.mj_step(model, data)
                if t % 8 == 0:
                    viewer.sync()
                    time.sleep(0.018)
                    if rec is not None:
                        rec.capture(data)
            settle(model, data, viewer, steps=120, rec=rec)

            # Lift
            print('[Robot] Lifting...')
            ik_move_to(model, data, lift_target, viewer=viewer,
                       max_steps=1500, rec=rec)
            settle(model, data, viewer, steps=200, rec=rec)

            # Result
            obj_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
            obj_z   = float(data.xpos[obj_id][2])
            e_pose  = float(np.linalg.norm(data.xpos[obj_id][:2] - grasp_pos[:2]))
            success = obj_z > LIFT_HEIGHT

            print(f'\n  RESULT attempt {attempt}: {"SUCCESS" if success else "FAILURE"}')
            print(f'  Object Z = {obj_z:.3f}m   e_pose = {e_pose:.4f}m')

            results.append({'attempt': attempt, 'score': cgn_score,
                            'success': success, 'obj_z': obj_z, 'e_pose': e_pose})

            # Pause between attempts so the professor can see the result
            if attempt < len(grasps) and viewer.is_running():
                print(f'\n  [Pausing 3 s before next attempt...]')
                for _ in range(150):
                    if not viewer.is_running():
                        break
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.02)
                    if rec is not None:
                        rec.capture(data)

        # Final summary
        print(f'\n{"="*60}')
        print(f'  SUMMARY  ({len(results)} attempts)')
        for r in results:
            tag = 'SUCCESS' if r['success'] else 'FAILURE'
            print(f'  Attempt {r["attempt"]}: {tag}  '
                  f'score={r["score"]:.3f}  e_pose={r["e_pose"]:.3f}m')
        print(f'{"="*60}\n')

        # Save video before the idle loop so the file is written even if
        # the user closes the viewer window early.
        if rec is not None:
            rec.save()

        print('[Viewer] Done. Close the window to exit.')
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo grasp with MuJoCo viewer')
    parser.add_argument('--phi',        type=float, default=45.,
                        help='Camera elevation angle (deg)')
    parser.add_argument('--theta',      type=float, default=0.,
                        help='Camera azimuth angle (deg)')
    parser.add_argument('--sigma_d',    type=float, default=0.,
                        help='Depth noise std dev (m) — causal var')
    parser.add_argument('--rho',        type=float, default=1.0,
                        help='Point cloud keep fraction — causal var')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--n_poses',    type=int,   default=3,
                        help='Number of CGN poses to attempt (default: 3)')
    parser.add_argument('--record',     type=str,   default=None,
                        help='Save simulation video to this path, e.g. demo.mp4')
    parser.add_argument('--record_fps', type=int,   default=30,
                        help='Frames per second for the output video (default: 30)')
    args = parser.parse_args()

    run_demo(phi=args.phi, theta=args.theta,
             sigma_d=args.sigma_d, rho=args.rho,
             seed=args.seed, n_poses=args.n_poses,
             record=args.record, record_fps=args.record_fps)
