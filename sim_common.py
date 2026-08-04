"""
sim_common.py
=============
Shared, object-agnostic simulation helpers for run_experiments_v2.py
(isolated objects, Experiment A) and run_clutter_experiments.py
(cluttered scene, Experiment B).

Generalizes run_experiments.py's originally-hardcoded, single-cylinder,
slice-based qpos indexing (`qpos[7:14]` assumed the arm always follows
exactly one freejoint) to NAME-based lookups via `model.jnt_qposadr` /
`model.jnt_dofadr`. This works regardless of how many object freejoints
precede the arm in the kinematic tree -- required for the clutter scene,
which has 3 freejoints (one per object) instead of 1.

Also implements the two determinism fixes agreed for the redesign
(response to markers' feedback + RIGOUR_LEDGER Stage 7 concern):
  1. Deterministic rho (point-cloud sparsity) downsampling -- replaces
     `rng.choice` (a second stochastic process layered on top of sigma_d)
     with an exact-fraction, order-preserving strided selection.
  2. CGN-internal determinism -- contact_graspnet_pytorch calls bare
     `np.random.*` in a few places (region cropping fallbacks); seeding
     the global numpy RNG state per trial (in addition to our own local
     `np.random.Generator`, which is unaffected by global seeding) makes
     those call sites reproducible too.
"""

import math
import numpy as np
import mujoco

ARM_JOINTS = [f'joint{i}' for i in range(1, 8)]
FINGER_JOINTS = ['finger_joint1', 'finger_joint2']
ARM_HOME_ANGLES = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])
FINGER_OPEN = 0.04


# ══════════════════════════════════════════════════════════════════════
#  Name-based qpos / dof lookups (generalizes fixed-slice indexing)
# ══════════════════════════════════════════════════════════════════════

def joint_qpos_adr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def joint_dof_adr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_dofadr[jid]


def body_freejoint_qpos_adr(model, body_name):
    """qpos start address (7 values: 3 pos + 4 quat) of a body's freejoint."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jadr = model.body_jntadr[bid]
    assert jadr >= 0, f'body {body_name} has no joint'
    return model.jnt_qposadr[jadr]


def arm_qpos_adr(model):
    return np.array([joint_qpos_adr(model, n) for n in ARM_JOINTS])


def arm_dof_adr(model):
    return np.array([joint_dof_adr(model, n) for n in ARM_JOINTS])


def finger_qpos_adr(model):
    return np.array([joint_qpos_adr(model, n) for n in FINGER_JOINTS])


def set_home_pose(model, data, object_world_poses):
    """
    Programmatic replacement for the old <keyframe> block.

    object_world_poses : dict {body_name: (x, y, z)} -- spawn each object's
    freejoint at this position with identity orientation, open the
    fingers, and set the arm to its home configuration. Generalizes to
    any number of object bodies (1 for isolated scenes, N for clutter).
    """
    data.qpos[:] = 0.
    data.qvel[:] = 0.
    data.ctrl[:] = 0.

    for body_name, (x, y, z) in object_world_poses.items():
        adr = body_freejoint_qpos_adr(model, body_name)
        data.qpos[adr:adr + 7] = [x, y, z, 1., 0., 0., 0.]

    aq = arm_qpos_adr(model)
    data.qpos[aq] = ARM_HOME_ANGLES
    data.ctrl[:7] = ARM_HOME_ANGLES

    fq = finger_qpos_adr(model)
    data.qpos[fq] = FINGER_OPEN
    data.ctrl[7] = 255.


# ══════════════════════════════════════════════════════════════════════
#  Camera
# ══════════════════════════════════════════════════════════════════════

def set_camera(model, phi_deg, theta_deg, radius, target, cam_body='perception_camera_body'):
    """Place camera on a sphere around `target`, looking at it."""
    phi, theta = math.radians(phi_deg), math.radians(theta_deg)
    dx = radius * math.cos(phi) * math.cos(theta)
    dy = radius * math.cos(phi) * math.sin(theta)
    dz = radius * math.sin(phi)
    target = np.asarray(target, dtype=float)
    cam_pos = target + np.array([dx, dy, dz])

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cam_body)
    model.body_pos[body_id] = cam_pos

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0., 0., 1.])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1., 0., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    R = np.column_stack((right, up, -forward))

    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.)
        w, x = 0.25 / s, (R[2, 1] - R[1, 2]) * s
        y, z = (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2. * math.sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2. * math.sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2. * math.sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    model.body_quat[body_id] = np.array([w, x, y, z])


def build_K(model, cam_name, img_w, img_h):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fov_y_rad = math.radians(model.cam_fovy[cam_id])
    fy = (img_h / 2.) / math.tan(fov_y_rad / 2.)
    return np.array([[fy, 0, img_w / 2.], [0, fy, img_h / 2.], [0, 0, 1]], dtype=np.float32)


def cam_to_world(pose_cam, model, data, cam_name='perception_camera'):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    cam_pos = data.cam_xpos[cam_id]
    flip = np.diag([1., -1., -1.])
    T = np.eye(4); T[:3, :3] = cam_rot; T[:3, 3] = cam_pos
    F = np.eye(4); F[:3, :3] = flip
    return T @ F @ pose_cam


# ══════════════════════════════════════════════════════════════════════
#  Perception: depth + multi-object segmentation
# ══════════════════════════════════════════════════════════════════════

def render_depth_seg(model, data, body_labels, cam_name='perception_camera',
                      img_w=640, img_h=480, sigma_d=0.0, rng=None):
    """
    Render depth + a segmentation map that labels EACH body in
    `body_labels` (dict {body_name: int label >= 1}) with its own
    integer id (0 = background). Generalizes the old binary seg_map
    (single target) to support the clutter scene (multiple targets).
    """
    if rng is None:
        rng = np.random.default_rng()
    renderer = mujoco.Renderer(model, height=img_h, width=img_w)

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth_raw = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_name)
    seg_raw = renderer.render()
    renderer.close()

    geom_ids_image = seg_raw[:, :, 0]
    seg_map = np.zeros(geom_ids_image.shape, dtype=np.int32)
    for body_name, label in body_labels.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                seg_map[geom_ids_image == gid] = label

    if seg_map.sum() == 0:
        seg_map = ((depth_raw > 0.2) & (depth_raw < 1.5)).astype(np.int32)

    if sigma_d > 0.:
        depth_noisy = np.clip(depth_raw + rng.normal(0., sigma_d, depth_raw.shape),
                               0., None).astype(np.float32)
    else:
        depth_noisy = depth_raw.copy()

    K = build_K(model, cam_name, img_w, img_h)
    return depth_noisy, K, seg_map


# ══════════════════════════════════════════════════════════════════════
#  Deterministic rho (sparsity) downsampling
# ══════════════════════════════════════════════════════════════════════

def deterministic_downsample_idx(n, rho):
    """
    Exact-fraction, order-preserving, seed-independent downsampling index.

    Replaces `rng.choice(n, size=int(n*rho), replace=False)`, which is a
    second stochastic process layered on top of sigma_d (RIGOUR_LEDGER
    Stage 7). Evenly-strided selection over the existing point order:
    keeps exactly floor(n*rho) points, same result every call for a
    given (n, rho) pair regardless of RNG state.
    """
    if rho >= 1.0 or n == 0:
        return np.arange(n)
    n_keep = max(1, int(round(n * rho)))
    idx = np.linspace(0, n - 1, n_keep)
    return np.unique(np.round(idx).astype(np.int64))


def seed_cgn_global_random(seed):
    """
    contact_graspnet_pytorch calls bare `np.random.*` in a few internal
    fallback paths (region cropping, point-count regularisation) rather
    than through a local Generator. Seeding the *global* numpy RNG state
    per trial makes those call sites reproducible without touching CGN's
    source. Does not affect our own `np.random.default_rng(seed)` local
    Generator instances, which are independent objects.
    """
    if seed is not None:
        np.random.seed(int(seed) % (2**32 - 1))


# ══════════════════════════════════════════════════════════════════════
#  IK (generalized: name-based joint lookup, not fixed qpos slices)
# ══════════════════════════════════════════════════════════════════════

def ik_move_to(model, data, target_pos, ee_site='ee_site',
                max_steps=2000, tol=0.010, lam=0.01):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    jacp = np.zeros((3, model.nv))
    aq = arm_qpos_adr(model)
    adof = arm_dof_adr(model)
    arm_ranges = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                           for n in ARM_JOINTS])

    for _ in range(max_steps):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            return True
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, adof]
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        sc = min(0.5, 0.1 / (np.linalg.norm(dq) + 1e-8))
        data.qpos[aq] += dq * sc
        np.clip(data.qpos[aq], arm_ranges[:, 0], arm_ranges[:, 1], out=data.qpos[aq])
        data.ctrl[:7] = data.qpos[aq]
        mujoco.mj_step(model, data)
    return False


def settle(model, data, steps=200):
    for _ in range(steps):
        mujoco.mj_step(model, data)


ARM_BODIES = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
              'link7', 'hand', 'left_finger', 'right_finger']


def disable_arm_collision(model):
    """
    Zero contype/conaffinity for every arm/gripper geom and return the
    saved arrays for restoration.

    Needed because the arm's parked "home" pose was calibrated against
    the original (short) cylinder; taller objects (e.g. the mustard
    bottle, ~19cm) physically overlap the stationary gripper at that
    pose. Rather than re-deriving a universal home pose, the object is
    allowed to settle under gravity/table contact only during the pre-
    capture phase; arm collision is restored before grasp execution
    (which needs it for finger-closing contact and the clutter
    inter-object collision outcome variable).
    """
    arm_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ARM_BODIES}
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in arm_bids]
    saved = (model.geom_contype[gids].copy(), model.geom_conaffinity[gids].copy())
    model.geom_contype[gids] = 0
    model.geom_conaffinity[gids] = 0
    return gids, saved


def restore_arm_collision(model, gids_saved):
    gids, (contype, conaffinity) = gids_saved
    model.geom_contype[gids] = contype
    model.geom_conaffinity[gids] = conaffinity


# ══════════════════════════════════════════════════════════════════════
#  Grasp selection helpers
# ══════════════════════════════════════════════════════════════════════

def best_grasp_overall(pred_grasps, scores):
    """Highest-scoring grasp across all segments (original single-object behaviour)."""
    best_score, best_pose, best_seg = -1., None, None
    for seg_id in pred_grasps:
        s, g = scores[seg_id], pred_grasps[seg_id]
        if len(s) == 0:
            continue
        idx = int(np.argmax(s))
        if float(s[idx]) > best_score:
            best_score, best_pose, best_seg = float(s[idx]), g[idx].copy(), seg_id
    return best_pose, best_score, best_seg


def best_grasp_for_segment(pred_grasps, scores, seg_id):
    """Highest-scoring grasp restricted to one segment id (used in
    clutter experiments, where a specific target is designated)."""
    if seg_id not in scores or len(scores[seg_id]) == 0:
        return None, None
    s, g = scores[seg_id], pred_grasps[seg_id]
    idx = int(np.argmax(s))
    return g[idx].copy(), float(s[idx])


# ══════════════════════════════════════════════════════════════════════
#  Collision outcome (Experiment B: finger vs non-target contact)
# ══════════════════════════════════════════════════════════════════════

def finger_nontarget_collision(model, data, target_body_name,
                                finger_bodies=('left_finger', 'right_finger', 'hand')):
    """
    True if, at the current data.contact state, any active contact pair
    involves a gripper body (finger/hand) and a body that is neither the
    target object nor a static scene fixture (table/floor/arm links).

    This is the "inter-object collision" outcome variable for the
    clutter experiment (Marker B): perception noise can make the CGN
    proposal geometrically plausible for the target while the physical
    approach clips a neighbouring, visually-blurred object.
    """
    finger_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in finger_bodies}
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)

    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        pair = {b1, b2}
        if not (pair & finger_bids):
            continue
        other = (pair - finger_bids)
        for ob in other:
            if ob == target_bid:
                continue
            if ob in finger_bids:
                continue
            return True
    return False
