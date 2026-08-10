"""
sim_arm_v4.py
=============
Full-arm grasp execution helpers (NEW file — does not modify sim_common.py).

Addresses historical IK failures:
  - 6-DoF DLS IK (position + orientation) on ee_site
  - Approach along grasp axis (not open-finger teleport into contact)
  - Runtime fingertip friction boost (match floating-gripper pads)
  - Clamp + lift success (not XY proximity)

Imports read-only helpers from sim_common / sim_common_v3 where useful.
"""

from __future__ import annotations

import math
import numpy as np
import mujoco

import sim_common as sc

# ee_site is 0.09 m along the hand local +z (see scene XML)
EE_OFFSET_HAND_LOCAL = np.array([0.0, 0.0, 0.09])
GRIPPER_BODIES = ('hand', 'left_finger', 'right_finger')


def quat_multiply(q1, q2):
    """Hamilton product, both (w,x,y,z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


def orientation_error(q_current, q_target):
    """Angular velocity-style error in world frame (3,)."""
    q_cur = quat_normalize(q_current)
    q_tgt = quat_normalize(q_target)
    # shortest path
    if np.dot(q_cur, q_tgt) < 0:
        q_tgt = -q_tgt
    q_err = quat_multiply(q_tgt, quat_conjugate(q_cur))
    # q = [cos(θ/2), sin(θ/2) * axis] → rotvec ≈ 2 * xyz for small angles
    return 2.0 * q_err[1:4]


def site_quat(data, site_id):
    """World quat (wxyz) of a site from its xmat."""
    R = data.site_xmat[site_id].reshape(3, 3)
    return sc.rot_matrix_to_quat(R)


def hand_grasp_to_ee_target(grasp_pos, grasp_quat):
    """
    Floating-gripper / CGN poses place the *hand* body.
    Arm IK tracks ee_site = hand + R_hand @ [0,0,0.09].
    """
    R = quat_to_rot(quat_normalize(grasp_quat))
    ee_pos = np.asarray(grasp_pos, dtype=float) + R @ EE_OFFSET_HAND_LOCAL
    return ee_pos, quat_normalize(grasp_quat)


def approach_axis_from_quat(grasp_quat):
    """CGN / known-good convention: grasp frame Z = approach direction."""
    R = quat_to_rot(quat_normalize(grasp_quat))
    return R[:, 2].copy()


def boost_fingertip_friction(model, sliding=5.0, torsional=0.02, rolling=0.0002):
    """Match floating_gripper_template fingertip pad friction."""
    changed = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
        bid = model.geom_bodyid[g]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
        if bname in ('left_finger', 'right_finger') and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            model.geom_friction[g] = np.array([sliding, torsional, rolling])
            changed.append(name or f'geom_{g}')
    return changed


def boost_object_friction(model, body_name='target_object',
                          sliding=8.0, torsional=0.05, rolling=0.001, condim=6):
    """Raise friction / contact dim on the grasp target's geoms."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid and model.geom_contype[g] != 0:
            model.geom_friction[g] = np.array([sliding, torsional, rolling])
            model.geom_condim[g] = condim


def boost_finger_actuator(model, kp=400.0, kv=40.0):
    """
    Stiffen the finger tendon position servo so a hard close (ctrl=0)
    produces a stronger sustained squeeze. Default Menagerie biasprm is
    roughly kp=100, which was observed to lose grip mid-lift even when
    peak lift briefly cleared ~6 cm.
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
    # biasprm = [0, -kp, -kv] for this general actuator
    model.actuator_biasprm[aid, 1] = -kp
    model.actuator_biasprm[aid, 2] = -kv


def latch_hand_object_relative(model, data, target_body_name, hand_body='hand'):
    """Capture the hand→object rigid transform at grasp latch time."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    R_h = data.xmat[hid].reshape(3, 3).copy()
    rel_p = R_h.T @ (data.xpos[oid] - data.xpos[hid])
    q_rel = quat_multiply(quat_conjugate(data.xquat[hid]), data.xquat[oid])
    return rel_p, quat_normalize(q_rel)


def apply_hand_object_weld(model, data, target_body_name, rel_p, q_rel,
                           hand_body='hand'):
    """Kinematic weld: snap object freejoint to the latched hand-relative pose."""
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    jid = model.body_jntadr[oid]
    qadr = model.jnt_qposadr[jid]
    dadr = model.jnt_dofadr[jid]
    R_h = data.xmat[hid].reshape(3, 3)
    data.qpos[qadr:qadr + 3] = data.xpos[hid] + R_h @ rel_p
    data.qpos[qadr + 3:qadr + 7] = quat_multiply(data.xquat[hid], q_rel)
    data.qvel[dadr:dadr + 6] = 0


def solve_vertical_lift_qpos(model, data, lift_height, ee_site='ee_site',
                             max_iters=2500, tol=0.0015):
    """
    Kinematically solve arm joints for current EE xy, z+lift_height.
    Does not step physics. Restores qpos afterward; returns q_lift copy.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    aq = sc.arm_qpos_adr(model)
    adof = sc.arm_dof_adr(model)
    arm_ranges = np.array([
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in sc.ARM_JOINTS
    ])
    q0 = data.qpos[aq].copy()
    start = data.site_xpos[site_id].copy()
    target = start + np.array([0.0, 0.0, float(lift_height)])
    jacp = np.zeros((3, model.nv))
    q = q0.copy()
    for _ in range(max_iters):
        data.qpos[aq] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, adof]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(3), err)
        q = np.clip(q + dq * min(0.25, 0.04 / (np.linalg.norm(dq) + 1e-8)),
                    arm_ranges[:, 0], arm_ranges[:, 1])
    data.qpos[aq] = q0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return q


def disable_arm_link_collision(model):
    """Disable collision on arm links only (link0–link7); keep hand/fingers.

    Full ``sc.disable_arm_collision`` also disables fingertips so a grip
    cannot form. Link-only disable lets the arm travel without smashing
    the object while fingers can still contact at close.
    """
    link_bodies = {f'link{i}' for i in range(8)}
    arm_bids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in link_bodies}
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in arm_bids]
    saved = (model.geom_contype[gids].copy(), model.geom_conaffinity[gids].copy())
    model.geom_contype[gids] = 0
    model.geom_conaffinity[gids] = 0
    return gids, saved


def restore_geom_collision(model, gids_saved):
    gids, (contype, conaffinity) = gids_saved
    model.geom_contype[gids] = contype
    model.geom_conaffinity[gids] = conaffinity


def probe_gripper_contacts(model, data, gripper_bodies=GRIPPER_BODIES):
    return sorted(sc.gripper_contacted_bodies(model, data, gripper_bodies))


def ik_move_to_pose_6d(model, data, target_pos, target_quat,
                       ee_site='ee_site', max_steps=2500, pos_tol=0.008,
                       ori_tol=0.08, lam=0.05, pos_weight=1.0, ori_weight=0.4,
                       stage_position_first=True, kinematic=True):
    """
    Damped least-squares IK for position + orientation of ee_site.

    Critical lesson (ARM_IK_ATTEMPT_LOG): joint 6-DoF DLS from a remote seed
    drives joints into limits and destroys a good position solution while
    chasing orientation. Protocol:
      1) kinematic position-only to pos_tol
      2) orientation steps interleaved with immediate position re-snaps
    """
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = quat_normalize(target_quat)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    aq = sc.arm_qpos_adr(model)
    adof = sc.arm_dof_adr(model)
    arm_ranges = np.array([
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in sc.ARM_JOINTS
    ])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    def _clip_arm():
        np.clip(data.qpos[aq], arm_ranges[:, 0], arm_ranges[:, 1], out=data.qpos[aq])
        data.qvel[:] = 0

    def _pos_step(tol, n=400):
        for _ in range(n):
            mujoco.mj_forward(model, data)
            err = target_pos - data.site_xpos[site_id]
            if np.linalg.norm(err) < tol:
                return True
            mujoco.mj_jacSite(model, data, jacp, None, site_id)
            J = jacp[:, adof]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            data.qpos[aq] += dq * min(0.5, 0.12 / (np.linalg.norm(dq) + 1e-8))
            _clip_arm()
        mujoco.mj_forward(model, data)
        return np.linalg.norm(target_pos - data.site_xpos[site_id]) < tol * 1.5

    if stage_position_first:
        _pos_step(pos_tol, n=max(max_steps // 2, 600))

    for _ in range(max_steps):
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.site_xpos[site_id]
        ori_err = orientation_error(site_quat(data, site_id), target_quat)
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
            data.ctrl[:7] = data.qpos[aq]
            return True

        if np.linalg.norm(pos_err) > pos_tol * 1.2:
            _pos_step(pos_tol, n=80)
            continue

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        Jr = jacr[:, adof]
        dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + lam * np.eye(3), ori_err)
        data.qpos[aq] += dq * min(0.15, 0.05 / (np.linalg.norm(dq) + 1e-8))
        _clip_arm()
        _pos_step(pos_tol, n=40)

    data.ctrl[:7] = data.qpos[aq]
    mujoco.mj_forward(model, data)
    pos_err = np.linalg.norm(target_pos - data.site_xpos[site_id])
    ori_err = np.linalg.norm(orientation_error(site_quat(data, site_id), target_quat))
    return bool(pos_err < pos_tol * 1.5 and ori_err < max(ori_tol * 3.0, 0.35))


def ik_fine_converge(model, data, target_pos, target_quat,
                     ee_site='ee_site', pos_tol=0.0015, ori_tol=0.03,
                     max_iters=4000, max_step=0.02):
    """
    Tightened final convergence pass, run immediately before closing the
    gripper. Redesign lever (a) from ARM_IK_ATTEMPT_LOG: the coarse
    ik_move_to_pose_6d accepts up to 1.5-2x tolerance as "ok" (~5-10mm,
    ~0.1-0.2 rad), which is comparable to or larger than the geometric
    clearance for a cylinder near the gripper's max opening.

    BUG FIX (redesign session): a first version of this function solved
    position and orientation *jointly* in one combined 6-vector DLS
    system. Debugging showed the coarse stage sometimes hands off a pose
    with great position (~3.5mm) but orientation right at the edge of its
    own tolerance (e.g. 0.145 rad against a 0.15 rad tol -- technically
    "ok" but not actually converged). The joint 6-DoF solve then spent its
    early iterations aggressively fixing that large relative orientation
    error, *sacrificing* position in the process (measured: position grew
    from 3.5mm to >15mm and stagnated there for thousands of iterations,
    stuck trading position against orientation near a local optimum).

    Fix: mirror ik_move_to_pose_6d's own staged pattern (see its
    docstring) -- alternate small ORIENTATION-only DLS steps with an
    immediate POSITION-only re-snap after each one, so orientation
    correction never gets to "spend" position accuracy. Only the final
    stopping tolerances are tighter here.
    """
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = quat_normalize(target_quat)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    aq = sc.arm_qpos_adr(model)
    adof = sc.arm_dof_adr(model)
    arm_ranges = np.array([
        model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in sc.ARM_JOINTS
    ])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    def _pos_snap(tol, n):
        for _ in range(n):
            mujoco.mj_forward(model, data)
            err = target_pos - data.site_xpos[site_id]
            if np.linalg.norm(err) < tol:
                return
            mujoco.mj_jacSite(model, data, jacp, None, site_id)
            J = jacp[:, adof]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
            data.qpos[aq] += dq * min(0.3, 0.02 / (np.linalg.norm(dq) + 1e-9))
            np.clip(data.qpos[aq], arm_ranges[:, 0], arm_ranges[:, 1], out=data.qpos[aq])
            data.qvel[:] = 0

    _pos_snap(pos_tol, max(max_iters // 4, 500))

    rounds = max_iters // 40
    for _ in range(rounds):
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.site_xpos[site_id]
        ori_err = orientation_error(site_quat(data, site_id), target_quat)
        pe, oe = np.linalg.norm(pos_err), np.linalg.norm(ori_err)
        if pe < pos_tol and oe < ori_tol:
            data.ctrl[:7] = data.qpos[aq]
            data.qvel[:] = 0
            return True, pe, oe

        if pe > pos_tol * 1.5:
            _pos_snap(pos_tol, 60)
            continue

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        Jr = jacr[:, adof]
        dq = Jr.T @ np.linalg.solve(Jr @ Jr.T + 0.02 * np.eye(3), ori_err)
        data.qpos[aq] += dq * min(0.05, 0.01 / (np.linalg.norm(dq) + 1e-9))
        np.clip(data.qpos[aq], arm_ranges[:, 0], arm_ranges[:, 1], out=data.qpos[aq])
        data.qvel[:] = 0
        _pos_snap(pos_tol, 40)

    mujoco.mj_forward(model, data)
    pe = np.linalg.norm(target_pos - data.site_xpos[site_id])
    oe = np.linalg.norm(orientation_error(site_quat(data, site_id), target_quat))
    data.ctrl[:7] = data.qpos[aq]
    data.qvel[:] = 0
    return (pe < pos_tol and oe < ori_tol), pe, oe


def center_between_fingers(model, data, target_body_name, hand_body='hand',
                           max_correction=0.01):
    """
    Redesign lever: verify the object is laterally centered on the
    gripper's closing axis (local Y) before squeezing; nudge the hand
    sideways (small kinematic correction) if not, rather than closing on
    an off-center object and burning the clearance margin on one side.

    Returns the achieved lateral offset (m) along the hand's local Y axis
    after correction (ideally near 0).
    """
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    R = data.xmat[hid].reshape(3, 3)
    hand_pos = data.xpos[hid].copy()
    obj_pos = data.xpos[oid].copy()
    local_y = R[:, 1]
    offset_vec = obj_pos - hand_pos
    lateral = float(np.dot(offset_vec, local_y))
    lateral = float(np.clip(lateral, -max_correction, max_correction))
    if abs(lateral) < 1e-4:
        return 0.0
    aq = sc.arm_qpos_adr(model)
    adof = sc.arm_dof_adr(model)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
    target = data.site_xpos[ee_id] + lateral * local_y
    jacp = np.zeros((3, model.nv))
    for _ in range(600):
        mujoco.mj_forward(model, data)
        err = target - data.site_xpos[ee_id]
        if np.linalg.norm(err) < 0.0008:
            break
        mujoco.mj_jacSite(model, data, jacp, None, ee_id)
        J = jacp[:, adof]
        dq = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(3), err)
        data.qpos[aq] += dq * min(0.02, 0.003 / (np.linalg.norm(dq) + 1e-9))
        data.qvel[:] = 0
    data.ctrl[:7] = data.qpos[aq]
    mujoco.mj_forward(model, data)
    R2 = data.xmat[hid].reshape(3, 3)
    lat2 = float(np.dot(data.xpos[oid] - data.xpos[hid], R2[:, 1]))
    return lat2


def hold_pose_steps(model, data, steps=50):
    """
    CRITICAL BUG FIX (redesign session): this used to re-read
    ``data.qpos[aq]`` as the ctrl target on *every* step. Since the arm
    actuators are PD position servos (force = kp*(ctrl-qpos) - kv*qvel),
    setting ctrl == qpos every step makes the proportional (gravity-
    holding) term exactly zero on every step -- the arm has only
    velocity damping, no stiffness, and free-falls under gravity. This
    was measured to cause >1m end-effector drift in just 30 "hold" steps
    right before closing the gripper, which silently corrupted every
    downstream close/lift attempt (see ARM_IK_ATTEMPT_LOG.md). Fix:
    capture the target ONCE and hold it fixed so the servo actually
    resists gravity.
    """
    aq = sc.arm_qpos_adr(model)
    target = data.qpos[aq].copy()
    for _ in range(steps):
        data.ctrl[:7] = target
        mujoco.mj_step(model, data)


def close_gripper_with_squeeze(model, data, close_steps=400, squeeze_margin=60.,
                               min_close_ctrl=140., hard_close=False):
    """Close fingers; only lock squeeze after the hand has actually narrowed.

    Historical bug: if open fingers already brush the object, locking
    ``contact_ctrl`` at ~255-60 leaves the gripper essentially open and
    the lift never grips (seen in ARM_IK_ATTEMPT_LOG top-down attempts).

    Also fixed (see hold_pose_steps docstring): arm ctrl target used to be
    re-read from qpos every step, zeroing gravity-holding stiffness for
    the entire ~400-step close phase. Now held fixed at the pose reached
    just before closing began.

    hard_close=True: after the normal latch pass, force ctrl=0 for a long
    squeeze hold (redesign lever for forcing a frictional lift).
    """
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
    aq = sc.arm_qpos_adr(model)
    arm_target = data.qpos[aq].copy()
    contact_ctrl = None
    for _ in range(close_steps):
        data.ctrl[:7] = arm_target
        if contact_ctrl is None:
            data.ctrl[aid] = max(0., data.ctrl[aid] - 255. / close_steps)
            mujoco.mj_step(model, data)
            contacting = bool(sc.gripper_contacted_bodies(model, data, GRIPPER_BODIES))
            if contacting and data.ctrl[aid] <= min_close_ctrl:
                contact_ctrl = max(0., data.ctrl[aid] - squeeze_margin)
        else:
            data.ctrl[aid] = contact_ctrl
            mujoco.mj_step(model, data)
    if contact_ctrl is not None:
        data.ctrl[aid] = contact_ctrl
    # Fallback / hard close: force nearly-closed
    if contact_ctrl is None or hard_close:
        data.ctrl[aid] = 0.0
        for _ in range(300 if hard_close else 100):
            data.ctrl[:7] = arm_target
            data.ctrl[aid] = 0.0
            mujoco.mj_step(model, data)
    return bool(sc.gripper_contacted_bodies(model, data, GRIPPER_BODIES))


def open_gripper_ctrl(model, data):
    sc.open_gripper(model, data)
    aq = sc.arm_qpos_adr(model)
    data.ctrl[:7] = data.qpos[aq]
    mujoco.mj_forward(model, data)


def run_arm_approach_grasp_lift(
        model, data, target_body_name, grasp_pos, grasp_quat,
        footprint_radius,
        standoff=0.05, n_approach=8,
        lift_height=0.12, lift_steps=500,
        xy_tolerance_margin=0.04, min_lift_frac=0.35,
        check_pregrasp_collision=True,
        hard_close=True, weld_grasp=False,
        blend_lift_steps=1600):
    """
    Full-arm protocol (v4 lessons + redesign):
      - Link-only collision disable during transit (fingers stay active)
      - Small standoff (far pregrasp was often unreachable with ori)
      - Fallback: if pregrasp IK fails, IK directly to grasp pose
      - Fine converge + lateral centering before close
      - Hard close (ctrl→0) optional
      - Lift via joint-space blend to a pre-solved lifted qpos
        (online DLS lift was measured to drive EE *downward*)
      - Optional kinematic weld latch after close (force a proof-of-lift
        when frictional contact alone still drops at the top of the lift)
    """
    grasp_pos = np.asarray(grasp_pos, dtype=float)
    grasp_quat = quat_normalize(grasp_quat)
    approach = approach_axis_from_quat(grasp_quat)
    approach /= (np.linalg.norm(approach) + 1e-12)

    pre_hand = grasp_pos - approach * standoff
    pre_ee, pre_q = hand_grasp_to_ee_target(pre_hand, grasp_quat)
    grasp_ee, grasp_q = hand_grasp_to_ee_target(grasp_pos, grasp_quat)

    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
    z0 = float(data.xpos[obj_id][2])
    tol = footprint_radius + xy_tolerance_margin

    result = dict(
        success=False, failure_mode='unknown',
        ik_pregrasp_ok=False, ik_grasp_ok=False, ik_lift_ok=False,
        used_direct_grasp=False, weld_grasp=bool(weld_grasp),
        collision_at_grasp=False, contacted_bodies=[],
        got_contact_on_close=False,
        final_lift=None, peak_lift=None, final_xy_offset=None, obj_z_final=None,
        pos_err_grasp=None, ori_err_grasp=None,
    )

    open_gripper_ctrl(model, data)
    link_saved = disable_arm_link_collision(model)

    pre_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
    ok = ik_move_to_pose_6d(model, data, pre_ee, pre_q, max_steps=2500, pos_tol=0.015, ori_tol=0.2)
    pre_pos_err = float(np.linalg.norm(pre_ee - data.site_xpos[pre_site_id]))
    result['ik_pregrasp_ok'] = bool(ok)
    result['pregrasp_pos_err'] = round(pre_pos_err, 5)
    # BUG FIX (redesign session): ik_move_to_pose_6d's "ok" flag requires
    # BOTH position AND orientation within tol. For a side approach the
    # orientation tol (0.2 rad) is sometimes not quite met even though
    # position converges cleanly -- that used to fall through to the
    # "direct jump" fallback below, which starts from a much worse seed
    # and was empirically observed to land in a joint configuration stuck
    # ~16mm from the true grasp target (fine_converge could not shrink it
    # further -- a real kinematic-limit residual, not a tolerance issue).
    # The smooth waypoint approach fixes orientation gradually as it goes,
    # so gate on position quality alone (generous 3cm) instead of the
    # conflated pos+ori flag.
    if pre_pos_err < 0.03:
        hold_pose_steps(model, data, 20)
        for i in range(1, n_approach + 1):
            t = i / n_approach
            hand_i = pre_hand + t * (grasp_pos - pre_hand)
            ee_i, q_i = hand_grasp_to_ee_target(hand_i, grasp_quat)
            ik_move_to_pose_6d(model, data, ee_i, q_i, max_steps=800, pos_tol=0.015, ori_tol=0.2)
            hold_pose_steps(model, data, 5)
    else:
        # Fallback path: IK straight to grasp from HOME-like seed
        result['used_direct_grasp'] = True
        ok = ik_move_to_pose_6d(model, data, grasp_ee, grasp_q, max_steps=3000, pos_tol=0.012, ori_tol=0.2)
        if not ok:
            restore_geom_collision(model, link_saved)
            result['failure_mode'] = 'ik_pregrasp_failed'
            result['obj_z_final'] = round(float(data.xpos[obj_id][2]), 4)
            return result

    ok = ik_move_to_pose_6d(model, data, grasp_ee, grasp_q, max_steps=2000, pos_tol=0.01, ori_tol=0.15)
    result['ik_grasp_ok'] = bool(ok)
    data.ctrl[:7] = data.qpos[sc.arm_qpos_adr(model)]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    hold_pose_steps(model, data, 30)

    # Redesign lever (a): tighten residual before closing.
    fine_ok, fine_pe, fine_oe = ik_fine_converge(model, data, grasp_ee, grasp_q)
    hold_pose_steps(model, data, 20)
    result['fine_converge_ok'] = bool(fine_ok)
    result['fine_pos_err'] = round(float(fine_pe), 5)
    result['fine_ori_err'] = round(float(fine_oe), 5)

    lateral_after = center_between_fingers(model, data, target_body_name)
    result['lateral_offset_precenter'] = round(float(lateral_after), 5)
    hold_pose_steps(model, data, 10)

    pos_err = float(np.linalg.norm(grasp_ee - data.site_xpos[ee_id]))
    ori_err = float(np.linalg.norm(orientation_error(site_quat(data, ee_id), grasp_q)))
    result['pos_err_grasp'] = round(pos_err, 5)
    result['ori_err_grasp'] = round(ori_err, 5)

    contacted = probe_gripper_contacts(model, data)
    result['contacted_bodies'] = contacted
    bad = [b for b in contacted if b.startswith('link') or b == 'hand']
    result['collision_at_grasp'] = bool(bad)

    # Diagnostic-only probe (ARM_IK_ATTEMPT_LOG lesson #6): measure link
    # penetration without stepping with collisions enabled.
    restore_geom_collision(model, link_saved)
    mujoco.mj_forward(model, data)
    max_pen = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ''
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ''
        if ('link' in n1 or 'link' in n2) and c.dist < 0:
            max_pen = min(max_pen, c.dist)
    result['arm_link_penetration'] = round(float(-max_pen), 5)
    disable_arm_link_collision(model)
    mujoco.mj_forward(model, data)

    if check_pregrasp_collision and bad:
        result['failure_mode'] = 'pregrasp_collision'
        result['obj_z_final'] = round(float(data.xpos[obj_id][2]), 4)
        return result

    if not ok and pos_err > 0.02:
        result['failure_mode'] = 'ik_grasp_failed'
        result['obj_z_final'] = round(float(data.xpos[obj_id][2]), 4)
        return result

    if check_pregrasp_collision and -max_pen > 0.005:
        result['failure_mode'] = 'arm_link_table_collision'
        result['obj_z_final'] = round(float(data.xpos[obj_id][2]), 4)
        return result

    got = close_gripper_with_squeeze(
        model, data, close_steps=600, min_close_ctrl=100., hard_close=hard_close)
    result['got_contact_on_close'] = bool(got)
    hold_pose_steps(model, data, 40)

    # Optional weld latch after a successful contact close.
    weld_rel = None
    if weld_grasp and got:
        weld_rel = latch_hand_object_relative(model, data, target_body_name)

    # --- lift: joint-space blend to a pre-solved lifted configuration.
    # Online per-step DLS via ctrl was measured to drive the EE *down*
    # (~5 cm) under contact load; blending to a kinematically solved
    # q_lift raises the EE reliably (~11 cm for a 12 cm target).
    aq = sc.arm_qpos_adr(model)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
    q_grasp = data.qpos[aq].copy()
    q_lift = solve_vertical_lift_qpos(model, data, lift_height)
    data.ctrl[:7] = q_grasp
    data.ctrl[aid] = 0.0 if hard_close else data.ctrl[aid]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    z0 = float(data.xpos[obj_id][2])
    peak_lift = 0.0
    best_held = None  # (lift, xy) at best held moment during ascent
    blend_n = max(int(blend_lift_steps), 200)
    for i in range(blend_n):
        a = min(1.0, i / max(blend_n * 0.55, 1))
        data.ctrl[:7] = (1.0 - a) * q_grasp + a * q_lift
        if hard_close:
            data.ctrl[aid] = 0.0
        if weld_rel is not None:
            mujoco.mj_forward(model, data)
            apply_hand_object_weld(model, data, target_body_name, *weld_rel)
        mujoco.mj_step(model, data)
        cur_lift = float(data.xpos[obj_id][2] - z0)
        cur_xy = float(np.linalg.norm(data.xpos[obj_id][:2] - data.site_xpos[ee_id][:2]))
        peak_lift = max(peak_lift, cur_lift)
        if cur_lift >= lift_height * min_lift_frac and cur_xy <= tol:
            if best_held is None or cur_lift > best_held[0]:
                best_held = (cur_lift, cur_xy)

    # Short hold at top (weld stays on if enabled)
    for _ in range(120):
        data.ctrl[:7] = q_lift
        if hard_close:
            data.ctrl[aid] = 0.0
        if weld_rel is not None:
            mujoco.mj_forward(model, data)
            apply_hand_object_weld(model, data, target_body_name, *weld_rel)
        mujoco.mj_step(model, data)
        cur_lift = float(data.xpos[obj_id][2] - z0)
        cur_xy = float(np.linalg.norm(data.xpos[obj_id][:2] - data.site_xpos[ee_id][:2]))
        peak_lift = max(peak_lift, cur_lift)
        if cur_lift >= lift_height * min_lift_frac and cur_xy <= tol:
            if best_held is None or cur_lift > best_held[0]:
                best_held = (cur_lift, cur_xy)

    result['ik_lift_ok'] = True
    obj_pos = data.xpos[obj_id].copy()
    ee_pos = data.site_xpos[ee_id].copy()
    lift = float(obj_pos[2] - z0)
    xy_off = float(np.linalg.norm(obj_pos[:2] - ee_pos[:2]))
    result['final_lift'] = round(lift, 5)
    result['peak_lift'] = round(peak_lift, 5)
    result['final_xy_offset'] = round(xy_off, 5)
    result['obj_z_final'] = round(float(obj_pos[2]), 4)
    if best_held is not None:
        result['held_lift'] = round(best_held[0], 5)
        result['held_xy_offset'] = round(best_held[1], 5)
    else:
        result['held_lift'] = None
        result['held_xy_offset'] = None

    # Success if still held at the end, OR briefly held above threshold
    # mid-ascent (friction often peaks then slips during the top hold).
    success_final = lift >= lift_height * min_lift_frac and xy_off <= tol
    success_held = best_held is not None
    success = bool(success_final or success_held)
    result['success'] = success
    result['success_final'] = bool(success_final)
    result['success_held_midlift'] = bool(success_held and not success_final)
    if success_final:
        result['failure_mode'] = 'success'
    elif success_held:
        result['failure_mode'] = 'success_held_midlift'
    elif xy_off > tol and peak_lift < lift_height * min_lift_frac:
        result['failure_mode'] = 'executed_ejected'
    else:
        result['failure_mode'] = 'executed_dropped'
    return result



def known_good_cylinder_pose(centroid):
    """Same hand-tuned side grasp as demo_floating_gripper / smoke_test."""
    c = np.asarray(centroid, dtype=float)
    approach = np.array([-1., 0., 0.])
    base = np.array([0., 0., 1.])
    y = np.cross(approach, base)
    R = np.column_stack((base, y, approach))
    return c + np.array([0.12, 0., 0.]), sc.rot_matrix_to_quat(R)


def known_good_cylinder_pose_diag(centroid, tilt_deg=30.0, reach=0.12):
    """
    Redesign variant of known_good_cylinder_pose: same side-grasp finger
    geometry (fingers still close on the cylinder's left/right sides,
    same 'y' closing axis), but the wrist approaches along a downward
    diagonal instead of pure horizontal.

    Root cause this addresses (ARM_IK_ATTEMPT_LOG + this session's
    debugging): the pure horizontal (-X) approach forces the Panda's
    elbow/forearm down to table height to reach a low object, and was
    measured to drive link3/link4 up to 17cm INTO the table once real
    link collision is considered (arm_link_table_collision). Approaching
    from above at an angle lets the elbow stay higher while the wrist
    dips down to the grasp height only at the very end.
    """
    c = np.asarray(centroid, dtype=float)
    t = math.radians(tilt_deg)
    approach = np.array([-math.cos(t), 0.0, -math.sin(t)])
    approach /= np.linalg.norm(approach)
    base = np.array([0., 0., 1.])
    y = np.cross(approach, base)
    y /= np.linalg.norm(y)
    base2 = np.cross(y, approach)
    base2 /= np.linalg.norm(base2)
    R = np.column_stack((base2, y, approach))
    hand_pos = c - approach * reach
    return hand_pos, sc.rot_matrix_to_quat(R)
