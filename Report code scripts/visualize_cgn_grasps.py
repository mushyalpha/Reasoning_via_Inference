"""
visualize_cgn_grasps.py
=======================
CGN Grasp Visualisation Engine – full pipeline viewer for the MSc report.

Shows exactly what the paper screenshot shows:
  • Point cloud of the MuJoCo scene
  • All proposed grasp wireframes coloured by confidence score (viridis)
  • Best grasp highlighted in RED / thicker lines
  • Saves PNG screenshots automatically for the MSc report

Three views are produced and saved:
  1. Camera-frame view  (the natural CGN output – matches paper screenshots)
  2. World-frame view   (after cam→world transform – shows grasps vs. robot arm)
  3. Matplotlib figure  (top-N grasp scores bar chart + depth map overlay)

Usage
-----
  # Run CGN live on the MuJoCo scene and visualise:
  python visualize_cgn_grasps.py

  # Save predictions first (fast re-visualise later):
  python visualize_cgn_grasps.py --save

  # Load a saved predictions file (no CGN needed):
  python visualize_cgn_grasps.py --from_file results/cgn_predictions.npz

  # Show only world-frame view:
  python visualize_cgn_grasps.py --view world

  # Control camera noise / sparsity (matching experiment conditions):
  python visualize_cgn_grasps.py --sigma_d 0.02 --rho 0.8 --phi 45 --theta 0

Options
-------
  --phi         Camera elevation angle (deg)            default: 45
  --theta       Camera azimuth angle (deg)              default: 0
  --sigma_d     Depth noise std-dev (m)                 default: 0.0
  --rho         Point-cloud keep fraction               default: 1.0
  --n_grasps    Number of grasps to draw                default: all
  --view        camera | world | both                   default: both
  --from_file   Path to a saved .npz predictions file
  --save        Save predictions .npz + screenshots to results/
  --out_dir     Where to write screenshots               default: results/figures
  --seed        RNG seed                                 default: 42
"""

import os
import sys
import math
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless for screenshot saving
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import mujoco

# ── CGN source on path ────────────────────────────────────────────────────────
_PROJECT  = os.path.dirname(os.path.abspath(__file__))
_CGN_REPO = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
_CGN_SRC  = os.path.join(_CGN_REPO, 'contact_graspnet_pytorch')
for _p in [_CGN_REPO, _CGN_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from contact_grasp_estimator import GraspEstimator
import config_utils
from checkpoints import CheckpointIO
from data import load_available_input_data
import torch

# ── Try Open3D ────────────────────────────────────────────────────────────────
try:
    import open3d as o3d
    HAS_O3D = True
    print('[Viz] Open3D', o3d.__version__, 'available (OK)')
except ImportError:
    HAS_O3D = False
    print('[Viz] Open3D not found – will use matplotlib 3D fallback only.')

try:
    import mesh_utils as _mu
    HAS_MESH = True
except ImportError:
    HAS_MESH = False

# ── Constants (match demo_grasp.py) ──────────────────────────────────────────
SCENE_XML   = os.path.join(_PROJECT, 'grasp_scene_v2.xml')
CGN_ROOT    = os.path.join(_PROJECT, 'contact_graspnet_pytorch')
CKPT_DIR    = os.path.join(CGN_ROOT, 'checkpoints', 'contact_graspnet')
CAM_NAME    = 'perception_camera'
TARGET_BODY = 'target_object'
IMG_W, IMG_H = 640, 480
TARGET_POS  = np.array([0.5, 0., 0.455])
GRIPPER_WIDTH = 0.08          # metres – Panda finger opening


# =============================================================================
#  CGN helpers
# =============================================================================

def load_cgn():
    global_config = config_utils.load_config(CKPT_DIR, batch_size=1,
                                              arg_configs=[])
    estimator = GraspEstimator(global_config)
    ckpt_io   = CheckpointIO(
        checkpoint_dir=os.path.join(CKPT_DIR, 'checkpoints'),
        model=estimator.model)
    ckpt_io.load('model.pt')
    estimator.model.eval()
    print('[CGN] Model loaded (CPU, eval mode).')
    return estimator


def set_camera(model, phi_deg, theta_deg, radius=0.8,
               target=TARGET_POS):
    """Place the MuJoCo perception camera on a sphere around the target."""
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


def render_scene(model, data, sigma_d=0.0, rng=None):
    """Render depth + segmentation + RGB from MuJoCo scene."""
    if rng is None:
        rng = np.random.default_rng()

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    # ── depth ──
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    depth_raw = renderer.render().copy()
    renderer.disable_depth_rendering()

    # ── segmentation ──
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    seg_raw = renderer.render()

    # ── RGB ──
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    rgb = renderer.render().copy()
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
        depth_noisy = np.clip(
            depth_raw + rng.normal(0., sigma_d, depth_raw.shape), 0., None
        ).astype(np.float32)

    return depth_noisy, build_K(model), seg_map, rgb, depth_raw


def run_cgn(depth, K, seg_map, estimator, rho=1.0, rng=None):
    """Run CGN inference; returns (pred_grasps_cam, scores, contact_pts, pc_full, pc_colors)."""
    if rng is None:
        rng = np.random.default_rng()

    tmp = os.path.join(_PROJECT, '_tmp_viz.npz')
    np.savez(tmp, depth=depth, K=K, seg=seg_map)
    segmap, rgb_ld, depth_in, cam_K, _, _ = load_available_input_data(tmp, K=None)

    pc_full, pc_segs, pc_colors = estimator.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb_ld,
        skip_border_objects=False, z_range=[0.1, 2.0])

    if rho < 1.0 and len(pc_full) > 0:
        n = max(1, int(len(pc_full) * rho))
        idx = rng.choice(len(pc_full), size=n, replace=False)
        pc_full   = pc_full[idx]
        pc_colors = pc_colors[idx] if pc_colors is not None else None
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
    return pred_grasps, scores, contact_pts, pc_full, pc_colors


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


def transform_grasps_to_world(pred_grasps_cam, model, data):
    """Return a new dict with all grasp poses in world frame."""
    pred_grasps_world = {}
    for k, grasps in pred_grasps_cam.items():
        if len(grasps) == 0:
            pred_grasps_world[k] = grasps
            continue
        world_grasps = np.stack([cam_to_world(g, model, data) for g in grasps])
        pred_grasps_world[k] = world_grasps
    return pred_grasps_world


# =============================================================================
#  Open3D Visualisation
# =============================================================================

def _make_grasp_lineset(grasps, gripper_openings, colors_list):
    """
    Build an Open3D LineSet for a batch of grasp poses.
    Each grasp is the classic 7-point Panda gripper wireframe:
      centre → wrist → left_tip → left_finger
                     → right_tip → right_finger
    """
    # Panda gripper control points (unit frame, before scaling)
    cp = np.array([
        [0,   0,   0],       # 0: base centre
        [0,   0,   0.059],   # 1: wrist
        [-GRIPPER_WIDTH/2, 0, 0.059],   # 2: left tip base
        [-GRIPPER_WIDTH/2, 0, 0.105],   # 3: left finger tip
        [ GRIPPER_WIDTH/2, 0, 0.059],   # 4: right tip base
        [ GRIPPER_WIDTH/2, 0, 0.105],   # 5: right finger tip
        [0,   0,   0.059],   # 6: wrist repeat (for palm bar)
    ], dtype=np.float64)

    # Edges: 0-1 (stem), 2-4 (palm bar), 2-3 (left), 4-5 (right)
    edges = [(0,1), (2,4), (2,3), (4,5)]

    all_pts  = []
    all_edges = []
    all_colors = []

    for gi, (g, opening) in enumerate(zip(grasps, gripper_openings)):
        # Scale finger opening
        pts = cp.copy()
        pts[2,0] = -opening / 2
        pts[3,0] = -opening / 2
        pts[4,0] =  opening / 2
        pts[5,0] =  opening / 2

        # Transform by grasp pose
        R_g = g[:3, :3]
        t_g = g[:3,  3]
        pts_world = (R_g @ pts.T).T + t_g

        base_idx = len(all_pts)
        all_pts.extend(pts_world.tolist())
        col = colors_list[gi] if gi < len(colors_list) else (0.2, 1.0, 0.2)
        for e in edges:
            all_edges.append([base_idx + e[0], base_idx + e[1]])
            all_colors.append(list(col))

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_pts)
    ls.lines  = o3d.utility.Vector2iVector(all_edges)
    ls.colors = o3d.utility.Vector3dVector(all_colors)
    return ls


def _score_colors(scores_arr, cmap_name='plasma'):
    """Map score array -> list of (r,g,b) tuples using a matplotlib colormap."""
    cmap = matplotlib.colormaps[cmap_name]
    mn, mx = scores_arr.min(), scores_arr.max()
    denom = max(mx - mn, 1e-8)
    return [cmap((s - mn) / denom)[:3] for s in scores_arr]


def _build_pcd(pc_full, pc_colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_full)
    if pc_colors is not None and len(pc_colors) == len(pc_full):
        c = pc_colors.astype(np.float64)
        if c.max() > 1.0:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(c)
    else:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
    return pcd


def _add_world_axes(vis, length=0.15):
    """Draw small XYZ axis at world origin."""
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=length, origin=[0, 0, 0])
    vis.add_geometry(axes)


def open3d_render_offscreen(pc_full, pred_grasps, scores, pc_colors=None,
                            title='CGN Grasp Proposals', n_grasps=None,
                            show_axes=True, screenshot_path=None):
    """
    Render grasp proposals HEADLESSLY using Open3D offscreen renderer.
    No GUI window required — saves directly to screenshot_path PNG.

    pc_full        : (N,3) point cloud
    pred_grasps    : dict {obj_id: (M,4,4)}
    scores         : dict {obj_id: (M,)}
    screenshot_path: where to write the PNG (required)
    """
    if not HAS_O3D:
        print('[Viz] Open3D not available, skipping 3D render.')
        return
    if screenshot_path is None:
        print('[Viz] No screenshot_path given, skipping offscreen render.')
        return

    print(f'\n[Viz] Offscreen rendering: {title}')

    # Collect all geometries
    geometries = []

    pcd = _build_pcd(pc_full, pc_colors)
    geometries.append(pcd)

    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        geometries.append(axes)

    total_drawn = 0
    for obj_id in pred_grasps:
        grasps_k = pred_grasps[obj_id]
        scores_k = scores[obj_id]
        if len(grasps_k) == 0:
            continue

        order = np.argsort(scores_k)[::-1]
        if n_grasps is not None:
            order = order[:n_grasps]
        grasps_sorted = grasps_k[order]
        scores_sorted = scores_k[order]

        colors_list = _score_colors(scores_sorted, cmap_name='plasma')
        gripper_openings = np.ones(len(grasps_sorted)) * GRIPPER_WIDTH
        ls = _make_grasp_lineset(grasps_sorted, gripper_openings, colors_list)
        geometries.append(ls)

        # Best grasp in red
        ls_best = _make_grasp_lineset(
            grasps_sorted[:1], gripper_openings[:1], [(1.0, 0.0, 0.0)])
        geometries.append(ls_best)

        total_drawn += len(grasps_sorted)
        print(f'  Segment {obj_id}: {len(grasps_sorted)} grasps '
              f'(best score={scores_sorted[0]:.4f})')

    print(f'  Total grasp proposals: {total_drawn}')

    # ── Compute a sensible camera viewpoint from the point cloud centre ──
    pts = np.asarray(pcd.points)
    centre = pts.mean(axis=0) if len(pts) > 0 else np.zeros(3)

    # Use Open3D draw_geometries with a fixed lookat for offscreen rendering
    # We render via the headless OffscreenRenderer
    try:
        # Open3D >= 0.13 headless rendering path
        render = o3d.visualization.rendering.OffscreenRenderer(1280, 800)
        render.scene.set_background([0.12, 0.12, 0.18, 1.0])

        mat_pc = o3d.visualization.rendering.MaterialRecord()
        mat_pc.shader = 'defaultUnlit'
        mat_pc.point_size = 3.0
        render.scene.add_geometry('pointcloud', pcd, mat_pc)

        mat_line = o3d.visualization.rendering.MaterialRecord()
        mat_line.shader = 'unlitLine'
        mat_line.line_width = 2.0

        for gi, geom in enumerate(geometries[1:]):   # skip pcd already added
            if isinstance(geom, o3d.geometry.LineSet):
                render.scene.add_geometry(f'grasp_{gi}', geom, mat_line)
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                mat_m = o3d.visualization.rendering.MaterialRecord()
                mat_m.shader = 'defaultUnlit'
                render.scene.add_geometry(f'mesh_{gi}', geom, mat_m)

        # Set camera: look from slightly above and to the side
        bounds = render.scene.bounding_box
        render.setup_camera(60.0, bounds, bounds.get_center())

        img = render.render_to_image()
        o3d.io.write_image(screenshot_path, img)
        print(f'  [Screenshot] Saved -> {screenshot_path}')
        del render

    except Exception as e:
        print(f'  [Viz] Offscreen render failed ({e}). Trying draw_geometries fallback...')
        # Fallback: matplotlib-based top-down scatter (always works)
        _fallback_scatter_render(pc_full, pred_grasps, scores, screenshot_path, title)


def _fallback_scatter_render(pc_full, pred_grasps, scores, out_path, title):
    """Pure matplotlib 3D scatter fallback if Open3D offscreen fails."""
    all_pos, all_sc = [], []
    for k in pred_grasps:
        for g, s in zip(pred_grasps[k], scores[k]):
            all_pos.append(g[:3, 3])
            all_sc.append(s)
    if not all_pos:
        return
    all_pos = np.array(all_pos)
    all_sc  = np.array(all_sc)

    fig = plt.figure(figsize=(12, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')
    step = max(1, len(pc_full) // 3000)
    pc_s = pc_full[::step]
    ax.scatter(pc_s[:,0], pc_s[:,1], pc_s[:,2], c='#555577', s=1, alpha=0.35)
    sc = ax.scatter(all_pos[:,0], all_pos[:,1], all_pos[:,2],
                    c=all_sc, cmap='plasma', s=20, alpha=0.9)
    fig.colorbar(sc, ax=ax, shrink=0.55, label='Score')
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors='white')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  [Fallback scatter] Saved -> {out_path}')


# =============================================================================
#  Matplotlib figures (always saved, no GUI needed)
# =============================================================================

def save_matplotlib_figures(depth_raw, rgb, seg_map,
                             pred_grasps_cam, scores, pc_full,
                             out_dir, phi, theta, sigma_d, rho):
    """
    Save a rich multi-panel figure for the MSc report.
    Four panels:
      1. RGB render of the scene
      2. Depth map (colourised)
      3. Segmentation mask overlay
      4. Bar chart of top-20 grasp confidence scores
    """
    os.makedirs(out_dir, exist_ok=True)

    # Collect all scores
    all_scores = []
    for k in scores:
        all_scores.extend(scores[k].tolist())
    all_scores = np.array(sorted(all_scores, reverse=True))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             facecolor='#1a1a2e')
    fig.suptitle(
        f'Contact-GraspNet Perception Pipeline\n'
        f'φ={phi}° θ={theta}° σ_d={sigma_d} ρ={rho}  '
        f'| {len(all_scores)} grasp candidates',
        color='white', fontsize=14, fontweight='bold')

    style = dict(facecolor='#16213e', title_color='white')

    # Panel 1: RGB
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title('RGB Render (MuJoCo scene)', color='white', fontsize=11)
    ax.axis('off')
    ax.set_facecolor('#16213e')

    # Panel 2: Depth map
    ax = axes[0, 1]
    dm = ax.imshow(depth_raw, cmap='inferno', vmin=0.3, vmax=1.5)
    ax.set_title('Depth Map', color='white', fontsize=11)
    ax.axis('off')
    ax.set_facecolor('#16213e')
    cbar = fig.colorbar(dm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (m)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Panel 3: Segmentation overlay
    ax = axes[1, 0]
    ax.imshow(rgb)
    seg_overlay = np.zeros((*seg_map.shape, 4))
    seg_overlay[seg_map == 1] = [1.0, 0.4, 0.0, 0.6]   # orange for target
    ax.imshow(seg_overlay)
    ax.set_title('Segmentation Mask (target = orange)', color='white', fontsize=11)
    ax.axis('off')
    ax.set_facecolor('#16213e')

    # Panel 4: Score distribution
    ax = axes[1, 1]
    ax.set_facecolor('#0f3460')
    n_show = min(30, len(all_scores))
    bar_colors = matplotlib.colormaps['plasma'](np.linspace(0.9, 0.2, n_show))
    bars = ax.bar(range(n_show), all_scores[:n_show], color=bar_colors)
    ax.set_xlabel('Grasp rank', color='white', fontsize=10)
    ax.set_ylabel('Confidence score', color='white', fontsize=10)
    ax.set_title(f'Top-{n_show} Grasp Confidence Scores', color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if len(all_scores) > 0:
        ax.axhline(all_scores[0], color='red', linewidth=1.5,
                   linestyle='--', label=f'Best: {all_scores[0]:.4f}')
        ax.legend(facecolor='#0f3460', labelcolor='white', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    tag = f'phi{int(phi)}_theta{int(theta)}_sd{sigma_d}_rho{rho}'
    out_path = os.path.join(out_dir, f'cgn_perception_{tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  [Figure 1] Saved → {out_path}')
    return out_path


def save_score_heatmap(pred_grasps_cam, scores, pc_full, out_dir, tag=''):
    """
    Save a 3D scatter plot (matplotlib) showing point cloud coloured by
    proximity to grasp contact points, for the MSc report.
    """
    os.makedirs(out_dir, exist_ok=True)

    all_scores = []
    all_positions = []
    for k in pred_grasps_cam:
        g_arr = pred_grasps_cam[k]
        s_arr = scores[k]
        for gi, si in zip(g_arr, s_arr):
            all_positions.append(gi[:3, 3])
            all_scores.append(float(si))

    if len(all_positions) == 0:
        return None

    all_positions = np.array(all_positions)
    all_scores    = np.array(all_scores)

    fig = plt.figure(figsize=(12, 8), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')

    # Point cloud (grey)
    if len(pc_full) > 0:
        step = max(1, len(pc_full) // 3000)   # thin for readability
        pc_s = pc_full[::step]
        ax.scatter(pc_s[:,0], pc_s[:,1], pc_s[:,2],
                   c='#555577', s=1, alpha=0.4, label='Point cloud')

    # Grasp origins coloured by score
    scatter = ax.scatter(
        all_positions[:,0], all_positions[:,1], all_positions[:,2],
        c=all_scores, cmap='plasma', s=18, alpha=0.85,
        label=f'{len(all_scores)} grasp proposals')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Confidence score', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_xlabel('X (m)', color='white');  ax.set_ylabel('Y (m)', color='white')
    ax.set_zlabel('Z (m)', color='white')
    ax.set_title('Grasp Proposal Distribution (camera frame)',
                 color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')

    best_idx = int(np.argmax(all_scores))
    ax.scatter(*all_positions[best_idx], c='red', s=120, marker='*',
               zorder=10, label=f'Best (score={all_scores[best_idx]:.4f})')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'cgn_grasp_distribution{tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  [Figure 2] Saved → {out_path}')
    return out_path


def save_top_grasps_table(pred_grasps_cam, scores, out_dir, tag=''):
    """Save a styled table figure of the top-10 grasps for the MSc report."""
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for k in pred_grasps_cam:
        g = pred_grasps_cam[k]
        s = scores[k]
        for gi, si in zip(g, s):
            pos = gi[:3, 3]
            rows.append((si, k, pos[0], pos[1], pos[2]))
    rows.sort(reverse=True)

    n = min(10, len(rows))
    if n == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')

    col_labels = ['Rank', 'Score', 'Seg ID', 'X (m)', 'Y (m)', 'Z (m)']
    cell_data  = [[str(i+1), f'{r[0]:.4f}', str(r[1]),
                   f'{r[2]:.4f}', f'{r[3]:.4f}', f'{r[4]:.4f}']
                  for i, r in enumerate(rows[:n])]

    cmap = matplotlib.colormaps['plasma']
    row_colors = [[cmap(1 - i/n)[:3] + (0.3,) for _ in range(6)]
                  for i in range(n)]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=row_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#444466')
        if r == 0:
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_text_props(color='white')

    ax.set_title('Top-10 CGN Grasp Candidates',
                 color='white', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'cgn_top_grasps_table{tag}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  [Figure 3] Saved → {out_path}')
    return out_path


# =============================================================================
#  Main pipeline
# =============================================================================

def run_visualization(args):
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f'_phi{int(args.phi)}_theta{int(args.theta)}_sd{args.sigma_d}_rho{args.rho}'

    # ── Either load saved predictions or run CGN live ──────────────────────
    if args.from_file:
        print(f'\n[Viz] Loading predictions from: {args.from_file}')
        data_np = np.load(args.from_file, allow_pickle=True)
        pc_full       = data_np['pc_full']
        pred_grasps   = data_np['pred_grasps_cam'].item()
        scores        = data_np['scores'].item()
        pc_colors     = data_np.get('pc_colors', None)
        if pc_colors is not None and len(pc_colors) == 0:
            pc_colors = None
        depth_raw = rgb = seg_map = None
        model = data = None
        print(f'  Loaded {sum(len(scores[k]) for k in scores)} grasp candidates.')

    else:
        print('\n[Viz] Live mode: rendering MuJoCo scene + running CGN...')
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        mj_data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, mj_data, 0)
        mujoco.mj_forward(model, mj_data)

        set_camera(model, args.phi, args.theta)
        mujoco.mj_forward(model, mj_data)

        print(f'  Rendering depth/seg/RGB (phi={args.phi}, theta={args.theta}, '
              f'sigma_d={args.sigma_d}, rho={args.rho})...')
        depth_noisy, K, seg_map, rgb, depth_raw = render_scene(
            model, mj_data, sigma_d=args.sigma_d, rng=rng)

        print(f'  Target pixels visible: {seg_map.sum()}')
        print('  Running CGN inference...')
        estimator = load_cgn()
        pred_grasps, scores, contact_pts, pc_full, pc_colors = run_cgn(
            depth_noisy, K, seg_map, estimator, rho=args.rho, rng=rng)
        data = mj_data

        n_total = sum(len(scores[k]) for k in scores)
        print(f'  CGN proposed {n_total} grasp candidates.')

        if args.save and n_total > 0:
            save_path = os.path.join(args.out_dir, f'cgn_predictions{tag}.npz')
            np.savez(save_path,
                     pc_full=pc_full,
                     pred_grasps_cam=pred_grasps,
                     scores=scores,
                     contact_pts=contact_pts,
                     pc_colors=pc_colors if pc_colors is not None else np.array([]))
            print(f'  [Saved] Predictions → {save_path}')

    n_total = sum(len(scores[k]) for k in scores)
    if n_total == 0:
        print('[Viz] No grasps found. Try adjusting phi/theta/sigma_d/rho.')
        return

    # ── Print top-10 summary ──────────────────────────────────────────────
    print(f'\n{"─"*60}')
    print(f'  TOP GRASP CANDIDATES  (total: {n_total})')
    print(f'{"─"*60}')
    all_rows = []
    for k in pred_grasps:
        for gi, si in zip(pred_grasps[k], scores[k]):
            all_rows.append((float(si), k, gi[:3, 3]))
    all_rows.sort(reverse=True)
    for i, (si, k, pos) in enumerate(all_rows[:10], 1):
        print(f'  #{i:2d}  score={si:.4f}  seg={k}  '
              f'pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})')
    print(f'{"─"*60}\n')

    # ── Matplotlib figures (always saved, no GUI needed) ──────────────────
    print('[Viz] Saving matplotlib figures for MSc report...')
    if depth_raw is not None and rgb is not None:
        save_matplotlib_figures(
            depth_raw, rgb, seg_map,
            pred_grasps, scores, pc_full,
            out_dir=args.out_dir,
            phi=args.phi, theta=args.theta,
            sigma_d=args.sigma_d, rho=args.rho)

    save_score_heatmap(pred_grasps, scores, pc_full, args.out_dir, tag=tag)
    save_top_grasps_table(pred_grasps, scores, args.out_dir, tag=tag)

    # ── World-frame grasps (if model is available) ────────────────────────
    pred_grasps_world = None
    if model is not None and data is not None:
        pred_grasps_world = transform_grasps_to_world(pred_grasps, model, data)
        save_score_heatmap(pred_grasps_world, scores, pc_full,
                           args.out_dir, tag=tag + '_world')
        # Rename the world-frame figure for clarity
        src = os.path.join(args.out_dir, f'cgn_grasp_distribution{tag}_world.png')
        if os.path.exists(src):
            dst = os.path.join(args.out_dir,
                               f'cgn_grasp_distribution_WORLD{tag}.png')
            os.replace(src, dst)
            print(f'  [Figure 4] World-frame → {dst}')

    # ── Open3D views ──────────────────────────────────────────────────────
    if HAS_O3D:
        n_draw = args.n_grasps   # None = all

        if args.view in ('camera', 'both'):
            sc_path = os.path.join(args.out_dir, f'cgn_o3d_camera{tag}.png')
            open3d_render_offscreen(
                pc_full, pred_grasps, scores, pc_colors,
                title='CGN Grasp Proposals - Camera Frame',
                n_grasps=n_draw,
                screenshot_path=sc_path)

        if args.view in ('world', 'both') and pred_grasps_world is not None:
            if model is not None and data is not None:
                cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
                cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
                cam_pos = data.cam_xpos[cam_id]
                flip    = np.diag([1., -1., -1.])
                pc_world = (cam_rot @ (flip @ pc_full.T)).T + cam_pos
            else:
                pc_world = pc_full

            sc_path_w = os.path.join(args.out_dir, f'cgn_o3d_world{tag}.png')
            open3d_render_offscreen(
                pc_world, pred_grasps_world, scores, pc_colors,
                title='CGN Grasp Proposals - World Frame',
                n_grasps=n_draw,
                show_axes=True,
                screenshot_path=sc_path_w)
    else:
        print('[Viz] Open3D not available – 3D views skipped (matplotlib figures saved).')

    print(f'\n[Viz] All outputs written to: {os.path.abspath(args.out_dir)}')
    print('      Add these figures to your MSc report with \\includegraphics.')


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CGN Grasp Visualisation Engine – generates report figures')
    parser.add_argument('--phi',       type=float, default=45.,
                        help='Camera elevation (deg)')
    parser.add_argument('--theta',     type=float, default=0.,
                        help='Camera azimuth (deg)')
    parser.add_argument('--sigma_d',   type=float, default=0.0,
                        help='Depth noise std-dev (m)')
    parser.add_argument('--rho',       type=float, default=1.0,
                        help='Point-cloud keep fraction')
    parser.add_argument('--n_grasps',  type=int,   default=None,
                        help='Max grasps to draw (default: all)')
    parser.add_argument('--view',      choices=['camera','world','both'],
                        default='both',
                        help='Which 3D views to open')
    parser.add_argument('--from_file', type=str,   default=None,
                        help='Load saved .npz predictions instead of running CGN')
    parser.add_argument('--save',      action='store_true',
                        help='Save raw predictions .npz for fast re-visualise')
    parser.add_argument('--out_dir',   type=str,
                        default='results/figures',
                        help='Directory for saved figures')
    parser.add_argument('--seed',      type=int,   default=42)
    args = parser.parse_args()

    run_visualization(args)
