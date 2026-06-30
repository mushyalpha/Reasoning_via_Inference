"""
grasp_simulation.py
--------------------
Stage 1 of the causal diagnosis pipeline.

This script demonstrates:
  1. Loading the MuJoCo scene (Panda arm + table + cylinder object)
  2. Positioning the virtual camera on a sphere defined by phi (elevation)
     and theta (azimuth) -- these are two of the four exogenous variables.
  3. Rendering a depth image from that camera viewpoint.
  4. Injecting Gaussian depth noise (sigma_d) -- third exogenous variable.
  5. Back-projecting the noisy depth image into a 3D point cloud.
  6. Randomly downsampling the point cloud by fraction rho -- fourth
     exogenous variable.
  7. Saving and visualising the point cloud so you can verify it looks correct.

Camera model
------------
MuJoCo uses the standard pinhole camera model, identical to what a real
RGB-D sensor (RealSense, Kinect, Azure Kinect) measures. The depth buffer
gives the distance (in metres) from the camera focal point to the nearest
surface at each pixel. Back-projection is:

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

where (fx, fy) are focal lengths derived from the camera fov and image
resolution, and (cx, cy) is the image principal point (usually image centre).

The camera here is NOT a physical object. It is a virtual viewpoint.
The green box in the viewer is just a marker so you can see where it sits.
"""

import os
import math
import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def build_camera_intrinsics(model, cam_name, img_width, img_height):
    """
    Computes the pinhole camera intrinsic matrix from MuJoCo model data.

    MuJoCo stores the field-of-view (fov) as the vertical fov in degrees.
    We derive focal lengths from this:

        fy = (img_height / 2) / tan(fov_y / 2)
        fx = fy  (MuJoCo uses square pixels)
        cx = img_width  / 2
        cy = img_height / 2

    Returns:
        fx, fy, cx, cy  -- standard intrinsic parameters
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fov_y_deg = model.cam_fovy[cam_id]
    fov_y_rad = math.radians(fov_y_deg)

    fy = (img_height / 2.0) / math.tan(fov_y_rad / 2.0)
    fx = fy   # MuJoCo renders with square pixels
    cx = img_width  / 2.0
    cy = img_height / 2.0

    return fx, fy, cx, cy


def depth_to_pointcloud(depth_image, fx, fy, cx, cy):
    """
    Back-projects a depth image (H x W float array, values in metres)
    into a 3D point cloud in the camera coordinate frame.

    Returns:
        points: (N, 3) numpy array of (X, Y, Z) coordinates in camera frame.
                Invalid pixels (depth == 0 or inf) are excluded.
    """
    H, W = depth_image.shape
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten
    u = u_coords.flatten().astype(np.float32)
    v = v_coords.flatten().astype(np.float32)
    d = depth_image.flatten()

    # Remove invalid pixels
    valid = (d > 0) & np.isfinite(d) & (d < 5.0)  # 5m max range
    u, v, d = u[valid], v[valid], d[valid]

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    points = np.stack([X, Y, Z], axis=1)
    return points


def set_camera_spherical(model, elevation_deg, azimuth_deg, radius, target_pos):
    """
    Moves the camera body to a position on a sphere of the given radius,
    centred on target_pos. The camera always looks toward target_pos.

    This is how viewpoint (phi, theta) is operationalised as a causal
    variable: same object, different camera elevation and azimuth.

    MuJoCo camera coordinate convention:
      - Camera looks along its local -Z axis
      - Local +X is right, local +Y is up

    We compute a rotation matrix from world axes and convert to a quaternion
    for the camera body.
    """
    phi   = math.radians(elevation_deg)
    theta = math.radians(azimuth_deg)

    # Camera position on sphere (Z-up world frame)
    dx = radius * math.cos(phi) * math.cos(theta)
    dy = radius * math.cos(phi) * math.sin(theta)
    dz = radius * math.sin(phi)
    cam_pos = np.array([target_pos[0] + dx,
                        target_pos[1] + dy,
                        target_pos[2] + dz])

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "perception_camera_body")
    if body_id == -1:
        raise RuntimeError("perception_camera_body not found in model")

    model.body_pos[body_id] = cam_pos

    # Build look-at rotation matrix
    forward = target_pos - cam_pos
    forward /= np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Column-major rotation: columns are right, up, -forward
    R = np.column_stack((right, up, -forward))

    # Rotation matrix → quaternion (w, x, y, z)
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    model.body_quat[body_id] = np.array([w, x, y, z])
    return cam_pos


# ---------------------------------------------------------------------------
# Depth capture + noise + downsampling
# ---------------------------------------------------------------------------

def capture_pointcloud(model, data,
                       cam_name="perception_camera",
                       img_width=640, img_height=480,
                       sigma_d=0.0, rho=1.0,
                       rng=None):
    """
    Captures a depth image from the named camera, applies noise and
    sparsity perturbations, then back-projects to a 3D point cloud.

    Parameters
    ----------
    model, data : mujoco.MjModel, mujoco.MjData
    cam_name    : str   -- name of the camera in the XML
    img_width   : int   -- render resolution width  (pixels)
    img_height  : int   -- render resolution height (pixels)
    sigma_d     : float -- std dev of additive Gaussian depth noise (metres)
                          0.0  = clean sensor
                          0.01 = 1 cm noise (good sensor)
                          0.02 = 2 cm noise (moderate degradation)
                          0.04 = 4 cm noise (severe degradation)
    rho         : float -- point cloud keep fraction after downsampling
                          1.0  = keep everything
                          0.5  = keep half the points
                          0.25 = keep quarter of the points
    rng         : np.random.Generator (optional, for reproducibility)

    Returns
    -------
    points_world : (N, 3) float array -- point cloud in world coordinates
    depth_raw    : (H, W) float array -- depth image before noise
    depth_noisy  : (H, W) float array -- depth image after noise injection
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- 1. Render depth image ---
    renderer = mujoco.Renderer(model, height=img_height, width=img_width)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_name)
    depth_raw = renderer.render().copy()   # (H, W) float32, values in metres
    renderer.disable_depth_rendering()
    renderer.close()

    # --- 2. Inject Gaussian depth noise (exogenous variable: sigma_d) ---
    if sigma_d > 0:
        noise = rng.normal(loc=0.0, scale=sigma_d, size=depth_raw.shape).astype(np.float32)
        depth_noisy = depth_raw + noise
        depth_noisy = np.clip(depth_noisy, 0, None)  # depth cannot be negative
    else:
        depth_noisy = depth_raw.copy()

    # --- 3. Compute camera intrinsics ---
    fx, fy, cx, cy = build_camera_intrinsics(model, cam_name, img_width, img_height)

    # --- 4. Back-project to point cloud in camera frame ---
    points_cam = depth_to_pointcloud(depth_noisy, fx, fy, cx, cy)

    # --- 5. Transform from camera frame to world frame ---
    # MuJoCo cam_xmat: rotation matrix from camera frame to world frame.
    # MuJoCo camera local convention:
    #   +X = right, +Y = down, +Z = toward the scene (into the image)
    # Our back-projection produces points in OpenCV convention:
    #   +X = right, +Y = down, +Z = away from camera (into scene)
    # These conventions match, so cam_xmat can be applied directly.
    mujoco.mj_forward(model, data)
    cam_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    cam_rot  = data.cam_xmat[cam_id].reshape(3, 3)   # world_R_cam
    cam_pos  = data.cam_xpos[cam_id]                  # camera origin in world

    # MuJoCo camera actually looks along -Z in world convention, so we
    # need to negate the Z column of cam_rot before applying.
    # Equivalently: flip Z sign in camera-frame points before rotating.
    points_cam_corrected = points_cam.copy()
    points_cam_corrected[:, 2] *= -1   # flip Z: cam looks along -Z in MuJoCo

    points_world = (cam_rot @ points_cam_corrected.T).T + cam_pos

    # --- 6. Downsample by rho (exogenous variable: rho) ---
    if rho < 1.0:
        n_keep = max(1, int(len(points_world) * rho))
        idx = rng.choice(len(points_world), size=n_keep, replace=False)
        points_world = points_world[idx]

    return points_world, depth_raw, depth_noisy


# ---------------------------------------------------------------------------
# Verification: print point cloud statistics
# ---------------------------------------------------------------------------

def verify_pointcloud(points, label="Point cloud"):
    """Prints basic statistics to sanity-check that the point cloud looks right."""
    if len(points) == 0:
        print(f"{label}: EMPTY — something went wrong with depth rendering.")
        return
    print(f"\n{label}")
    print(f"  Number of points : {len(points)}")
    print(f"  X range          : [{points[:,0].min():.3f}, {points[:,0].max():.3f}] m")
    print(f"  Y range          : [{points[:,1].min():.3f}, {points[:,1].max():.3f}] m")
    print(f"  Z range          : [{points[:,2].min():.3f}, {points[:,2].max():.3f}] m")
    centroid = points.mean(axis=0)
    print(f"  Centroid         : ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}) m")
    print(f"  Expected target  : near (0.500, 0.000, 0.450) m")


# ---------------------------------------------------------------------------
# Main: demonstrate the full Stage 1 pipeline
# ---------------------------------------------------------------------------

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "mujoco_menagerie",
                            "franka_emika_panda", "grasp_scene.xml")
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return

    print(f"Loading model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Step forward once so all body positions are populated
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Experimental parameters (from the 4 x 3 x 4 x 3 grid)
    # ------------------------------------------------------------------
    phi     = 45    # elevation: {30, 45, 60, 75}
    theta   = 60    # azimuth:   {0, 60, 120}
    sigma_d = 0.02  # depth noise std dev in metres: {0, 0.01, 0.02, 0.04}
    rho     = 0.5   # point cloud keep fraction: {1.0, 0.5, 0.25}
    radius  = 0.8   # fixed sphere radius (metres) — not a variable
    target_pos = np.array([0.5, 0.0, 0.45])  # centre of the graspable object

    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # Position camera
    # ------------------------------------------------------------------
    print(f"\nPositioning camera: phi={phi}deg, theta={theta}deg, radius={radius}m")
    cam_pos = set_camera_spherical(model, phi, theta, radius, target_pos)
    print(f"Camera world position: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")

    # Step forward again so camera body position propagates to cam_xpos
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Capture point cloud
    # ------------------------------------------------------------------
    print(f"\nCapturing depth image (sigma_d={sigma_d}m, rho={rho})...")
    points, depth_raw, depth_noisy = capture_pointcloud(
        model, data,
        cam_name="perception_camera",
        img_width=640, img_height=480,
        sigma_d=sigma_d,
        rho=rho,
        rng=rng
    )

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    verify_pointcloud(points, label=f"Degraded point cloud (sigma_d={sigma_d}, rho={rho})")

    # Also show what the clean point cloud looks like
    points_clean, _, _ = capture_pointcloud(
        model, data,
        cam_name="perception_camera",
        img_width=640, img_height=480,
        sigma_d=0.0,
        rho=1.0,
        rng=rng
    )
    verify_pointcloud(points_clean, label="Clean point cloud (sigma_d=0, rho=1)")

    # ------------------------------------------------------------------
    # Optional: save depth images for inspection
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(depth_raw,   cmap="plasma", vmin=0, vmax=2)
        axes[0].set_title(f"Depth (clean) | phi={phi}, theta={theta}")
        axes[0].set_xlabel("pixel u"); axes[0].set_ylabel("pixel v")
        plt.colorbar(axes[0].images[0], ax=axes[0], label="depth (m)")

        axes[1].imshow(depth_noisy, cmap="plasma", vmin=0, vmax=2)
        axes[1].set_title(f"Depth (sigma_d={sigma_d}m)")
        axes[1].set_xlabel("pixel u")
        plt.colorbar(axes[1].images[0], ax=axes[1], label="depth (m)")

        out_path = os.path.join(current_dir, "depth_check.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"\nDepth image saved to: {out_path}")
    except ImportError:
        print("\n(matplotlib not available — skipping depth image save)")

    # ------------------------------------------------------------------
    # Launch viewer so you can visually confirm camera position
    # ------------------------------------------------------------------
    print("\nLaunching viewer. Green box = camera location. Press Escape to quit.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
