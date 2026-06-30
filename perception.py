"""Depth capture and point cloud generation from the MuJoCo perception camera."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np

from grasp_simulation import set_camera_position

CAMERA_NAME = "perception_camera"
TARGET_OBJECT_NAME = "target_object"
DEFAULT_IMAGE_SIZE = (480, 640)  # (height, width)
DEFAULT_CAMERA_RADIUS = 0.8
DEFAULT_TARGET_POS = np.array([0.5, 0.0, 0.45])


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class PerceptionCapture:
    depth: np.ndarray
    rgb: np.ndarray | None
    points: np.ndarray
    colors: np.ndarray | None
    valid_mask: np.ndarray
    intrinsics: CameraIntrinsics
    camera_pos: np.ndarray
    camera_rot: np.ndarray


def get_scene_xml_path() -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        repo_root, "mujoco_menagerie", "franka_emika_panda", "grasp_scene.xml"
    )


def load_grasp_scene(xml_path: str | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    path = xml_path or get_scene_xml_path()
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


def get_target_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_OBJECT_NAME)
    if body_id == -1:
        return DEFAULT_TARGET_POS.copy()
    return data.xpos[body_id].copy()


def configure_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    elevation_deg: float,
    azimuth_deg: float,
    radius: float = DEFAULT_CAMERA_RADIUS,
    target_pos: np.ndarray | None = None,
) -> None:
    target = DEFAULT_TARGET_POS if target_pos is None else target_pos
    set_camera_position(
        model,
        data,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        radius=radius,
        target_pos=target,
    )
    mujoco.mj_forward(model, data)


def get_camera_intrinsics(
    model: mujoco.MjModel,
    camera_name: str = CAMERA_NAME,
    height: int = DEFAULT_IMAGE_SIZE[0],
    width: int = DEFAULT_IMAGE_SIZE[1],
) -> CameraIntrinsics:
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found in model.")

    fovy_rad = math.radians(model.cam_fovy[camera_id])
    fx = width / (2.0 * math.tan(fovy_rad / 2.0))
    fy = height / (2.0 * math.tan(fovy_rad / 2.0))
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
    )


def get_depth_truncation(model: mujoco.MjModel) -> float:
    """Return the maximum valid depth in metres for this scene."""
    return float(model.vis.map.zfar * model.stat.extent * 0.999)


def inject_depth_noise(
    depth: np.ndarray,
    sigma_d: float,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian depth noise to valid pixels only."""
    if sigma_d <= 0.0:
        return depth.copy()

    noisy = depth.copy()
    noise = rng.normal(loc=0.0, scale=sigma_d, size=depth.shape)
    noisy[valid_mask] = np.maximum(noisy[valid_mask] + noise[valid_mask], 1e-4)
    return noisy


def downsample_points(
    points: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Randomly keep a fraction rho of the point cloud."""
    if points.size == 0:
        return points, colors
    if rho >= 1.0:
        return points, colors

    keep_count = max(1, int(round(len(points) * rho)))
    indices = rng.choice(len(points), size=keep_count, replace=False)
    sampled_colors = None if colors is None else colors[indices]
    return points[indices], sampled_colors


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    camera_pos: np.ndarray,
    camera_rot: np.ndarray,
    depth_trunc: float,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Back-project a MuJoCo depth image into a world-frame point cloud.

    MuJoCo depth values are metric distances after undoing the OpenGL projection.
    We treat each pixel as pinhole projection with z equal to the rendered depth.
    """
    valid = (depth > 0.0) & (depth < depth_trunc)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), None, valid

    u_grid, v_grid = np.meshgrid(
        np.arange(intrinsics.width, dtype=np.float64),
        np.arange(intrinsics.height, dtype=np.float64),
        sparse=True,
    )

    z = depth.astype(np.float64)
    x_cam = z * (u_grid - intrinsics.cx) / intrinsics.fx
    y_cam = z * (v_grid - intrinsics.cy) / intrinsics.fy

    points_cam = np.stack(
        (
            x_cam[valid],
            -y_cam[valid],
            -z[valid],
        ),
        axis=-1,
    )

    # cam_xmat rows are camera axes expressed in world coordinates.
    points_world = points_cam @ camera_rot + camera_pos

    colors = None
    if rgb is not None:
        colors = rgb[valid].astype(np.float64) / 255.0

    return points_world, colors, valid


def capture_perception(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    elevation_deg: float,
    azimuth_deg: float,
    sigma_d: float = 0.0,
    rho: float = 1.0,
    radius: float = DEFAULT_CAMERA_RADIUS,
    target_pos: np.ndarray | None = None,
    camera_name: str = CAMERA_NAME,
    rng: np.random.Generator | None = None,
    capture_rgb: bool = True,
) -> PerceptionCapture:
    """Render depth (and optionally RGB), apply noise/sparsity, and back-project."""
    rng = np.random.default_rng() if rng is None else rng

    configure_camera(
        model,
        data,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        radius=radius,
        target_pos=target_pos,
    )

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    intrinsics = get_camera_intrinsics(
        model,
        camera_name=camera_name,
        height=renderer.height,
        width=renderer.width,
    )
    depth_trunc = get_depth_truncation(model)

    renderer.update_scene(data, camera=camera_name)
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()

    rgb = None
    if capture_rgb:
        rgb = renderer.render().copy()

    valid_mask = (depth > 0.0) & (depth < depth_trunc)
    depth_noisy = inject_depth_noise(depth, sigma_d, valid_mask, rng)

    points, colors, _ = depth_to_pointcloud(
        depth_noisy,
        intrinsics,
        data.cam_xpos[camera_id].copy(),
        data.cam_xmat[camera_id].reshape(3, 3).copy(),
        depth_trunc=depth_trunc,
        rgb=rgb,
    )
    points, colors = downsample_points(points, rho, rng, colors)

    return PerceptionCapture(
        depth=depth_noisy,
        rgb=rgb,
        points=points,
        colors=colors,
        valid_mask=valid_mask,
        intrinsics=intrinsics,
        camera_pos=data.cam_xpos[camera_id].copy(),
        camera_rot=data.cam_xmat[camera_id].reshape(3, 3).copy(),
    )
