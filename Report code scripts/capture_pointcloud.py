# NOTE: superseded by the inline perception pipeline in run_experiments.py
# Kept for reference only.
"""Capture depth and point cloud from the MuJoCo perception camera."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from perception import (
    DEFAULT_IMAGE_SIZE,
    capture_perception,
    get_scene_xml_path,
    load_grasp_scene,
)


def save_depth_image(depth: np.ndarray, valid_mask: np.ndarray, output_path: str) -> None:
    depth_vis = depth.copy()
    depth_vis[~valid_mask] = np.nan
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_vis, cmap="viridis")
    plt.colorbar(label="Depth (m)")
    plt.title("Perception camera depth map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_pointcloud_plot(
    points: np.ndarray,
    target_pos: np.ndarray,
    output_path: str,
    colors: np.ndarray | None = None,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1, alpha=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.8)

    ax.scatter(
        target_pos[0],
        target_pos[1],
        target_pos[2],
        c="red",
        s=80,
        marker="x",
        label="Target object centre",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Back-projected point cloud")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi", type=float, default=45.0, help="Camera elevation (deg)")
    parser.add_argument("--theta", type=float, default=60.0, help="Camera azimuth (deg)")
    parser.add_argument(
        "--sigma-d",
        type=float,
        default=0.0,
        help="Gaussian depth noise standard deviation (m)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Point cloud retention fraction in (0, 1]",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for saved depth image and point cloud plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, data = load_grasp_scene(get_scene_xml_path())
    height, width = DEFAULT_IMAGE_SIZE
    renderer = mujoco.Renderer(model, height=height, width=width)
    rng = np.random.default_rng(args.seed)

    capture = capture_perception(
        model,
        data,
        renderer,
        elevation_deg=args.phi,
        azimuth_deg=args.theta,
        sigma_d=args.sigma_d,
        rho=args.rho,
        rng=rng,
    )
    renderer.close()

    target_pos = np.array([0.5, 0.0, 0.45])
    depth_path = os.path.join(args.output_dir, "depth_map.png")
    cloud_path = os.path.join(args.output_dir, "pointcloud.png")
    points_path = os.path.join(args.output_dir, "pointcloud.npy")

    save_depth_image(capture.depth, capture.valid_mask, depth_path)
    save_pointcloud_plot(capture.points, target_pos, cloud_path, capture.colors)
    np.save(points_path, capture.points)

    print(f"Captured {capture.points.shape[0]} points")
    print(f"Depth map saved to: {depth_path}")
    print(f"Point cloud plot saved to: {cloud_path}")
    print(f"Raw points saved to: {points_path}")
    print(f"Camera position: {capture.camera_pos}")
    print(f"Depth range (valid pixels): "
          f"{capture.depth[capture.valid_mask].min():.3f} - "
          f"{capture.depth[capture.valid_mask].max():.3f} m")


if __name__ == "__main__":
    main()
