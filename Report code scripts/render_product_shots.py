"""
Render the three confirmatory grasp objects as isolated product shots
on a pure white background (supermarket-brochure style).

Uses the same meshes / primitive as the experiment:
  cylinder  — MuJoCo primitive, radius 36 mm, height 110 mm, experiment blue
  sugar box — YCB 004_sugar_box visual.obj + official texture
  mustard   — YCB 006_mustard_bottle visual.obj + official texture
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "product_shots"
OUT.mkdir(parents=True, exist_ok=True)

W = H = 1600


def _xml_textured(name: str, mesh_path: Path, tex_path: Path) -> str:
    return f"""
<mujoco model="{name}_product">
  <compiler angle="radian" meshdir="{mesh_path.parent}" texturedir="{tex_path.parent}"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.65 0.65 0.65" specular="0.12 0.12 0.12"/>
    <rgba haze="0 0 0 0"/>
    <global offwidth="{W}" offheight="{H}" fovy="24"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
    <texture name="tex" type="2d" file="{tex_path.name}"/>
    <material name="mat" texture="tex" specular="0.18" shininess="0.12" reflectance="0"/>
    <mesh name="obj" file="{mesh_path.name}"/>
  </asset>
  <worldbody>
    <light pos="0.6 0.5 1.4" dir="-0.25 -0.2 -1" diffuse="0.45 0.45 0.45"
           specular="0.18 0.18 0.18" directional="true"/>
    <light pos="-0.5 -0.6 1.0" dir="0.25 0.3 -1" diffuse="0.28 0.28 0.28"
           directional="true"/>
    <body name="obj" pos="0 0 0">
      <geom name="obj" type="mesh" mesh="obj" material="mat"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _xml_cylinder() -> str:
    # Same primitive as object_specs.py: radius 0.036, half-height 0.055, rgba 0.1 0.5 0.8
    return f"""
<mujoco model="cylinder_product">
  <compiler angle="radian"/>
  <visual>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <rgba haze="0 0 0 0"/>
    <global offwidth="{W}" offheight="{H}" fovy="24"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="16" height="16"/>
    <material name="cyl" rgba="0.10 0.50 0.80 1" specular="0.35" shininess="0.35"/>
  </asset>
  <worldbody>
    <light pos="0.6 0.5 1.4" dir="-0.25 -0.2 -1" diffuse="0.5 0.5 0.5"
           specular="0.25 0.25 0.25" directional="true"/>
    <light pos="-0.5 -0.6 1.0" dir="0.25 0.3 -1" diffuse="0.3 0.3 0.3"
           directional="true"/>
    <body name="obj" pos="0 0 0.055">
      <geom name="obj" type="cylinder" size="0.036 0.055" material="cyl"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _mesh_center_extent(model: mujoco.MjModel):
    adr = int(model.mesh_vertadr[0])
    n = int(model.mesh_vertnum[0])
    verts = np.array(model.mesh_vert[adr : adr + n])
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return 0.5 * (lo + hi), float((hi - lo).max())


def _geom_center_extent(model: mujoco.MjModel, data: mujoco.MjData):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
    center = np.array(data.geom_xpos[gid])
    r = float(model.geom_rbound[gid])
    return center, 2.0 * r


def render_object(xml: str, azimuth: float, elevation: float, distance_scale: float):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if model.nmesh > 0:
        center, extent = _mesh_center_extent(model)
    else:
        center, extent = _geom_center_extent(model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = extent * distance_scale
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)

    with mujoco.Renderer(model, height=H, width=W) as renderer:
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render().copy()

    return rgb, depth


def isolate_on_white(rgb: np.ndarray, depth: np.ndarray, pad_frac: float = 0.16) -> Image.Image:
    finite = np.isfinite(depth)
    positive = depth > 1e-5
    # MuJoCo background depth is typically 0; object pixels are in front of the far plane.
    if finite.any():
        dmax = np.nanmax(depth)
        mask = positive & (depth < 0.99 * dmax if dmax > 0 else positive)
        # If almost everything is "object", fall back to any positive depth.
        if mask.mean() > 0.95:
            mask = positive
    else:
        mask = np.any(rgb < 250, axis=2)

    if mask.sum() < 50:
        mask = np.any(rgb < 250, axis=2)

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Composite object onto pure white using the depth mask (keeps white labels).
    canvas = np.full_like(rgb, 255)
    canvas[mask] = rgb[mask]

    crop = canvas[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = crop.shape[:2]
    side = int(max(ch, cw) * (1.0 + 2 * pad_frac))
    side = max(side, 64)
    out = np.full((side, side, 3), 255, dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    out[y_off : y_off + ch, x_off : x_off + cw] = crop
    return Image.fromarray(out)


def save(img: Image.Image, name: str) -> Path:
    path = OUT / name
    img.save(path, format="PNG")
    print(f"wrote {path}  ({img.size[0]}x{img.size[1]})")
    return path


def main():
    mustard_xml = _xml_textured(
        "mustard",
        ROOT / "assets" / "ycb" / "mustard_bottle" / "visual.obj",
        ROOT / "assets" / "ycb" / "mustard_bottle" / "texture_map.png",
    )
    sugar_xml = _xml_textured(
        "sugar_box",
        ROOT / "assets" / "ycb" / "sugar_box" / "visual.obj",
        ROOT / "assets" / "ycb" / "sugar_box" / "texture_map.png",
    )

    shots = [
        ("cylinder", _xml_cylinder(), 140.0, -18.0, 2.55),
        ("sugar_box", sugar_xml, 125.0, -16.0, 2.35),
        ("mustard", mustard_xml, 240.0, -16.0, 2.45),
    ]

    for name, xml, az, el, dist in shots:
        print(f"rendering {name} ...")
        rgb, depth = render_object(xml, az, el, dist)
        img = isolate_on_white(rgb, depth)
        save(img, f"{name}_white.png")

    print("done.")


if __name__ == "__main__":
    main()
