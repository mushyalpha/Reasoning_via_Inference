"""
object_specs.py
================
Object geometry specifications for the multi-object grasp experiments
(response to preliminary marking feedback: "shape matters" / geometry as
a moderator, Marker A & B).

Three objects, spanning the shape categories markers asked for:
  cylinder  - rotationally symmetric curved surface (original object, kept
              for continuity with the existing 432-trial dataset)
  box       - flat faces + edges (YCB SugarBox mesh; see note below on the
              earlier GelatinBox attempt)
  mustard   - irregular / asymmetric curved surface, shifting cross-section
              (YCB MustardBottle mesh)

Meshes are real YCB geometry (visual .obj). The cylinder-category-2 objects
(mustard, and the original gelatin box) came via the pre-processed mirror at
github.com/eleramp/pybullet-object-models; the sugar box comes directly from
the official YCB Benchmarks mesh release (google_16k scan) since that mirror
does not carry 004_sugar_box (see assets/ycb/README.md for attribution and
exact sources). Collision for the two non-cylinder objects is a simple
axis-aligned box primitive fit to the mesh's own bounding box, not a mesh --
see the box-category note below and the mustard bottle's own docstring for
why. This is a deliberate simplification: the outcome criterion is now the
physically-grounded floating-gripper shake test (RIGOUR_LEDGER.md Stage 21),
not a proximity threshold, but exact concave-mesh collision fidelity is
still not required for that test to be valid -- a box primitive is exact
for a box-shaped object, and conservative (slightly larger than the true
concave hull) for the mustard bottle. Only the *visual/depth* mesh (which
keeps full YCB geometry) matters for what CGN perceives. Logged in
RIGOUR_LEDGER.md.

Box category note: the original choice, YcbGelatinBox, turned out to be a
pathological case for a parallel-jaw gripper -- at ~2.8cm thick it is thin
enough that the fingertip pads struggle to seat flush against its faces
before contact-closing, producing a high failure rate that was hard to
distinguish from a genuine "thin objects are hard" finding vs. a smoke-test
artifact (see the implementation write-up in RIGOUR_LEDGER.md Stage 21).
It was replaced with YcbSugarBox: still squarely in the box/cuboid category
(flat faces, sharp edges) that Marker A asked for, but ~4.95cm across its
thinnest axis (measured from the mesh's own bounding box, not the nominal
factory dimensions) -- comfortably thicker than the gelatin box, still well
under the Panda gripper's 8cm max opening, and, being an almost-perfect
rectangular prism, exactly suited to a clean `type="box"` collision
primitive (the collision.obj V-HACD mesh is dropped entirely for this
object, not just approximated).

Physical parameters (mass, friction) are kept in the same light / high-
friction regime as the original cylinder (not the real YCB inertial
values) for consistency with the already-validated single-object pipeline
-- see the cylinder object's own docstring precedent in grasp_scene_v2.xml.

Table top surface is at world z = 0.4 (table body pos.z=0.2 + half-size 0.2).
Each object's `bottom_local_z` places it resting exactly on the table top,
matching the original cylinder's convention (body_z = 0.4 + half_height).
"""

import os
import math

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_TOP_Z = 0.4


def _quat_from_yaw(deg):
    r = math.radians(deg) / 2.0
    return (math.cos(r), 0.0, 0.0, math.sin(r))


OBJECT_SPECS = {
    # ------------------------------------------------------------------
    # Cylinder: original object, primitive geometry, unchanged from
    # grasp_scene_v2.xml so the existing 432-trial dataset remains a
    # valid comparison point.
    # ------------------------------------------------------------------
    'cylinder': dict(
        label='Cylinder (curved, rotationally symmetric)',
        kind='primitive',
        body_name='target_object',
        geom_name='cup',
        primitive_type='cylinder',
        primitive_size=(0.036, 0.055),
        bottom_local_z=-0.055,
        centroid_local_offset=(0.0, 0.0, 0.0),
        footprint_radius=0.036,
        half_height=0.055,
        mass=0.05,
        friction=(2.5, 0.01, 0.0001),
        color=(0.1, 0.5, 0.8, 1.0),
        spawn_xy=(0.5, 0.0),
        spawn_quat=(1.0, 0.0, 0.0, 0.0),
        inertial_mass=0.05,
        inertial_diaginertia=(0.0002, 0.0002, 0.0001),
    ),

    # ------------------------------------------------------------------
    # Box: YCB SugarBox mesh. Flat faces + edges (Marker A category 1).
    # Replaces the original YcbGelatinBox (see module docstring): the
    # gelatin box's ~2.8cm thickness was a pathological case for the
    # parallel-jaw gripper. All geometry fields below were measured
    # directly from the mesh's own vertex bounding box (google_16k scan;
    # object rests with its tall axis vertical, footprint ~4.95x9.42cm,
    # height ~17.6cm), not from nominal YCB factory dimensions, following
    # the same empirical convention used for the mustard bottle. The
    # mesh's own origin already sits at its base (min z ~ 0), unlike the
    # pybullet-mirror meshes (gelatin/mustard) whose origin sits near the
    # centroid -- hence bottom_local_z ~ 0 here.
    # Collision: a `type="box"` primitive fit to the same bounding box --
    # this object is an almost-perfect rectangular prism, so (unlike the
    # mustard bottle, where the box primitive is a conservative proxy for
    # a genuinely irregular shape) the primitive is essentially exact.
    # No collision.obj/V-HACD mesh is used at all for this object.
    # ------------------------------------------------------------------
    'box': dict(
        label='Box (flat faces, edges) - YCB SugarBox',
        kind='mesh',
        body_name='target_object',
        geom_name='sugarbox_target',
        visual_mesh='ycb_sugarbox_visual',
        visual_mesh_file=os.path.join('assets', 'ycb', 'sugar_box', 'visual.obj'),
        collision_kind='box',
        collision_half_size=(0.024748, 0.047081, 0.088008),
        bottom_local_z=0.0,
        centroid_local_offset=(-0.007467, -0.016712, 0.088039),
        footprint_radius=0.024748,   # half of the 4.95cm graspable (thin) axis
        half_height=0.088008,
        mass=0.06,
        friction=(2.0, 0.01, 0.0001),
        color=(0.8, 0.72, 0.2, 1.0),
        spawn_xy=(0.5, 0.0),
        spawn_quat=(1.0, 0.0, 0.0, 0.0),
        inertial_mass=0.06,
        inertial_diaginertia=(0.0002, 0.000167, 0.0000566),
    ),

    # ------------------------------------------------------------------
    # Mustard bottle: YCB MustardBottle mesh. Irregular / asymmetric
    # curved surface, off-centre mass, tapered shoulders (Marker A
    # category 4 / user's chosen "asymmetric" object).
    # ------------------------------------------------------------------
    'mustard': dict(
        label='Mustard bottle (irregular, asymmetric) - YCB MustardBottle',
        kind='mesh',
        body_name='target_object',
        geom_name='mustard_target',
        visual_mesh='ycb_mustard_visual',
        visual_mesh_file=os.path.join('assets', 'ycb', 'mustard_bottle', 'visual.obj'),
        # Collision: the merged V-HACD convex hull (4 decomposed pieces ->
        # 1 MuJoCo mesh geom) does not preserve a flat, stable base -- the
        # bottle tips and slides on settling (tested: ~22cm drift over 500
        # settle steps). Use a bounding-box primitive for collision instead
        # (mirrors the visual-mesh/collision-primitive split already used
        # for the Panda arm links). The real mesh remains the *visual*
        # (depth-rendered) geometry -- what matters for the perception
        # causal story -- while physics gets a stable flat-bottomed proxy.
        collision_kind='box',
        collision_half_size=(0.0482, 0.0291, 0.0958),
        bottom_local_z=-0.083973,
        centroid_local_offset=(0.0053, -0.0076, 0.0118),
        footprint_radius=0.029,   # ~ half of 5.8cm narrow (graspable) axis
        half_height=0.0955,
        mass=0.08,
        friction=(2.0, 0.01, 0.0001),
        color=(0.85, 0.75, 0.1, 1.0),
        spawn_xy=(0.5, 0.0),
        spawn_quat=(1.0, 0.0, 0.0, 0.0),
        inertial_mass=0.08,
        inertial_diaginertia=(0.001, 0.001, 0.0003),
    ),
}

OBJECT_NAMES = list(OBJECT_SPECS.keys())


def spawn_pos(spec):
    """World (x, y, z) resting position for a single-object scene."""
    x, y = spec['spawn_xy']
    z = TABLE_TOP_Z - spec['bottom_local_z']
    return (x, y, z)


def centroid_world(spec, body_xpos):
    """World-frame centroid given the body's xpos (identity-orientation
    assumption -- all objects are spawned with spawn_quat=(1,0,0,0), so
    the local centroid offset can be added directly without rotation)."""
    ox, oy, oz = spec['centroid_local_offset']
    return (body_xpos[0] + ox, body_xpos[1] + oy, body_xpos[2] + oz)


def _asset_xml_mesh(spec):
    parts = []
    if spec['kind'] == 'mesh':
        parts.append(f'    <mesh name="{spec["visual_mesh"]}" '
                      f'file="{os.path.join(_PROJECT_DIR, spec["visual_mesh_file"])}"/>')
        if spec.get('collision_kind') != 'box':
            parts.append(f'    <mesh name="{spec["collision_mesh"]}" '
                          f'file="{os.path.join(_PROJECT_DIR, spec["collision_mesh_file"])}"/>')
    return '\n'.join(parts)


def _body_xml(spec, body_name_override=None, xy_override=None, suffix_geom_names=''):
    """Build the <body>...</body> XML fragment for one object instance."""
    body_name = body_name_override or spec['body_name']
    x, y = xy_override if xy_override is not None else spec['spawn_xy']
    z = TABLE_TOP_Z - spec['bottom_local_z']
    qw, qx, qy, qz = spec['spawn_quat']
    im = spec['inertial_mass']
    dix, diy, diz = spec['inertial_diaginertia']
    fr = ' '.join(str(v) for v in spec['friction'])
    r, g, b, a = spec['color']

    if spec['kind'] == 'primitive':
        pt = spec['primitive_type']
        ps = ' '.join(str(v) for v in spec['primitive_size'])
        geom = (f'      <geom name="{spec["geom_name"]}{suffix_geom_names}" type="{pt}" size="{ps}" '
                f'rgba="{r} {g} {b} {a}" mass="{spec["mass"]}" condim="4" friction="{fr}"/>')
    else:
        vis_geom = (
            f'      <geom name="{spec["geom_name"]}{suffix_geom_names}_vis" type="mesh" '
            f'mesh="{spec["visual_mesh"]}" rgba="{r} {g} {b} {a}" '
            f'contype="0" conaffinity="0" group="2"/>'
        )
        if spec.get('collision_kind') == 'box':
            cx, cy, cz = spec['centroid_local_offset']
            hx, hy, hz = spec['collision_half_size']
            col_geom = (
                f'      <geom name="{spec["geom_name"]}{suffix_geom_names}_col" type="box" '
                f'pos="{cx} {cy} {cz}" size="{hx} {hy} {hz}" mass="{spec["mass"]}" '
                f'condim="4" friction="{fr}" group="3" rgba="{r} {g} {b} 0"/>'
            )
        else:
            col_geom = (
                f'      <geom name="{spec["geom_name"]}{suffix_geom_names}_col" type="mesh" '
                f'mesh="{spec["collision_mesh"]}" mass="{spec["mass"]}" '
                f'condim="4" friction="{fr}" group="3" rgba="{r} {g} {b} 0"/>'
            )
        geom = vis_geom + '\n' + col_geom

    return (
        f'    <body name="{body_name}" pos="{x} {y} {z}">\n'
        f'      <freejoint/>\n'
        f'      <inertial pos="0 0 0" mass="{im}" diaginertia="{dix} {diy} {diz}"/>\n'
        f'{geom}\n'
        f'    </body>'
    )


# ══════════════════════════════════════════════════════════════════════
# Experiment B (clutter): all 3 objects together in a fixed triangular
# arrangement, ~0.11 m from the cluster centre -- close enough to create
# genuine visual occlusion / boundary ambiguity between objects (Marker
# B's causal story) without objects overlapping at spawn (largest
# footprint radius among the three is ~0.048 m, well under half the
# 0.11 m spacing).
# ══════════════════════════════════════════════════════════════════════
CLUTTER_CENTER_XY = (0.5, 0.0)
CLUTTER_RADIUS = 0.11
CLUTTER_BODY_NAMES = {'cylinder': 'obj_cylinder', 'box': 'obj_box', 'mustard': 'obj_mustard'}


def clutter_layout():
    """Returns the (spec_key, body_name, (x, y)) list for build_scene_xml,
    and a dict {spec_key: (x, y)} for convenience."""
    angles_deg = {'cylinder': 90., 'box': 210., 'mustard': 330.}
    cx, cy = CLUTTER_CENTER_XY
    xy = {}
    for key, ang in angles_deg.items():
        a = math.radians(ang)
        xy[key] = (cx + CLUTTER_RADIUS * math.cos(a), cy + CLUTTER_RADIUS * math.sin(a))
    objects = [(key, CLUTTER_BODY_NAMES[key], xy[key]) for key in ['cylinder', 'box', 'mustard']]
    return objects, xy


def clutter_spawn_positions():
    """dict {body_name: (x, y, z)} resting position for each object in
    the fixed clutter arrangement."""
    _, xy = clutter_layout()
    out = {}
    for key, body_name in CLUTTER_BODY_NAMES.items():
        spec = OBJECT_SPECS[key]
        x, y = xy[key]
        z = TABLE_TOP_Z - spec['bottom_local_z']
        out[body_name] = (x, y, z)
    return out


FLOATING_GRIPPER_TEMPLATE = os.path.join(_PROJECT_DIR, 'floating_gripper_template.xml')


def build_scene_xml(objects, out_path, template_path=None):
    """
    Build a scene XML for one or more objects and write it to out_path.

    Parameters
    ----------
    objects : list of (spec_key, body_name, (x, y)) tuples for clutter,
              or a single spec_key string for the isolated-object case
              (uses the default body name 'target_object' and spawn_xy).
    out_path : destination .xml path
    template_path : defaults to grasp_scene_template.xml. Pass
              FLOATING_GRIPPER_TEMPLATE to build the arm-free floating-
              gripper scene instead (same OBJECT_ASSETS/OBJECT_BODIES
              placeholders, so this function works unchanged for both --
              see run_experiments_v2.execute_grasp / sim_common.run_floating_gripper_test).

    Returns
    -------
    out_path
    """
    if template_path is None:
        template_path = os.path.join(_PROJECT_DIR, 'grasp_scene_template.xml')

    if isinstance(objects, str):
        spec = OBJECT_SPECS[objects]
        asset_xml = _asset_xml_mesh(spec)
        body_xml = _body_xml(spec)
    else:
        asset_chunks, body_chunks = [], []
        seen_meshes = set()
        for key, body_name, xy in objects:
            spec = OBJECT_SPECS[key]
            if spec['kind'] == 'mesh' and spec['visual_mesh'] not in seen_meshes:
                asset_chunks.append(_asset_xml_mesh(spec))
                seen_meshes.add(spec['visual_mesh'])
            body_chunks.append(_body_xml(spec, body_name_override=body_name, xy_override=xy))
        asset_xml = '\n'.join(asset_chunks)
        body_xml = '\n'.join(body_chunks)

    with open(template_path) as f:
        template = f.read()

    meshdir = os.path.join(_PROJECT_DIR, 'mujoco_menagerie', 'franka_emika_panda', 'assets')
    xml = template.replace('__MESHDIR__', meshdir)
    xml = xml.replace('<!--OBJECT_ASSETS-->', asset_xml)
    xml = xml.replace('<!--OBJECT_BODIES-->', body_xml)

    with open(out_path, 'w') as f:
        f.write(xml)
    return out_path
