"""
smoke_test_floating_gripper.py
================================
Standalone sanity check for the new floating-gripper shake test
(sim_common.run_floating_gripper_test), independent of CGN/perception.

For each object below, runs three hand-constructed poses:
  1. A KNOWN GOOD grasp pose (fingers straddling the object at a side
     approach) should be collision_free=True and success=True.
  2. A KNOWN BAD grasp pose (30cm off to the side, nothing between the
     fingers) should be collision_free=True (nothing there to collide
     with) but success=False (object never gets lifted / stays out of
     tolerance during the shake).
  3. A KNOWN COLLIDING pose (gripper placed so it intersects the object
     with fingers still open -- e.g. positioned so the object is INSIDE
     one of the finger volumes rather than between them) should be
     collision_free=False.

Objects covered: cylinder (original mechanism validation) and box (the
YcbSugarBox replacement for the pathologically thin YcbGelatinBox -- see
RIGOUR_LEDGER.md Stage 21/22). Mustard bottle is not hand-pose-tested here
since its irregular geometry makes a hand-tuned pose unrepresentative of
what CGN would propose; it is instead validated via genuine CGN proposals
in the end-to-end pipeline test (see render_floating_gripper_figures.py).
"""
import os
import numpy as np
import mujoco

import sim_common as sc
from object_specs import OBJECT_SPECS, spawn_pos, centroid_world, build_scene_xml, FLOATING_GRIPPER_TEMPLATE

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR = os.path.join(ROOT, 'generated_scenes')
os.makedirs(SCENES_DIR, exist_ok=True)


def cylinder_side_quat():
    """Hand approaches horizontally (local z = world -x), fingers close
    along world y (a horizontal diameter of the cylinder's circular
    cross-section). Used instead of a top-down grasp: for a cylinder of
    uniform diameter, a top-down approach only has a few mm of clearance
    between the fully-open fingers and the object (open gap ~0.083m vs.
    diameter 0.072m), which is fragile to reproduce reliably; gripping
    the side at a modest standoff is the geometry validated during
    development of run_floating_gripper_test (see RIGOUR_LEDGER)."""
    approach = np.array([-1., 0., 0.])
    base = np.array([0., 0., 1.])
    y = np.cross(approach, base)
    R = np.column_stack((base, y, approach))
    return sc.rot_matrix_to_quat(R)


def box_side_quat():
    """Hand approaches horizontally (local z = world -y), fingers close
    along world x -- the sugar box's ~4.95cm thin axis (its bounding-box
    footprint is ~4.95 x 9.42cm; the gripper can only span the shorter of
    the two, so this is the only feasible closing axis for a side grasp).
    'Up' (world z) matches the box's long (17.6cm) vertical axis."""
    approach = np.array([0., -1., 0.])
    base = np.array([0., 0., 1.])
    y = np.cross(approach, base)
    R = np.column_stack((base, y, approach))
    return sc.rot_matrix_to_quat(R)


# Per-object: (grasp-quat function, standoff axis+distance for the
# good/bad poses, squeeze margin). Standoff is added to the centroid to
# get the good pose (and a larger multiple for the empty/bad pose).
OBJECTS = {
    'cylinder': dict(quat_fn=cylinder_side_quat, standoff_axis=np.array([1., 0., 0.]),
                      standoff_good=0.12, standoff_bad=0.30),
    'box':      dict(quat_fn=box_side_quat,      standoff_axis=np.array([0., 1., 0.]),
                      standoff_good=0.12, standoff_bad=0.30),
}

all_pass = True

for obj_name, cfg in OBJECTS.items():
    print(f'\n{"=" * 60}\nOBJECT: {obj_name}\n{"=" * 60}')
    spec = OBJECT_SPECS[obj_name]

    fg_xml = os.path.join(SCENES_DIR, f'scene_{obj_name}_floating_gripper.xml')
    build_scene_xml(obj_name, fg_xml, template_path=FLOATING_GRIPPER_TEMPLATE)
    print(f'Built floating-gripper scene: {fg_xml}')

    model = mujoco.MjModel.from_xml_path(fg_xml)
    print(f'Loaded model OK. nbody={model.nbody} nmocap={model.nmocap} njnt={model.njnt}')

    obj_pos = np.array(spawn_pos(spec))
    obj_quat = np.array([1., 0., 0., 0.])
    centroid = np.array(centroid_world(spec, obj_pos))
    print(f'Object spawn pos={obj_pos}  centroid={centroid}  footprint_radius={spec["footprint_radius"]}')

    def fresh_data():
        data = mujoco.MjData(model)
        sc.set_object_pose(model, data, spec['body_name'], obj_pos, obj_quat)
        mujoco.mj_forward(model, data)
        sc.settle(model, data, 30)
        return data

    grasp_quat_good = cfg['quat_fn']()

    print('\n--- TEST 1: known-good side-grasp pose ---')
    grasp_pos_good = centroid + cfg['standoff_axis'] * cfg['standoff_good']
    data = fresh_data()
    result = sc.run_floating_gripper_test(
        model, data, spec['body_name'], grasp_pos_good, grasp_quat_good,
        footprint_radius=spec['footprint_radius'], squeeze_margin=60.)
    print(result)
    try:
        assert result['collision_free'], 'Expected collision-free at the good pose'
        assert result['success'], 'Expected the good pose to succeed the shake test'
        print('PASS')
    except AssertionError as e:
        print(f'FAIL: {e}')
        all_pass = False

    print('\n--- TEST 2: known-bad grasp pose (30cm to the side, nothing there) ---')
    grasp_pos_bad = centroid + cfg['standoff_axis'] * cfg['standoff_bad']
    data = fresh_data()
    result = sc.run_floating_gripper_test(
        model, data, spec['body_name'], grasp_pos_bad, grasp_quat_good,
        footprint_radius=spec['footprint_radius'], squeeze_margin=60.)
    print(result)
    try:
        assert result['collision_free'], 'Expected collision-free (nothing near the gripper)'
        assert not result['success'], 'Expected the bad (empty) pose to fail the shake test'
        print('PASS')
    except AssertionError as e:
        print(f'FAIL: {e}')
        all_pass = False

    print('\n--- TEST 3: known-colliding pose (gripper base pushed through the object) ---')
    grasp_pos_collide = centroid.copy()  # hand origin planted at the object's centroid
    data = fresh_data()
    result = sc.run_floating_gripper_test(
        model, data, spec['body_name'], grasp_pos_collide, grasp_quat_good,
        footprint_radius=spec['footprint_radius'], squeeze_margin=60.)
    print(result)
    try:
        assert not result['collision_free'], 'Expected a collision at this pose'
        print('PASS')
    except AssertionError as e:
        print(f'FAIL: {e}')
        all_pass = False

print(f'\n{"=" * 60}')
if all_pass:
    print('=== All floating-gripper smoke tests passed (cylinder + box) ===')
else:
    print('=== SOME SMOKE TESTS FAILED -- see above ===')
    raise SystemExit(1)
