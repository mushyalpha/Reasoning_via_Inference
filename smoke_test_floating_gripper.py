"""
smoke_test_floating_gripper.py
================================
Standalone sanity check for the new floating-gripper shake test
(sim_common.run_floating_gripper_test), independent of CGN/perception.

Tests, on the cylinder object:
  1. A KNOWN GOOD grasp pose (directly above the object, fingers aligned
     to straddle it, oriented so approach = -z i.e. top-down) should be
     collision_free=True and success=True.
  2. A KNOWN BAD grasp pose (30cm off to the side, nothing between the
     fingers) should be collision_free=True (nothing there to collide
     with) but success=False (object never gets lifted / stays out of
     tolerance during the shake).
  3. A KNOWN COLLIDING pose (gripper placed so it intersects the object
     with fingers still open -- e.g. positioned so the object is INSIDE
     one of the finger volumes rather than between them) should be
     collision_free=False.
"""
import os
import numpy as np
import mujoco

import sim_common as sc
from object_specs import OBJECT_SPECS, spawn_pos, centroid_world, build_scene_xml, FLOATING_GRIPPER_TEMPLATE

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR = os.path.join(ROOT, 'generated_scenes')
os.makedirs(SCENES_DIR, exist_ok=True)

OBJECT = 'cylinder'
spec = OBJECT_SPECS[OBJECT]

fg_xml = os.path.join(SCENES_DIR, 'scene_cylinder_floating_gripper.xml')
build_scene_xml(OBJECT, fg_xml, template_path=FLOATING_GRIPPER_TEMPLATE)
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


def side_quat():
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


print('\n--- TEST 1: known-good side-grasp pose ---')
# Hand origin offset from the centroid along +x by less than the
# footprint radius (0.036m) plus the fingers' reach, so the pads
# straddle the cylinder's side at its mid-height.
X_OFFSET_GOOD = 0.12
grasp_pos_good = centroid + np.array([X_OFFSET_GOOD, 0., 0.])
grasp_quat_good = side_quat()
data = fresh_data()
result = sc.run_floating_gripper_test(
    model, data, spec['body_name'], grasp_pos_good, grasp_quat_good,
    footprint_radius=spec['footprint_radius'], squeeze_margin=60.)
print(result)
assert result['collision_free'], 'Expected collision-free at the good pose'
assert result['success'], 'Expected the good pose to succeed the shake test'
print('PASS')

print('\n--- TEST 2: known-bad grasp pose (30cm to the side, nothing there) ---')
grasp_pos_bad = centroid + np.array([0.30, 0., 0.])
data = fresh_data()
result = sc.run_floating_gripper_test(
    model, data, spec['body_name'], grasp_pos_bad, grasp_quat_good,
    footprint_radius=spec['footprint_radius'], squeeze_margin=60.)
print(result)
assert result['collision_free'], 'Expected collision-free (nothing near the gripper)'
assert not result['success'], 'Expected the bad (empty) pose to fail the shake test'
print('PASS')

print('\n--- TEST 3: known-colliding pose (gripper base pushed through the object) ---')
grasp_pos_collide = centroid.copy()  # hand origin planted at the object's centroid
data = fresh_data()
result = sc.run_floating_gripper_test(
    model, data, spec['body_name'], grasp_pos_collide, grasp_quat_good,
    footprint_radius=spec['footprint_radius'], squeeze_margin=60.)
print(result)
assert not result['collision_free'], 'Expected a collision at this pose'
print('PASS')

print('\n=== All floating-gripper smoke tests passed ===')
