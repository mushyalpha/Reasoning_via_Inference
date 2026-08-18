"""
object_specs_v4_patch.py
=========================
Registers additional cylinder radius variants into object_specs.OBJECT_SPECS
at runtime (monkeypatch), WITHOUT modifying object_specs.py on disk.

Motivation (ARM_IK_ATTEMPT_LOG.md redesign lever (b)):
  Original cylinder footprint_radius=0.036m vs gripper half-opening
  FINGER_OPEN=0.04m leaves only ~4mm clearance per side at the moment of
  grasp -- comparable to or smaller than the arm IK's residual position
  error (~5-10mm), so the fingers frequently make asymmetric/marginal
  contact before the intended squeeze even starts.

  This does NOT change the floating-gripper baseline (which teleports
  exactly, so the tight original cylinder is fine for that protocol) --
  only adds new keys for the full-arm redesign experiment.

Variants:
  cylinder_thin    r=0.030m (6.0cm diameter, ~standard drink-can size)
                   -> 1.0cm clearance per side
  cylinder_thinner r=0.026m (5.2cm diameter)
                   -> 1.4cm clearance per side

Usage:
    import object_specs_v4_patch  # side-effect: registers new keys
    from object_specs import OBJECT_SPECS
    OBJECT_SPECS['cylinder_thin']  # now available
"""

import copy
from object_specs import OBJECT_SPECS

_BASE = OBJECT_SPECS['cylinder']


def _make_variant(radius, mass_scale=1.0, suffix=''):
    spec = copy.deepcopy(_BASE)
    half_height = spec['primitive_size'][1]
    spec['label'] = f'Cylinder r={radius*100:.1f}cm (arm-redesign variant{suffix})'
    spec['primitive_size'] = (radius, half_height)
    spec['footprint_radius'] = radius
    spec['mass'] = round(_BASE['mass'] * mass_scale, 4)
    spec['inertial_mass'] = spec['mass']
    return spec


OBJECT_SPECS['cylinder_thin'] = _make_variant(0.030)
OBJECT_SPECS['cylinder_thinner'] = _make_variant(0.026)

if 'cylinder_thin' not in __import__('object_specs').OBJECT_NAMES:
    __import__('object_specs').OBJECT_NAMES.append('cylinder_thin')
    __import__('object_specs').OBJECT_NAMES.append('cylinder_thinner')
