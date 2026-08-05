# YCB object meshes

Geometry sourced from the YCB Object and Model Set (Calli et al., 2015).

- `mustard_bottle/` — YcbMustardBottle (irregular/asymmetric category: curved,
  tapered, off-centre mass). Visual mesh (`textured_simple_reoriented.obj`,
  renamed `visual.obj`) and V-HACD convex-decomposition collision mesh
  (`collision_vhacd.obj`, renamed `collision.obj`, present on disk but not
  loaded into the scene -- collision uses a fitted `type="box"` primitive
  instead, see `object_specs.py`) via the pre-processed mesh mirror at
  https://github.com/eleramp/pybullet-object-models. Texture/material files
  stripped since this project uses depth-only rendering.

- `sugar_box/` — 004_sugar_box (box/cuboid category: flat faces, edges).
  Visual mesh (`textured.obj`, renamed `visual.obj`, mtllib/usemtl lines
  stripped) taken directly from the official YCB Benchmarks mesh release
  (`google_16k` scan, http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/)
  rather than the pybullet-object-models mirror above, which does not carry
  this object. No collision mesh is shipped for this object at all: it is an
  almost-perfect rectangular prism, so collision uses a `type="box"`
  primitive fit directly to the mesh's own bounding box (see
  `object_specs.py`).

  Replaces an earlier choice, YcbGelatinBox (from the pybullet-object-models
  mirror): at ~2.8cm thick it was a pathological case for the parallel-jaw
  gripper (fingertip pads struggled to seat flush before contact-closing).
  See `RIGOUR_LEDGER.md` Stage 21 for the full account.

Cite: Calli, B., Singh, A., Walsman, A., Srinivasa, S., Abbeel, P., & Dollar,
A. M. (2015). The YCB object and Model set: Towards common benchmarks for
manipulation research. IEEE ICAR.
