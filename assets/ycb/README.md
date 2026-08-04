# YCB object meshes

Geometry sourced from the YCB Object and Model Set (Calli et al., 2015), via the
pre-processed mesh mirror at https://github.com/eleramp/pybullet-object-models
(visual mesh `textured_simple_reoriented.obj` + V-HACD convex-decomposition
collision mesh `collision_vhacd.obj`, texture/material files stripped since
this project uses depth-only rendering).

- `gelatin_box/` — YcbGelatinBox (box/cuboid category: flat faces, edges)
- `mustard_bottle/` — YcbMustardBottle (irregular/asymmetric category: curved,
  tapered, off-centre mass)

Cite: Calli, B., Singh, A., Walsman, A., Srinivasa, S., Abbeel, P., & Dollar,
A. M. (2015). The YCB object and Model set: Towards common benchmarks for
manipulation research. IEEE ICAR.
