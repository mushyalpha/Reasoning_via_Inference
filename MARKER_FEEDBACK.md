# Preliminary marking feedback — 4 Aug 2026

Verbatim feedback received from preliminary marking, preserved here so the
response (RIGOUR_LEDGER.md Stage 20, `object_specs.py`,
`run_experiments_v2.py`, `run_clutter_experiments.py`) can be checked
against the original wording later.

---

## Marker A

**Shape matters, and a cylinder alone is actually a bad choice.** Two reasons:

1. CGN's predictions are geometry-driven, so any causal structure you extract
   from one shape may be specific to that shape. Conclusions from a cylinder
   are conclusions about cylinders.
2. Worse, a cylinder is **rotationally symmetric**, which makes evaluation
   ambiguous: many distinct poses are functionally identical grasps, and
   errors along the symmetry axis don't matter. This muddies exactly the
   pose-error → success relationship you're studying.

**How many?** A defensible minimal set is **3–5 objects** spanning:

- A box/cuboid (flat faces, edges)
- A cylinder (curved surface — keep yours)
- Something with a concavity or handle (e.g., a mug)
- Something small relative to gripper width, and/or something irregular

The **YCB object set** is the standard choice - meshes are freely
available, MuJoCo-compatible, and using it means zero justification burden
("we use standard YCB objects").

## Marker B

**The key framing for your thesis:** intervention variable is perception
degradation, outcome is grasp success, and object geometry is a potential
**moderator**. With one object you can't distinguish "degradation X causes
failure" from "degradation X causes failure *on cylinders*." With 3–5
diverse objects, you can either show

> the causal structure is consistent across geometry (stronger claim)

or

> that it interacts with geometry (interesting finding in itself).

Either way you win — with one object, you can't tell which world you're in.

Because your thesis focuses on **degrading the perception system to
extract the causal structure of the process**, **the number of objects in
the scene matters immensely.**

The causal dynamics shift completely based on scene complexity:

```
[ Single Isolated Object Scene ]
  Degraded Perception ───> Sparser Point Cloud ───> Fewer Grasp Candidates ───> Lower Success Rate

[ Multi-Object Cluttered Scene ]
  Degraded Perception ───> Shifting Object Boundaries ───> Spatial Occlusions / Ghost Artifacts
                                                                     │
                                                                     ▼
                                                      Finger Collides with Object B
                                                      While Attempting to Grab Object A
```

### Why 1 Object vs. Clutter Changes the Science

- **With 1 Object (Singulation):** You are evaluating the *direct causal
  link* between visual clarity and grasp precision. If you degrade the
  point cloud, the surface normals become noisy, and the network can no
  longer tell if a surface is flat or tilted. The failure mode here is
  purely **missed grasps or slipping**.
- **With Multiple Objects (Clutter):** You introduce confounding causal
  variables like **spatial occlusion and collision boundaries**. When you
  degrade a cluttered scene, the network can no longer cleanly separate
  where Object A ends and Object B begins. A grasp pose might look
  geometrically flawless on a degraded point cloud, but executing it
  causes the physical Panda fingers to smash violently into an adjacent
  object that was visually blurred or obscured.

### Recommendation for Your Thesis

Do not use a million objects, but do not use just one. Structure your
experiments into two clean, distinct chapters or sections:

1. **Experiment A (The Baselines):** Test 3–5 isolated objects. Measure how
   your perception degradation directly scales down grasp confidence
   scores and success metrics.
2. **Experiment B (The Complex Causal Test):** Throw those same 3–5
   objects together into a single, cluttered pile. Show how perception
   degradation introduces unexpected causal failures — specifically
   proving that visual noise in clutter leads to **inter-object
   collisions** rather than just simple missing/slipping errors.

---

## Response summary (see RIGOUR_LEDGER.md Stage 20 for full detail)

- Objects: cylinder (kept) + box (YCB GelatinBox) + mustard bottle (YCB
  MustardBottle, irregular/asymmetric) — real YCB visual meshes via the
  `eleramp/pybullet-object-models` mirror (`assets/ycb/`).
- Experiment A: `run_experiments_v2.py` — same 3 objects, expanded/densified
  grid (dense σ_d and φ, deterministic ρ, 5 seeds), 7560 trials.
- Experiment B: `run_clutter_experiments.py` — all 3 objects in a fixed
  clutter arrangement, rotating grasp target, new `collision_with_neighbor`
  outcome variable, 504-trial targeted grid.
- Known gap: the "concavity/handle" category (e.g. a mug) is not covered by
  the 3 chosen objects — documented trade-off, candidate 4th object if time
  allows.
- Status as of 4 Aug: code built and locally validated (physics stability,
  CGN determinism, synthetic-input pipeline checks); trial batches not yet
  run — see `RUNPOD_SETUP.md` for the handoff to a GPU pod.
