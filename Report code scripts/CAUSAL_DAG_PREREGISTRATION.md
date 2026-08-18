# Causal DAG Pre-Registration

**Date registered:** 5 August 2026
**Registered by:** Bonolo Masima, from a joint session re-deriving the DAG from source rather than from the previously-drawn figure (`scm_fit.py::fig_dag`).
**Status:** Layer 1 (graph) only. Functional forms (Layer 2) are deliberately left unspecified — see `scm_nonparametric.py` and the "Two layers" note at the end of this file.
**Git context:** commit this file in the same commit as the code it cites, so the citations below stay pinned to a specific version of the pipeline. If the pipeline code changes later, this document must be re-derived, not hand-edited — that is the whole point of deriving it from dataflow rather than intuition.

---

## 0. Why this document exists, and the rule it follows

An SCM has two layers: the graph (a causal claim) and the functional forms (a statistical-modelling choice). This document handles **only** the graph.

Ordinarily, deciding a causal graph requires either domain expertise (contestable), causal discovery algorithms (require assumptions of their own — faithfulness, no latent confounding — that are just as strong as the thing being discovered), or randomized experiments (gold standard, but usually only available for a handful of edges at a time). This project is in an unusual position: **the true SCM is not a model we are guessing at — it is the pipeline's dataflow.** The renderer, the noise-injection code, Contact-GraspNet's forward pass, and MuJoCo's physics step are a fully known computational graph, sitting on disk, in this repository. We can read it.

The rule applied throughout this document:

> A variable computed at pipeline stage *k* cannot be caused by anything computed at stage *k+1*. Every edge (and every deliberately *absent* edge) below must be justifiable as "the code that computes X reads the value of Y" (edge X→Y... wait, direction: "the code that computes Y reads the value of X" ⇒ edge X→Y), with a file:line citation. If no such citation exists, there is no edge, regardless of how correlated X and Y turn out to be in the data.

This is exactly the reasoning that caught the original error: `C_pc` is computed before noise/downsampling are applied, so σ_d and ρ *cannot* have edges into it — not a judgement call, a dataflow fact (§2 below). The same procedure, applied a second time in this session, catches a **second, previously undocumented error**: the current DAG (`scm_fit.py::fig_dag`) draws `q_grasp → e_pose` as a mediation edge, and `RIGOUR_LEDGER.md` Stage 12/Eq4 reports a "% mediated via q_grasp." Reading `sim_common.py::best_grasp_overall` / `run_experiments.py::best_grasp_cam` shows this edge does not exist in the code (§4 below). This is corrected here.

The four exogenous variables (σ_d, ρ, φ, θ) need no argument for having no incoming edges: they are set by the experimenter via `np.random.default_rng(seed)` draws and grid assignment (`run_experiments.py` main loop), never read from any other pipeline variable. **Exogeneity by randomization.**

---

## 1. Nodes and where each one is computed

| Node | Meaning | Computed by | Reads |
|---|---|---|---|
| σ_d | Depth-noise std dev (m) | Experimenter-set, `run_experiments.py` grid loop | nothing (exogenous) |
| ρ | Point-cloud keep fraction | Experimenter-set, grid loop | nothing (exogenous) |
| φ | Camera elevation (deg) | Experimenter-set, grid loop | nothing (exogenous) |
| θ | Camera azimuth (deg) | Experimenter-set, grid loop | nothing (exogenous) |
| `C_pc` | Fraction of image pixels showing the target (viewpoint coverage) | `run_experiments.py:418` `C_pc = seg_map.sum() / (IMG_W*IMG_H)` | `seg_map` only |
| `S` (unobserved, compound) | CGN's full internal scored-candidate output: the `{pred_grasps_cam, scores}` dict returned by `estimator.predict_scene_grasps` | `run_experiments.py:248-275` (`run_cgn`) | `depth` (=depth_noisy), `pc_full` (downsampled by ρ), `seg_map` |
| `n_grasps` | Count of candidates in `S` | `run_experiments.py:423` `n_grasps = sum(len(scores[k]) for k in scores)` | `S` (via `scores`) |
| `has_grasps` | 1{n_grasps > 0} | `run_experiments.py:425` (`if n_grasps == 0: ... error='no_grasps'`) | `n_grasps` |
| `q_grasp` | Score of the argmax-selected candidate | `run_experiments.py:278-287` (`best_grasp_cam`): `idx = argmax(s)`; `q_grasp = s[idx]` | `S` (via `scores`), `n_grasps` (sample-size / order-statistic effect, see §4) |
| `pose_cam`, `pose_world`, `grasp_pos` (intermediate, not logged) | The pose at the *same* `idx` | `run_experiments.py:286` `best_pose = g[idx]`, then `cam_to_world` (`run_experiments.py:290-297`) | `S` (via `pred_grasps`, same `idx` as `q_grasp` — see §4), camera extrinsics (`cam_xmat`/`cam_xpos`, functions of φ, θ) |
| `e_pose` | ‖true object XY − proposed grasp XY‖ | `run_experiments.py:441` `e_pose = norm(obj_pos[:2] - grasp_pos[:2])` | `grasp_pos` (⇒ `S`, φ, θ); `obj_pos` is a fixed constant, not a random variable, in the single-object scenes |
| `Y` (success) | 1{post-execution EE within `GRASP_RADIUS` of object} (historical proximity metric, `experiment_results.csv`) — **superseded** for new data by the floating-gripper shake test (`RIGOUR_LEDGER.md` Stage 21) | `run_experiments.py:347-395` (`execute_grasp`): re-measures `xy_dist` from a **fresh** `ee_pos`/`obj_pos` read, post-IK | `grasp_pos` (via the IK target), **not** `e_pose` or `q_grasp` by name (see §5) |

**Nothing in this table was estimated from `experiment_results.csv`.** All of it comes from reading the four files that constitute the actual structural equations: `run_experiments.py`, `sim_common.py`, `mujoco_cgn_bridge.py`, `contact_graspnet_pytorch/contact_graspnet_pytorch/contact_grasp_estimator.py`. §6 lists the interventional tests that were then run against the data to check this table for errors — that is the falsification step, done *after* the graph was fixed, not used to build it.

---

## 2. Edge: φ, θ → `C_pc` (and the non-edges: σ_d, ρ ↛ `C_pc`)

`run_experiments.py:215-241` (`render_depth_seg`):

```215:241:run_experiments.py
def render_depth_seg(model, data, sigma_d=0.0, rng=None):
    ...
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    depth_raw = renderer.render().copy()
    renderer.disable_depth_rendering()

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    seg = renderer.render(); renderer.close()

    tgt_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    gid_img = seg[:, :, 0]
    seg_map = np.zeros(gid_img.shape, dtype=np.int32)
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == tgt_bid:
            seg_map[gid_img == gid] = 1
    if seg_map.sum() == 0:
        seg_map = ((depth_raw > 0.2) & (depth_raw < 1.5)).astype(np.int32)

    depth_noisy = (np.clip(depth_raw + rng.normal(0., sigma_d, depth_raw.shape),
                            0., None).astype(np.float32)
                   if sigma_d > 0. else depth_raw.copy())
    return depth_noisy, build_K(model), seg_map
```

`seg_map` (lines 229-236) is fully computed — from the **segmentation** render channel, keyed by geometry id, which depends only on which geometry is visible from the camera pose set by φ,θ (`set_camera(model, phi, theta)` at `run_experiments.py:412`) — **before** `depth_noisy` is computed on line 238. σ_d is added to `depth_raw` to produce `depth_noisy` on line 238, strictly after `seg_map` is final; `C_pc` (`run_experiments.py:418`) reads `seg_map`, never `depth_noisy`. ρ plays no role anywhere in this function — downsampling happens later, inside `run_cgn` (§3), on `pc_full`, which `C_pc` never touches.

**Edges:** φ → `C_pc`, θ → `C_pc`.
**Non-edges (justified, not assumed):** σ_d ↛ `C_pc`, ρ ↛ `C_pc`.

This reproduces the original PhD-student-caught fix and is the template the rest of this document follows.

---

## 3. Edge: σ_d, ρ, φ, θ → `S` (and hence → `n_grasps`, `has_grasps`, `q_grasp`, `e_pose`)

`run_experiments.py:248-275` (`run_cgn`):

```248:275:run_experiments.py
def run_cgn(depth, K, seg_map, estimator, rho=1.0, rng=None):
    ...
    pc_full, pc_segs, _ = estimator.extract_point_clouds(
        depth_in, cam_K, segmap=segmap, rgb=rgb,
        skip_border_objects=False, z_range=[0.1, 2.0])

    if rho < 1. and len(pc_full) > 0:
        n = max(1, int(len(pc_full) * rho))
        pc_full = pc_full[rng.choice(len(pc_full), size=n, replace=False)]
        ...
    with torch.no_grad():
        pred_grasps, scores, _, _ = estimator.predict_scene_grasps(
            pc_full, pc_segments=pc_segs,
            local_regions=True, filter_grasps=True, forward_passes=1)
    ...
    return pred_grasps, scores
```

`depth` here is `depth_noisy` (parents: σ_d via the noise term, φ,θ via the underlying geometry). `pc_full`/`pc_segs` are extracted from that noisy depth, **then** downsampled by ρ (lines 259-267), **then** passed into CGN's forward pass. So the compound object `S = (pred_grasps, scores)` has all four exogenous variables as parents: **σ_d, ρ, φ, θ → S**.

`n_grasps = sum(len(scores[k]) for k in scores)` (`run_experiments.py:423`) is a pure count over `S`. `has_grasps = 1{n_grasps > 0}` (`run_experiments.py:425`). Both inherit `S`'s parents.

**Edges:** σ_d, ρ, φ, θ → `S` → `n_grasps` → `has_grasps`.

---

## 4. The correction: `q_grasp` and `e_pose` are siblings, not a mediation chain

This is the edge the original DAG got wrong, caught the same way `C_pc` was caught: by reading, not by intuition.

`run_experiments.py:278-297`:

```278:297:run_experiments.py
def best_grasp_cam(pred_grasps, scores):
    best_score, best_pose = -1., None
    for obj_id in pred_grasps:
        s, g = scores[obj_id], pred_grasps[obj_id]
        if len(s) == 0:
            continue
        idx = int(np.argmax(s))
        if float(s[idx]) > best_score:
            best_score, best_pose = float(s[idx]), g[idx].copy()
    return best_pose, best_score
```

`q_grasp` (= `best_score` = `s[idx]`) and `pose_cam` (= `best_pose` = `g[idx]`) are read from the **same index** `idx` of the **same underlying object `S`**. `idx` is a function of the full `s` array (`argmax`). Critically: **the code that computes `pose_cam`/`e_pose` never reads the value of `q_grasp`.** `e_pose` is computed from `grasp_pos` (`run_experiments.py:436`, derived from `pose_cam`/`g[idx]`) and a fixed `obj_pos` (`run_experiments.py:440`) — `q_grasp`'s value (`s[idx]`) does not appear anywhere in that computation.

This means:

- **There is no `q_grasp → e_pose` edge.** `scm_fit.py::fit_eq4_e_pose`'s framing of `q_grasp` as a "mediator" and its reported "% mediated via q_grasp" (`RIGOUR_LEDGER.md` Stage 12, `msc_report.tex:474-522`) is not a dataflow-justified causal claim. It is a **selection artifact**: both variables are indexed by the same `argmax(S.scores)`, so they will be correlated in the data (confirmed empirically, §6.2) without either causing the other.
- The correct structure is: `S → q_grasp` and `S → e_pose` as **siblings under a common parent**, with **correlated error terms** (the shared `idx`). This is a concrete, dataflow-provable **violation of the NPSEM-ie independent-errors assumption** (`RIGOUR_LEDGER.md` Stage 15) for this specific pair of variables — worth stating explicitly rather than assuming independence project-wide.
- `n_grasps → q_grasp` **is** kept as a directed edge, but for a different, legitimate reason: `q_grasp = max` over an array whose length *is* `n_grasps`; by order-statistics, the expected maximum of a same-distributed pool mechanically increases with pool size. `scm_fit.py::fit_eq3_q_grasp`'s `log(n_grasps)` correction is doing exactly the right thing here (this one is *not* a spurious link, it is a real sample-size effect on a `max` operator) — kept unchanged.

**Corrected local structure:**

```
S ─┬─→ n_grasps ──→ q_grasp   (n_grasps → q_grasp: order-statistic / sample-size effect)
   └─→ e_pose                  (same S, same idx as q_grasp; NOT caused by q_grasp)
φ, θ ────────────────→ e_pose  (second path: camera-to-world transform, run_experiments.py:290-297)
```

---

## 5. `Y` (success): what it actually reads

`run_experiments.py:347-395` (`execute_grasp`):

```385:395:run_experiments.py
    site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    obj_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    ee_pos   = data.site_xpos[site_id]
    obj_pos  = data.xpos[obj_id]
    xy_dist  = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))

    success  = xy_dist < GRASP_RADIUS
```

`Y`'s code does **not** read `e_pose`'s stored value or `q_grasp`'s. It recomputes an equivalent quantity (`xy_dist`) from a **fresh** `ee_pos` (post-IK, post-settle, post-gripper-close) and a fresh `obj_pos` read. Since IK is targeted at `grasp_pos` — the same quantity `e_pose` was computed from — and converges to within 1.5 cm (`run_experiments.py:376`, `tol=0.015`) against a 6.5 cm success threshold, `xy_dist ≈ e_pose` whenever IK converges. This is why the two agree 97.9% of the time (`RIGOUR_LEDGER.md` Stage 9) — not because `e_pose` causes `Y`, but because both are near-duplicate measurements of the same underlying quantity (`grasp_pos` vs. true object position), taken at two points in the pipeline.

**Edges into `Y` (historical proximity metric, `experiment_results.csv`):** `has_grasps` (→ `Y=0` directly when 0), and σ_d, ρ, φ, θ via `grasp_pos` (i.e. via `S`) — **not** via `e_pose` or `q_grasp` as causal mediators, for the same reason as §4. This also reinforces, from a different angle, why `RIGOUR_LEDGER.md` Stage 10 correctly flags this metric as *circular with the SCM's own mediator* — the DAG derived here shows precisely why: `Y` and `e_pose` are two measurements of the same underlying `grasp_pos` error, not outcome-and-cause.

For any data collected under the **floating-gripper shake-test** outcome (`RIGOUR_LEDGER.md` Stage 21, `sim_common.run_floating_gripper_test`), `Y` becomes a genuinely distinct downstream node (physical disturbance response), still caused by σ_d, ρ, φ, θ via `grasp_pos`/the full 6-DoF pose, but no longer a near-copy of `e_pose`. This document's graph is written to be correct for both outcome definitions; only the interpretation of the `Y` node's independence from `e_pose`/`q_grasp` improves under Stage 21's metric.

---

## 6. Full corrected DAG

```
σ_d ─┐
ρ    ├──────────────────────────────→ S ─┬──→ n_grasps ──→ has_grasps ──→ (has_grasps=0 ⇒ Y=0)
φ ───┼──→ C_pc                           ├──→ q_grasp  (sibling of e_pose, NOT its cause)
θ ───┘                                   └──→ e_pose ←── φ, θ (2nd path, frame transform)
                                                           │
                                          (has_grasps=1) → Y   [historical metric: Y ≈ duplicate
                                                                 measurement of e_pose's underlying
                                                                 quantity, not caused by it;
                                                                 Stage-21 metric: Y is a distinct
                                                                 physical outcome]
```

Compared to `scm_fit.py::fig_dag` (the previously-registered figure), this DAG:
1. Keeps: φ,θ → `C_pc`; σ_d,ρ,φ,θ → `has_grasps`/`n_grasps`; `n_grasps` → `q_grasp`.
2. **Removes:** `q_grasp → e_pose` (redrawn as siblings with correlated errors under common parent `S`).
3. **Makes explicit** a previously-implicit node, `S` (CGN's raw scored-candidate output), as the actual parent of `n_grasps`, `q_grasp`, and `e_pose`, rather than drawing `has_grasps → n_grasps → q_grasp → e_pose` as one causal chain.

---

## 7. Exogeneity by randomization

σ_d, ρ, φ, θ have no incoming edges because the experimenter sets them directly in the grid loop (`run_experiments.py` `main()`, full factorial assignment) and independently draws the trial's RNG seed (`np.random.default_rng(seed)`, `run_experiments.py:403`) before any pipeline code runs. No pipeline code reads σ_d/ρ/φ/θ from any other variable in this system. This is not an assumption to be defended statistically (e.g. via a balance test) — it is true by construction of the experiment loop, the same way a randomized controlled trial's treatment assignment is exogenous by design, not by post-hoc adjustment. (§8.1 below runs a balance check anyway, as a sanity test, not because it is load-bearing.)

---

## 8. Interventional falsification tests

Because this project owns the simulator, every edge above can be tested **interventionally** rather than merely by conditional-independence (d-separation) reasoning on observational data. Since σ_d, ρ, φ, θ are independently randomized by the grid design, conditioning on a subset of them **is** an intervention (`do(·) = condition on ·`, exactly as in a designed experiment) — no backdoor adjustment is needed. Full results, numbers, and pass/fail verdicts are in `results/dag_edge_falsification_report.md`, generated by `test_dag_edges.py`; do not hand-edit that file. Summary:

1. **Non-edges σ_d → `C_pc`, ρ → `C_pc`:** fix nothing (φ,θ vary freely too, since they're independent of σ_d/ρ by design) and regress `C_pc` on all four; the σ_d and ρ coefficients must be statistically indistinguishable from 0 and small enough to matter practically. If either moves `C_pc` detectably, this document's DAG (or the measurement definition of `C_pc`) is wrong, not the data.
2. **Edges φ, θ → `C_pc`:** same regression; φ's coefficient must be non-zero (θ may or may not be, depending on object symmetry — recorded, not assumed).
3. **`q_grasp`/`e_pose` sibling claim:** test whether `q_grasp` still predicts `e_pose` after conditioning on {σ_d, ρ, φ, θ, n_grasps}. A non-zero residual link is *expected* under the sibling/shared-`idx` reading (§4) and must **not** be reinterpreted as evidence for a `q_grasp → e_pose` edge — the dataflow trace in §4 already rules that direction out; the statistical test alone cannot distinguish direction (chain and fork produce the same conditional-independence signature), which is exactly why the code-reading step, not the statistics, is what settles this edge.
4. **σ_d, ρ, φ, θ mutual independence (design check):** pairwise correlations among the four exogenous variables should be ≈0 (they are assigned via `itertools.product` over independent grids, not sampled), confirming no accidental confounding was introduced by the grid construction itself.

---

## 9. The two layers, and why only this document is pre-registered

- **Layer 1 (this document):** the graph above. Pre-registered, dated, independent of any results — it was derived by reading `run_experiments.py`, `sim_common.py`, and `contact_grasp_estimator.py`, not by looking at `experiment_results.csv`. If §8's tests come back violating an edge, that is a **finding** (the code doesn't do what was assumed, or the measurement of a node is wrong) — not a reason to quietly redraw the graph to fit the data.
- **Layer 2 (functional forms — deliberately NOT pre-registered here):** `scm_nonparametric.py` implements Option A — every structural equation `f` (`C_pc = f_C(φ,θ,U_C)`, `q_grasp = f_q(n_grasps, S, U_q)`, etc.) is left **unspecified**. Because the design is a complete factorial grid with independent randomization, every causal estimand this thesis needs (total effects, φ-moderated effects, has_grasps-mediated path effects) is identified directly from stratified empirical averages over the graph in §6 — no linear/logistic functional form is fit or defended. This is the only fully correct choice for a system whose true mechanism is a deep network (CGN's confidence is not a linear function of anything upstream of it).
