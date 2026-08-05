# PROJECT CONTEXT — Paste this at the start of every new AI session
**Last updated:** 2026-07-09 (Week 7, Wednesday)
**Student:** Bonolo Masima | MSc project, University of Lincoln
**Supervisor:** Dezong Zhao | PhD advisor contact: Jingzhi Ruan

---

## What this project is

**Title:** Causal Inference for Robotic Grasp Failure Diagnosis under Perceptual Degradation

**Core question:** Can a Structural Causal Model (SCM) diagnose the root cause of robot grasp failures more reliably than a zero-shot LLM, when failures are caused by controlled perceptual degradation?

**Not:** A new grasping model. Not a perception model. A *causal reasoning study* that uses a pre-trained grasp algorithm as the system under test.

---

## Key design decisions (already made — do not re-open)

| Decision | What was chosen | Why |
|---|---|---|
| Simulator | MuJoCo (not Isaac Sim, not PyBullet) | Deterministic, CPU-runnable, no dependency hell |
| Grasp algorithm | Contact-GraspNet (PyTorch port, local) | Pre-trained, 6-DoF, point-cloud-driven |
| Robot | Franka Panda | Standard manipulation benchmark |
| Scene geometry | Reverting to Menagerie mesh files (STL collision + OBJ visual) | Capsule approach was overly cautious; CGN authors used full mesh URDF; Menagerie already has simplified collision meshes |
| Success criterion | Proximity fallback: EE XY within 0.065 m of object centroid after approach | Physical lift unreliable with primitive gripper; preserves causal signal |
| Causal variables | sigma_d (depth noise), rho (point cloud sparsity), phi (camera elevation), theta (camera azimuth) | All directly controllable; no confounders |

---

## Experimental design

**Grid (432 trials, already collected):**
| Variable | Symbol | Values |
|---|---|---|
| Depth noise (Gaussian σ on depth buffer) | σ_d | 0, 0.005, 0.02, 0.04 m |
| Point cloud sparsity (random downsample fraction) | ρ | 1.0, 0.75, 0.5, 0.25 |
| Camera elevation | φ | 30°, 45°, 60° |
| Camera azimuth | θ | 0°, 45°, 90° |
| Seeds per condition | — | 3 |

**Results so far:** 134/432 successes (31.0%); 141 trials returned `no_grasps` from CGN (mostly at φ=60°)

---

## Codebase — key files

| File | Purpose |
|---|---|
| `grasp_scene_v2.xml` | MuJoCo scene: Panda arm (capsule geometry), cylinder target object, perception camera |
| `mujoco_cgn_bridge.py` | MuJoCo depth + segmentation → CGN-compatible .npz → inference → world-frame grasp pose |
| `demo_grasp.py` | Visual pipeline: camera capture → CGN inference → pre-grasp → descend → close → lift → record |
| `run_experiments.py` | Headless 432-trial batch runner, CSV logging to `results/experiment_results.csv` |
| `grasp_simulation.py` | Depth rendering, Gaussian noise injection, pinhole back-projection, point cloud pipeline |
| `visualize_cgn_grasps.py` | Thesis figure generation (4-panel perception, 3D grasp distributions) |
| `msc_report.tex` | Main thesis LaTeX document |
| `thesis_direction_example.tex` | Detailed method write-up sections (camera model, IK, success criterion) |
| `project_pipeline.md` | Full living journal — all decisions, bugs, resolutions, open questions |
| `RIGOUR_LEDGER.md` | Rigour ledger — every design choice & assumption tracked per pipeline stage, with strength/weakness, whether it holds, severity, and disposition (keep-for-MSc / fix-now / future-work / redesign-candidate). Use to test the approach against new papers or criticisms, and as the input to any future experiment redesign |

---

## Current status (as of last update)

### Done
- MuJoCo environment with Panda arm (capsule geometry), cylinder target, perception camera
- Depth capture + pinhole back-projection + Gaussian noise injection
- Contact-GraspNet integration (PyTorch port, CPU inference)
- DLS Jacobian IK (`mj_jacSite`, arm qpos[7:14], joint7 wrist alignment)
- Gripper control via tendon actuator (ctrl[7]: 255=open, 0=closed)
- 432-trial dataset collected → `results/experiment_results.csv`
- CGN visualisation engine (headless Open3D, matplotlib fallback)
- Key bug fixed: IK was writing to qpos[0:7] (free joint of target object) instead of qpos[7:14] (arm joints)
- **SCM fitted** (9 July) — `scm_fit.py`, all 4 structural equations estimated, 6 thesis figures saved to `results/figures/scm_*.png`
  - Eq1 C_pc ~ φ+θ: R²=0.893, α_φ=−0.000259/deg ✓
  - Eq2A has_grasps ~ σ_d+ρ+φ+θ: pseudo-R²=0.554, AUC=0.943; σ_d OR≈0 (dominant), ρ n.s.
  - Eq2B n_grasps (NegBin): σ_d and ρ both significant — separates "pipeline collapse" from "count"
  - Eq3 q_grasp: R²=0.699; log(n_grasps) positive (order-statistic correction works)
  - Eq4 e_pose: σ_d mediated 10.9% through q_grasp; ρ 90.9% mediated — clean mediation structure
  - Outputs: `results/scm_coefficients.csv`, `results/scm_model.json`
- **Counterfactual ground truth complete** (9 July) — `run_counterfactual_groundtruth.py`
  - 292 failed trials × 4 interventions = 1,168 re-simulations in 261.8 min
  - Output: `results/counterfactual_groundtruth.csv`
  - Key finding: **57.2% of failures (167/292) have no single-variable fix** — multi-causal or irreducible
  - Primary cause breakdown: sigma_d=36, theta=35, phi=23, rho=1, joint=30, none=167
  - φ=60° accounts for 98/167 "none" cases (overhead geometry causes irreducible failure)

### In progress / next up
- **LLM baseline comparison** — prompt + scoring rubric must be written before any trials run (~14 July)
  - Must account for 57.2% multi-causal trials — rubric needs "none/joint" response category
  - Run at three observation depths: inputs only → +perception → +grasp quality
- **msc_report.tex** — Ch.7 SCM Fit Quality placeholder needs real numbers (now available)
- **Counterfactual diagnosis procedure** — run SCM abduction-action-prediction on the 292 failed trials
  - Compare SCM ΔP ranking against ground truth from `counterfactual_groundtruth.csv`

### Hard deadlines
- Experiment freeze: 27 July 2026 (superseded — see below)
- Report submission: 14 August 2026 → **extended** (preliminary marking response in progress; confirm new date)
- Poster: 19 August 2026

### Preliminary marking response (4 Aug 2026) — multi-object redesign in progress
Both markers flagged the single-cylinder scene as a validity problem, not
just a scope compromise: geometry must be a variable (Marker A: shape-
specific conclusions + rotational symmetry muddies pose-error evaluation;
Marker B: object geometry is a potential *moderator* of the degradation→
failure relationship, and scene clutter changes the causal mechanism
entirely — occlusion/collision vs. missed/slipped grasps). Full feedback
text: `MARKER_FEEDBACK.md`. Full response record: `RIGOUR_LEDGER.md` Stage 20.

**What was built this session (not yet run):**
- 2 new objects added to the existing cylinder: box (YCB GelatinBox mesh)
  and mustard bottle (YCB MustardBottle mesh, irregular/asymmetric) — real
  YCB visual meshes, simplified collision primitives. See `object_specs.py`.
- `sim_common.py` — object-agnostic simulation helpers, generalizing the
  original hardcoded single-cylinder qpos slicing to name-based joint
  lookup (needed for the clutter scene's 3 freejoints).
- `run_experiments_v2.py` (Experiment A) — same 3 objects, densified grid
  (σ_d and φ densified around known collapse/pathological regions,
  deterministic ρ downsampling, 5 seeds) → 7560 trials.
- `run_clutter_experiments.py` (Experiment B) — all 3 objects together in
  a fixed clutter arrangement, rotating grasp target, new
  `collision_with_neighbor` outcome variable → 504 trials.
- Local validation only (no GL rendering available in the dev sandbox):
  physics stability, CGN-determinism (bit-exact, see Stage 20), synthetic-
  input pipeline checks. **Trial batches have not been run.**
- Handoff: `RUNPOD_SETUP.md` — user is running the actual batches on a
  rented RunPod GPU pod (Linux + CUDA sidesteps the macOS rendering
  sandbox issue entirely).

**Next steps once data lands:** refit SCM with object identity as a
moderator; analyse `collision_with_neighbor` vs. σ_d/ρ; update DAG,
methods chapter, limitations section; update this file's "Current status"
and the experimental grid table above (still describes the *original*
432-trial single-cylinder grid, now superseded for the multi-object work).

---

## Known open issues / deferred decisions

1. **Gripper geometry revert** — Decision made 6 July: revert `grasp_scene_v2.xml` to Menagerie mesh-based body chain. Capsules were overly cautious; CGN authors used full mesh URDF. In progress. Professor noticed the disjointed appearance in Meeting 3 (3 July) and expects a fix or clearer justification.
2. **Success criterion** — Proximity fallback (not physical lift) must be clearly defended in the thesis. Physical lift is a stretch goal. (Already written in Ch.5 of msc_report.tex.)
3. **LLM test rubric** — Must be written *before* any LLM trials to avoid post-hoc interpretation bias. Critical: rubric must include a "none/joint" response category given the 57.2% multi-causal finding.
4. **Ch.7 SCM Fit Quality table** — Placeholder in msc_report.tex now has real numbers available from scm_coefficients.csv. Fill in before report freeze.
5. **SCM DAG update in thesis** — Current DAG in Ch.3 shows σ_d→C_pc and ρ→C_pc as edges; empirical audit confirmed C_pc is independent of both (computed before noise/downsample). DAG must be corrected.
6. **"none" attribution handling** — 57.2% of failures cannot be attributed to a single variable. The thesis limitations section (Ch.9 item 9) references "single-cause attribution" but the ground truth now quantifies this concretely. Update the limitations discussion with the 57.2% figure.

---

## What NOT to suggest (already rejected / not applicable)

- Do not suggest switching back to Isaac Sim
- Do not suggest training a new grasping model from scratch
- Do not suggest using lighting as a causal variable (MuJoCo depth buffer is geometric-only, unaffected by lighting)
- Do not re-open the capsule-vs-mesh debate — decision is made, reverting to Menagerie meshes
- Do not suggest adding more causal variables — scope is deliberately narrow for an MSc

---

## How to use this file

1. **At the start of every new chat session**, paste this file's contents (or attach it) before asking your question.
2. **At the end of every working session** (~5 min), update "Current status" and "Known open issues" to reflect what happened.
3. The full journal is in `project_pipeline.md` — if the model needs deep background on any decision, point it there.
4. **Rigour tracking lives in `RIGOUR_LEDGER.md`.** It exists because the project was built through a chain of design choices (simulator, CGN, MuJoCo, the 432-trial grid, structural-equation forms, NPSEM-ie, etc.), each carrying assumptions and weaknesses. Rather than overhauling the setup, the ledger logs each choice's assumption, strength, weakness, whether it still holds, severity, and a disposition (keep-for-MSc / fix-now / future-work / redesign-candidate). When a new paper or criticism arrives, test it against the ledger rows and update in place; before any experiment redesign, read only the `redesign-candidate` rows. This file should be referenced whenever the question is "is my approach still sound?" or "what should I change in a follow-up experiment?"
