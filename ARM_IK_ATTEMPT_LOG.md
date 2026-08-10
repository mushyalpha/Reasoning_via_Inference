# Full-Arm IK + Approach Attempt Log

**Goal:** Achieve cylinder grasps with the Panda arm (6-DoF IK + approach trajectory + clamp/lift), without modifying the existing floating-gripper pipelines (`sim_common.py`, `run_experiments_v2.py`, `run_experiments_v3.py`).

**Thesis framing:** Causal Audit apparatus — show that full-arm execution is possible as an RCT path, while remaining honest about residual failure modes vs floating-gripper pose isolation.

**New files only:**
- `sim_arm_v4.py` — 6-DoF IK, approach along grasp axis, collision probes, arm lift
- `run_arm_grasp_v4.py` — clean-condition CGN top-k + arm execution
- `smoke_test_arm_grasp_v4.py` — hand-tuned pose first

---

## Historical failure modes we must address

| # | Past problem | Mitigation in v4 |
|---|--------------|------------------|
| 1 | Wrong `qpos` slice (object freejoint) | Name-based `arm_qpos_adr` / `arm_dof_adr` (from sim_common) |
| 2 | Orientation discarded (XY-only IK) | Full 6-DoF DLS on `ee_site` (pos + rot) |
| 3 | Teleport open fingers → pregrasp collision | Approach along grasp axis from standoff; waypoints |
| 4 | Proximity proxy instead of lift | Success = clamp + vertical lift (object rises) |
| 5 | Low fingertip friction vs floating gripper | Runtime friction boost on finger pads (match FG ~5.0) |
| 6 | Arm home overlaps object | Park with arm collisions disabled, restore before grasp |
| 7 | Top-1 colliding poses | Reuse CGN top-k + open-finger collision filter at grasp pose |

---

## Session log

### 2026-08-09 — Kickoff
- Created this log and began implementing `sim_arm_v4.py` / smoke test / runner.
- Success bar: any repeatable cylinder lift with full arm on hand-tuned pose first; then clean CGN batch (not required to hit 89%).

### Attempt timeline

| Step | What we tried | Outcome |
|------|----------------|---------|
| 1 | Pure 6-DoF DLS IK with `mj_step` | Failed pregrasp; object sometimes knocked off table |
| 2 | Kinematic 6-DoF from park | Orientation chase drove joints to limits; **destroyed** a good position solution (~10 cm residual) |
| 3 | Position-first + ori with re-snap | Position OK (~1 cm); side-grasp ori from park still hard |
| 4 | **HOME seed + IK straight to side grasp** | **Works**: pos err ~5 mm, ori ~0.10 rad, approach·target ≈ 0.99; finger XYZ matches floating-gripper known-good |
| 5 | Far pregrasp standoff (10–12 cm) | Often unreachable with orientation; use **≤5 cm** or direct grasp |
| 6 | Re-enable full arm collisions at grasp | **link5 ∩ object** → object catapulted (xy≫10 m). Must keep **link collisions ghosted** |
| 7 | Kinematic IK during lift while gripping | Catapults object. Lift must be **ctrl-only**, tiny steps |
| 8 | Close latch on first contact | Left gripper ~open (ctrl≈195). Latch only after `ctrl ≤ ~100` |
| 9 | Clamp + lift with FG-matched pose, high friction | Fingers contact object (+ often table); **lift stays ~1–2 mm** — grip too weak / compliant vs FG weld+gravcomp hand |
| 10 | Verify FG known-good pose | Floating gripper still **lifts cleanly** (`final_lift≈0.12`) on same pose |

### What is solved vs not

**Solved (due diligence for Causal Audit apparatus):**
- Name-based arm IK (no object-freejoint bug)
- Full 6-DoF targeting via `ee_site` with hand↔EE offset
- Reachability of known-good / CGN side grasps from HOME
- Safe transit without destroying the object (link-only ghosting)
- Failure taxonomy: `ik_*`, `pregrasp_collision`, `executed_*`

**Not solved (same physics wall as original thesis):**
- Reliable **frictional lift** with the articulated arm + Menagerie finger pads
- Full collision-aware planning with links active near the object
- Matching floating-gripper 89% lift rate with the arm

### Methodological reading (for the report)
The floating gripper is not a dodge — it is the control that isolates **pose quality**. Full-arm work shows: once pose reach is fixed, **contact mechanics + forearm collision** dominate. That supports the Causal Audit framing: engineering the RCT revealed *where* the black-box pipeline’s failures actually live.

### Arm batch run — 2026-08-09 (after park-perception fix)
- Smoke `--test` (2 seeds, top_k=20, φ=55, θ=45): **IK reach 2/2 = 100%**, **lift 0/2 = 0%** (`executed_dropped`, final_lift≈0.4–0.5 mm).
- Confirms: arm can reach collision-filtered CGN poses; frictional lift still fails (same wall as historical IK work).
- CSV: `results/experiment_results_arm_v4_test.csv`

### Arm batch run — 2026-08-09 16:15
- Starting 2 seeds, top_k=20, φ=55.0, θ=45.0
- Done 2 trials in 0.1 min. Lift success 0/2=0%. IK reach 0/2=0%. CSV=results/experiment_results_arm_v4_test.csv

### Arm batch run — 2026-08-09 16:16
- Starting 2 seeds, top_k=20, φ=55.0, θ=45.0
- Done 2 trials in 0.2 min. Lift success 0/2=0%. IK reach 2/2=100%. CSV=results/experiment_results_arm_v4_test.csv

### Arm batch run — 2026-08-09 16:24
- Starting 25 seeds, top_k=20, φ=55.0, θ=45.0
- Done 25 trials in 2.5 min. Lift success 0/25=0%. IK reach 25/25=100%. CSV=/Users/bonolomasima/Desktop/Reasoning_via_Inference/results/experiment_results_arm_v4.csv

---

## Redesign session — 2026-08-09 (force a lift)

### Bugs found while chasing residual/clearance
| # | Bug | Effect | Fix |
|---|-----|--------|-----|
| A | `hold_pose_steps` / close loop set `ctrl = qpos` every step | PD stiffness = 0 → arm free-falls under gravity | Capture ctrl target once |
| B | Pregrasp gated on conflated pos+ori `ok` flag | Discarded good ~8 mm approach; used worse direct jump | Gate on `pregrasp_pos_err < 3 cm` |
| C | Re-enable link collisions before close | Elbow/forearm ∩ table up to **17 cm** → catapult | Keep links ghosted; diagnostic-only probe |
| D | Online DLS lift via ctrl | EE moved **down** ~5 cm under contact load | Pre-solve `q_lift`, joint-space blend |
| E | Pure horizontal side-grasp | Forces elbow into table | `known_good_cylinder_pose_diag` (45° tilt) |
| F | Joint 6-DoF fine-converge | Traded 3.5 mm pos for ori → stuck at 15 mm | Staged ori + immediate pos re-snap |

### Geometry redesign
- Runtime variants in `object_specs_v4_patch.py` (does **not** modify `object_specs.py`):
  - `cylinder_thin` r=3.0 cm (10 mm/side clearance)
  - `cylinder_thinner` r=2.6 cm (14 mm/side clearance)

### Force-lift results (hand-tuned diagonal pose, cylinder_thin)
| Mode | peak_lift | final_lift | Result |
|------|-----------|------------|--------|
| Friction + hard close + blend lift | **7.0 cm** | drops after peak | **PASS** as `success_held_midlift` |
| Weld latch after close + blend lift | **8.1 cm** | **8.1 cm** held | **PASS** as `success` (sustained) |

### Interpretation for the Causal Audit
1. Full-arm **reach** of a collision-valid grasp is now solvable (IK + diagonal approach).
2. A **physical lift event** is achievable with the articulated arm once the lift controller actually raises the EE.
3. Sustained frictional hold at the top of the lift is still fragile vs floating-gripper (mocap-pinned hand); the weld latch is an explicit proof-of-mechanism, not a claim that Menagerie contact matches FG reliability.
4. Primary thesis outcome remains the floating-gripper RCT; full-arm is now a completed due-diligence path with logged residual contact limits.

### How to reproduce
```bash
python3 smoke_test_arm_grasp_v4.py cylinder_thin          # friction mid-lift pass
python3 smoke_test_arm_grasp_v4.py --weld cylinder_thin   # sustained weld pass
```
