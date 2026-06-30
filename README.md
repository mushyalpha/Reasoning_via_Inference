# Causal Inference for Robotic Grasp Failure Diagnosis under Perceptual Degradation

MSc Thesis Project — University of Glasgow, 2026

## What This Project Does

Builds a **Structural Causal Model (SCM)** that diagnoses *why* a robotic grasp failed by identifying which environmental perturbation caused the failure, using Pearl's counterfactual reasoning.

## Current Design

| Component | Choice |
|---|---|
| **Simulator** | MuJoCo |
| **Grasp algorithm** | Contact-GraspNet (pre-trained, not retrained) |
| **SCM type** | Linear, fitted with regression |
| **Counterfactual method** | Pearl's 3-step: Abduction → Action → Prediction |
| **Comparison baseline** | LLM/VLM-based failure diagnosis |
| **Scope** | Diagnosis only — recovery actions are out of scope |

## Exogenous Variables (Controlled by Experimenter)

| Symbol | Name | Domain |
|---|---|---|
| σ_d | Depth noise | {0, 0.01, 0.02, 0.04} m — Gaussian noise injected on MuJoCo depth buffer |
| ρ | Point cloud sparsity | {1.0, 0.5, 0.25} — random downsample fraction |
| φ | Viewpoint elevation | {30°, 45°, 60°, 75°} — camera on fixed-radius sphere |
| θ | Viewpoint azimuth | {0°, 60°, 120°} — camera on fixed-radius sphere |

## Endogenous Variables (Measured)

| Symbol | Name |
|---|---|
| C_pc | Point cloud completeness |
| q_grasp | Contact-GraspNet top-1 confidence |
| e_pose | Grasp pose error (Frobenius norm vs oracle) |
| Y | Grasp success (binary) |

## Experimental Grid

4 × 3 × 4 × 3 = 144 unique conditions × 3 seeded noise draws = **432 trials**

## What Is NOT In This Project

- ❌ Isaac Sim (was considered, dropped — MuJoCo chosen for physics fidelity and setup speed)
- ❌ CausalVAE / latent representations (was in early template, replaced with explicit linear SCM)
- ❌ Recovery actions (diagnosis only; recovery is future work)
- ❌ Lighting / shadow variables (MuJoCo depth buffer is geometric; noise injected synthetically)
- ❌ Distribution shift framing (this is controlled perturbation, not domain adaptation)

## File Notes

- `thesis_template.tex` — **ARCHIVED**: describes an older, different project design. See banner at top.
- `project_pipeline.md` — **ARCHIVED**: outdated week-by-week plan with old variables. See banner at top.
- `thesis_clarity_session.md` — **ARCHIVED**: early planning document. See banner at top.
- `build_in_public_strategy.md` — **ARCHIVED**: references old project design. See banner at top.
- `mujoco_test.py`, `visualize_panda.py` — Active: MuJoCo setup tests.
- `mujoco_menagerie/` — Active: Franka Panda robot model assets.
