# Rigour Ledger — Design Choices, Assumptions & Weaknesses

**Purpose.** A single living file that tracks, across *every* stage of the project, what design choice was made, what assumption it carries, the strength and the weakness of that choice, whether the assumption still holds, how severe a breach would be, and the **disposition** — i.e. whether it is fine to leave for an MSc, something to fix now, future work, or a redesign candidate for a later experiment.

It exists so that, as you read more papers or receive criticisms, you can test each new piece of information against a fixed spine instead of re-deriving the rigour of the whole approach each time. When a new source challenges a row, update that row's `Holds?` / `Evidence` and, if needed, promote its `Disposition`. The file doubles as the input to any future experiment redesign: filter for `Disposition = redesign-candidate` and you have the triaged change list.

**How to use it.**
1. When you encounter a new paper / paragraph / criticism, run it against every row: does it strengthen, weaken, or add an assumption? Edit in place.
2. At the end of each working session, scan rows marked `Holds? = open` and re-rank `Severity`.
3. Before any experiment redesign, read only the `redesign-candidate` rows.
4. Keep entries honest — a weakness listed here is a weakness defused, not a fault introduced.

**Legend.**
- **Holds?** `yes` = defended by design or evidence · `partial` = defensible but with caveats · `open` = not yet tested · `challenged` = evidence against
- **Severity** = how much a breach would invalidate the *conclusions* (not the thesis grade): `low` / `med` / `high`
- **Disposition:** `keep-for-MSc` = acceptable scope compromise · `fix-now` = actionable before freeze (27 Jul) · `future-work` = acknowledged, out of scope · `redesign-candidate` = should drive a later experiment redesign

**Quick-scan summary** (counts of dispositions): see end of file.

---

## Stage 1 — Hypothesis & framing

- **Design choice:** Frame grasp-failure diagnosis as a causal-inference problem; compare SCM counterfactual attribution against a zero-shot LLM baseline.
- **Assumption(s):** Failure causes are *diagnosable* from observable intermediates; the SCM is the right formal object for this; LLM baseline is a meaningful comparison point.
- **Strength:** Clean, falsifiable core question; LLM comparison gives a non-trivial bar.
- **Weakness / risk:** "Diagnosis" presupposes a single root cause exists; your own ground truth shows 57.2% of failures have no single-variable fix, so the framing partially mismatches the data.
- **Holds?** partial
- **Severity** high
- **Disposition** redesign-candidate — consider a multi-cause / set-cause framing for the next experiment.
- **Evidence / notes:** `results/counterfactual_groundtruth.csv`; 167/292 "none" cases.

## Stage 2 — Causal variable selection (σ_d, ρ, φ, θ)

- **Design choice:** Four exogenous variables chosen by selection, not by causal discovery.
- **Assumption(s):** These four capture the perceptually-relevant degradation axes; no important cause is omitted.
- **Strength:** All directly controllable ⇒ no confounders, clean factorial design, true exogeneity by construction.
- **Weakness / risk:** Selection was principled-but-informal; omitted variables (lighting irrelevant per MuJoCo, but object mass, friction, gripper force, scene clutter) are out of scope by choice, not by proof.
- **Holds?** partial
- **Severity** med
- **Disposition** keep-for-MSc (scope deliberately narrow); revisit variable set in redesign.
- **Evidence / notes:** CONTEXT.md "What NOT to suggest" §5.

## Stage 3 — Simulator (MuJoCo)

- **Design choice:** MuJoCo, CPU, deterministic.
- **Assumption(s):** Simulator is faithful enough that causal relationships discovered here transfer in *structure* (not necessarily in magnitude) to real systems.
- **Strength:** Deterministic, reproducible, no GPU/dependency variance; you own the ground-truth mechanism.
- **Weakness / risk:** No sim-to-real claim is made (correctly), but the *causal structure* may still differ from reality (e.g., real depth noise is structured, not Gaussian).
- **Holds?** partial
- **Severity** med
- **Disposition** keep-for-MSc; `future-work` real-robot validation already listed in thesis.
- **Evidence / notes:** Limitations §7.4 item 7.

## Stage 4 — Grasp algorithm (Contact-GraspNet)

- **Design choice:** CGN as a fixed black box; no retraining, no architecture change.
- **Assumption(s):** CGN is a representative, sufficiently-capable grasp proposer; its failures under degradation are informative about *perception* failure, not about CGN-specific bugs.
- **Strength:** Pre-trained, 6-DoF, point-cloud-driven; isolates perception→planning pathway.
- **Weakness / risk:** CGN has a systematic ~6 cm lateral bias at φ=30° — a training-distribution artefact folded into the causal model as a "viewpoint effect," which blurs the perception-vs-model distinction.
- **Holds?** partial
- **Severity** med
- **Disposition** keep-for-MSc; document the bias explicitly (already done); redesign could isolate CGN-intrinsic vs perceptual effects.
- **Evidence / notes:** `msc_report.tex:512`; Limitations §7.4 item 5.

## Stage 5 — Robot & scene (Franka Panda, single cylinder, primitive/mesh geometry)

- **Design choice:** Panda + single cylindrical target + Menagerie meshes.
- **Assumption(s):** Robot kinematics identical to real Panda; single object suffices; geometry only enters via occlusion.
- **Strength:** Standard benchmark; controlled single-object scene removes nuisance variation.
- **Weakness / risk:** Single geometry ⇒ no generalisation claim; capsule-vs-mesh history shows geometry choices affect collision behaviour.
- **Holds?** superseded (was `yes (for stated scope)`) — **4 Aug preliminary marking**: both markers independently flagged this as a validity issue, not a scope compromise. Marker A: a rotationally-symmetric cylinder makes evaluation ambiguous (pose errors along the symmetry axis don't matter) and any causal structure extracted is specific to that shape. Marker B: object geometry is a potential *moderator* of the degradation→failure relationship; with one object the thesis cannot distinguish "degradation X causes failure" from "degradation X causes failure *on cylinders*."
- **Severity** high (upgraded from low)
- **Disposition** redesign-candidate → **actioned 4 Aug**: added 2 objects (box: YCB GelatinBox mesh; mustard bottle: YCB MustardBottle mesh, irregular/asymmetric) spanning the shape categories Marker A listed (flat faces/edges, curved/symmetric, irregular/asymmetric). Concavity/handle category (e.g. mug) deliberately not covered — documented gap, candidate 4th object if time allows. See new Stage 20 below for full design record (mesh sourcing, collision-stability fixes, expanded grid, clutter scene).
- **Evidence / notes:** Limitations §7.4 items 4, 6; `object_specs.py`; `RUNPOD_SETUP.md`.

## Stage 6 — Depth rendering & Gaussian noise injection

- **Design choice:** Additive Gaussian noise `d_noisy = d_true + N(0, σ_d²)` on the depth buffer.
- **Assumption(s):** First-order Gaussian approximation captures the perceptually-relevant noise structure.
- **Strength:** Parametrically controllable; single parameter σ_d sweeps severity.
- **Weakness / risk:** Real sensors exhibit structured noise (IR failure near specular surfaces, depth-discontinuity bias, multipath) — all omitted.
- **Holds?** partial
- **Severity** med
- **Disposition** keep-for-MSc; `future-work` structured noise model.
- **Evidence / notes:** Background §6.3; Limitations §7.4 item 2.

## Stage 7 — Point cloud as geometric representation

- **Design choice:** Pinhole back-projection → point cloud → CGN.
- **Assumption(s):** Point cloud is the correct interface between perception and grasp planning (matches CGN's input).
- **Strength:** Native to CGN; preserves 3D geometry; downsampled-cloud formalisation gives a clean ρ handle.
- **Weakness / risk:** Downsampling is *random* — a second stochastic process layered on top of σ_d, potentially confounding sparsity with realised coverage.
- **Holds?** fixed (was `partial`) — **4 Aug**: `sim_common.deterministic_downsample_idx()` replaces `rng.choice` with an exact-fraction, order-preserving strided selection (evenly spaced indices over the existing point order). Verified: exact `floor(n·ρ)` point count and identical output across repeated calls for every (n, ρ) tested. Used in `run_experiments_v2.py` and `run_clutter_experiments.py`.
- **Severity** low (downgraded from low–med — second stochastic process removed, not just documented)
- **Disposition** fix-now, closed 4 Aug. (Original `run_experiments.py` / `experiment_results.csv` untouched, so this only applies to the new v2 grid.)
- **Evidence / notes:** `msc_report.tex:773`-`796`; `sim_common.py::deterministic_downsample_idx`.

## Stage 8 — Viewpoint parameterisation (φ, θ on fixed-radius sphere)

- **Design choice:** Camera on 0.8 m sphere, elevation × azimuth grid.
- **Assumption(s):** Fixed radius ⇒ distance effects constant; grid samples the relevant viewpoint space.
- **Strength:** Decouples viewpoint from distance; small grid is tractable.
- **Weakness / risk:** 3×3 azimuth/elevation grid is coarse; φ=60° turns out to be a pathological regime (98/167 "none" failures), suggesting the grid straddles a regime boundary rather than sampling it.
- **Holds?** superseded (was `partial`) — **4 Aug**: `run_experiments_v2.py` densifies φ to [30, 45, 50, 55, 60, 65] (was [30, 45, 60]), directly testing whether 60° is a hard wall or a steep-but-gradual slope. σ_d is likewise densified to [0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.04] (was [0, 0.005, 0.02, 0.04]) around the 0.005→0.02 collapse region identified in the original fit.
- **Severity** med (unchanged until the new grid's results confirm whether the boundary is sharp or gradual)
- **Disposition** redesign-candidate → **actioned 4 Aug** (grid built; not yet run — see `RUNPOD_SETUP.md`). Re-close this row once `experiment_results_v2.csv` is analysed.
- **Evidence / notes:** Counterfactual ground truth, φ=60° cluster; `run_experiments_v2.py` PHI_VALS/SIGMA_D_VALS.

## Stage 9 — IK controller (DLS Jacobian)

- **Design choice:** Damped Least Squares Jacobian IK to track the CGN pose target.
- **Assumption(s):** IK reliably reaches the requested pose; controller error is negligible relative to perception error.
- **Strength:** Standard, well-derived; closed-form per-timestep.
- **Weakness / risk:** If IK fails or stalls near joint limits, the outcome `Y` is contaminated by *control* failure mis-attributed to *perception*.
- **Holds?** partial (upgraded from open) — indirect check 3 Aug: thresholding the pre-execution `e_pose` (CGN proposal vs. true centroid) at D_τ=0.065 agrees with the logged post-IK `success` on 97.9% of trials with a proposed grasp (279/285); all 6 disagreements are cases where e_pose ≥ D_τ but execution still succeeded (never the reverse). This means IK/execution noise never *destroys* a good proposal — it occasionally rescues a marginal one. Not a per-trial IK-convergence log, but strong indirect evidence that control failure is not contaminating the outcome.
- **Severity** low (downgraded from med)
- **Disposition** keep-for-MSc. A true per-trial IK-convergence flag remains a cheap `future-work` addition if time allows before freeze, but is no longer blocking.
- **Evidence / notes:** `msc_report.tex:852`-`928`; agreement check in `success_threshold_sensitivity.py` session notes, 3 Aug.

## Stage 10 — Success criterion (proximity, EE within D_τ of centroid)

- **Design choice:** Proximity fallback replaces physical lift.
- **Assumption(s):** Proximity is a causally-valid proxy for graspability; threshold D_τ is well-calibrated.
- **Strength:** Stable, repeatable, removes contact-tuning variance; measures exactly what the SCM models (perception-derived pose close enough for contact).
- **Weakness / risk:** A grasp that is *close* but not *successful* overstates success; conversely a good grasp with a bad proximity score understates it.
- **Holds?** superseded (was `yes`) — **4 Aug preliminary marking, second round**: both markers independently flagged this as invalid, this time targeting the metric itself rather than the threshold value. Marker A: name what X actually measures and use a physically-grounded alternative (floating-gripper shake test, the ACRONYM/6-DOF-GraspNet standard) as the main outcome. Marker B, more sharply: `success = 1{e_pose < D_τ}` is a deterministic function of `e_pose`, which is *itself a mediator already in the SCM* (`exogenous → C_pc, q_grasp, e_pose, n_grasps → success`) — so the outcome node carries no information beyond `e_pose`, `q_grasp`/`n_grasps` become decorative, and any later "diagnose which cause led to failure" step will trivially recover `e_pose` because the outcome was *built from* a variable the SCM fitted. Also flagged: D_τ was moved from 4cm (~real gripper tolerance, per this project's own notes) to 6.5cm explicitly to produce a target success-rate distribution, not derived from physics. All three points are correct and the threshold-sensitivity robustness check above (still true on its own terms) does not answer any of them — a threshold-robust *invalid* metric is still invalid.
- **Severity** high (upgraded from low — this is a validity issue in the outcome variable itself, not a calibration nicety)
- **Disposition** fix-now, actioned 4 Aug → see new **Stage 21**, which replaces this success criterion outright with the floating-gripper shake test for all Experiment A/B trials going forward. This stage is kept (not deleted) as the historical record for the existing 432-trial dataset and the threshold-sensitivity check, which remain valid *as an analysis of that specific (now-superseded) metric* — they should not be presented as validating "grasp success" in the redesigned thesis.
- **Evidence / notes:** `msc_report.tex:944`-`972`; Limitations §7.4 item 3; `results/success_threshold_sensitivity.csv`. Marker feedback quoted verbatim in `MARKER_FEEDBACK.md`.
- **Earlier issue (3 Aug, still relevant to the historical dataset only):** the calibration text at `msc_report.tex:978` claims ~85% success under clean conditions (σ_d=0, ρ=1.0); the actual full-grid value is **51.9%** (14/27), because that cell includes φ=60° trials (0% success there) dragging the average down. If the historical proximity-based results are retained anywhere in the writeup (e.g. as a "what we tried first" appendix), this wording still needs correcting; the redesigned Stage 21 metric makes it moot for the headline results.

## Stage 11 — Experimental grid (432 trials, 3 seeds)

- **Design choice:** 4 σ_d × 4 ρ × 3 φ × 3 θ × 3 seeds = 432.
- **Assumption(s):** 3 seeds gives adequate residual df for OLS; per-condition replication sufficient for the effect sizes of interest.
- **Strength:** Full factorial ⇒ no aliasing among exogenous variables; clean main effects.
- **Weakness / risk:** 3 seeds is the *minimum* for 2-df residual ⇒ wide CIs on per-condition rates; the non-monotone ρ response and viewpoint interactions are underpowered.
- **Holds?** superseded (was `partial`) — **4 Aug**: `run_experiments_v2.py` uses 5 seeds/condition (was 3) on top of the densified grid, giving 2520 trials/object × 3 objects = 7560 for Experiment A. A separate 504-trial targeted grid (`run_clutter_experiments.py`) covers Experiment B (clutter). Report submission deadline extended (see Stage 20), removing the original time pressure that motivated the 3-seed minimum.
- **Severity** med (unchanged until run)
- **Disposition** redesign-candidate → **actioned 4 Aug** (not yet run — see `RUNPOD_SETUP.md`; a `--lean` 3-seed fallback grid exists if the full run doesn't finish in time).
- **Evidence / notes:** Limitations §7.4 item 10; 6 incomplete trials (1.4%); `run_experiments_v2.py` N_REPEATS.

## Stage 12 — SCM structural equations (linear + logistic functional form)

- **Design choice:** Linear-in-parameters for `C_pc, q, n, e`; logistic for `Y`.
- **Assumption(s):** Relationships are approximately monotone and linear over the grid range; logistic link is correct for the binary outcome.
- **Strength:** Interpretable coefficients; cheap to fit; OLS has closed form.
- **Weakness / risk:** Functional-form assumptions are "costly" (per the causality literature) — the 6 cm φ=30° bias and the σ_d→n_grasps collapse are non-linear effects forced into linear terms.
- **Holds?** partial
- **Severity** high
- **Disposition** redesign-candidate — non-parametric / neural structural equations in a follow-up; `future-work` already listed.
- **Evidence / notes:** `msc_report.tex:474`-`522`; Future Work §7.5.

## Stage 13 — Causal DAG (expert-specified)

- **Design choice:** DAG specified from domain knowledge, not discovered.
- **Assumption(s):** No omitted direct effects among modelled variables; no latent common causes.
- **Strength:** Grounded in mechanism; factorial design validates L2 predictions.
- **Weakness / risk:** If the true structure differs (e.g., the σ_d→C_pc / ρ→C_pc edges that the empirical audit showed should be removed — CONTEXT.md open issue #5), counterfactuals are wrong.
- **Holds?** challenged
- **Severity** high
- **Disposition** fix-now — correct the DAG in `msc_report.tex` Ch.3 (C_pc computed before noise/downsample, so those edges are spurious).
- **Evidence / notes:** CONTEXT.md open issue #5; Yang & Bareinboim hierarchy ⇒ L3 assumptions unfalsifiable from L1/L2 data.

## Stage 14 — SCM fitting (OLS + MLE logistic)

- **Design choice:** OLS for continuous equations, MLE logistic for `Y`; single train split.
- **Assumption(s):** Residuals well-behaved; no overfitting given the small grid; fit generalises to held-out conditions.
- **Strength:** Standard, reproducible, no NN training.
- **Weakness / risk:** Fit quality on the training split ≠ counterfactual accuracy on held-out trials; no held-out L2 interventional check has been run.
- **Holds?** open
- **Severity** high
- **Disposition** fix-now — run a held-out interventional check: predict `P(Y|do(σ_d=s))` from the fitted SCM and compare to re-simulated empirical rates. This is the single highest-value rigour test available because you own the simulator.
- **Evidence / notes:** R²/AUC values in CONTEXT.md "SCM fitted (9 July)"; surrogate-model concern (Stage 16).

## Stage 15 — Exogenous error independence (NPSEM-ie)

- **Design choice:** Treat the SCM as Markovian (independent errors) to identify the ETT query.
- **Assumption(s):** `U_C, U_q, U_n, U_e` are mutually independent.
- **Strength:** Gives identifiability of L3 counterfactuals; abduction becomes per-equation and unique.
- **Weakness / risk:** Independence is *assumed*, not tested; the modern causality literature (Richardson & Robins) treats NPSEM-ie as optional and contested, yet the thesis adopts it without naming it.
- **Holds?** open
- **Severity** med
- **Disposition** fix-now — (a) name the NPSEM-ie assumption explicitly at `msc_report.tex:1009`; (b) add a one-sentence design-based defence (the `U` terms arise from mechanistically distinct sources: pixel coverage geometry, CGN scoring stochasticity, network-head pose regression).
- **Evidence / notes:** `msc_report.tex:1009`; previous dissection points 3–5.

## Stage 16 — Surrogate-model status of the fitted SCM

- **Design choice:** Use the fitted linear-logistic SCM *as* the model of the system.
- **Assumption(s):** The fitted SCM has enough expressive power for the queries being asked of it (L3).
- **Strength:** Tractable, interpretable.
- **Weakness / risk:** The *true* mechanism is CGN + MuJoCo + IK + proximity — opaque and unattainable in closed form. The fitted SCM is a surrogate; the literature only promises surrogates are safe at L1/L2, but the thesis pushes the surrogate to L3, where mis-specification is amplified by abduction.
- **Holds?** open
- **Severity** high
- **Disposition** fix-now — reframe §4.10 explicitly as a *surrogate* (cite Rubenstein et al.); run the L2 interventional check from Stage 14 to bound the surrogate's L2 error before trusting L3.
- **Evidence / notes:** previous dissection point 7.

## Stage 17 — Counterfactual diagnosis (single-cause, argmax Δ)

- **Design choice:** Attribute failure to `argmax_v Δ_v` over the four exogenous variables.
- **Assumption(s):** The largest counterfactual improvement identifies the root cause; single-cause attribution is the right target.
- **Strength:** Simple, deterministic, reproducible; gives a ranked attribution.
- **Weakness / risk:** argmax is "sufficient but not minimal" (Halpern–Pearl actual causation adds minimality/normality); 57.2% of failures have no single-variable fix ⇒ the target is partly ill-posed for the data.
- **Holds?** partial
- **Severity** high
- **Disposition** redesign-candidate — support set-cause / multi-cause attribution; `future-work` Halpern–Pearl minimality.
- **Evidence / notes:** `msc_report.tex:1011`, `1604`, `1610`.

## Stage 18 — LLM baseline

- **Design choice:** Zero-shot VLM, fixed prompt, fixed temperature, rubric defined before trials.
- **Assumption(s):** Prompt is information-leakage-free; rubric mapping is fair; temperature controls stochasticity adequately.
- **Strength:** No task-specific training; represents state of practice; pre-registered rubric prevents post-hoc bias.
- **Weakness / risk:** VLM may infer degraded conditions from observable intermediates in ways that shortcut the comparison; consistency across 3 repeats may be too few to estimate LLM variance well.
- **Holds?** open
- **Severity** med
- **Disposition** fix-now — ensure the prompt excludes exogenous ground-truth values (already planned, CONTEXT.md open issue #3); include a "none/joint" rubric category.
- **Evidence / notes:** `msc_report.tex:1013`-`1024`; CONTEXT.md in-progress §LLM baseline.

## Stage 19 — Attribution vs actual causation

- **Design choice:** Use argmax-Δ attribution; do not check Halpern–Pearl minimality/normality.
- **Assumption(s):** Sufficiency is enough for the diagnostic claim.
- **Strength:** Scope-appropriate; minimality is expensive to verify.
- **Weakness / risk:** A "cause" that is sufficient but not minimal can mis-rank when multiple variables each suffice.
- **Holds?** partial
- **Severity** low–med
- **Disposition** future-work (already listed); no MSc-scope action.
- **Evidence / notes:** `msc_report.tex:1011`, `1610`.

## Stage 20 — Multi-object redesign & clutter experiment (preliminary marking response, 4 Aug)

- **Design choice:** Add 2 objects (box: YCB GelatinBox; mustard bottle: YCB MustardBottle, irregular/asymmetric) alongside the existing cylinder — see Stage 5 — plus a new Experiment B: all 3 objects together in a fixed triangular clutter arrangement (`object_specs.clutter_layout`, 0.11 m spacing), with a designated grasp target rotating across the 3 objects and a new outcome variable `collision_with_neighbor` (gripper contacts a non-target body during approach/descend/close, read live from MuJoCo's contact array).
- **Assumption(s):** (a) Real YCB visual meshes + simplified collision primitives give perceptually-realistic depth/point-cloud input without requiring physically-accurate collision on the target object (justified because the success criterion is proximity-based, not a real lift — see Stage 10). (b) A single fixed clutter arrangement (not randomised per trial) is sufficient to test the occlusion/collision mechanism Marker B describes. (c) Object mass/friction kept in the same light/high-friction regime as the original cylinder (not real YCB inertial values) for physics-stability consistency with the already-validated single-object pipeline.
- **Strength:** Directly answers both markers' core ask (geometry as moderator; scene complexity as a second causal regime) with real YCB geometry (Marker A's "zero justification burden"), reusing the *same* validated single-object pipeline pattern (visual mesh / collision primitive split) already established for the Panda arm. Object-agnostic code (`sim_common.py`) generalizes qpos/joint indexing by NAME rather than fixed slices, so it works for both 1-freejoint (isolated) and 3-freejoint (clutter) scenes without duplicated logic.
- **Weakness / risk:** (i) Concavity/handle category (e.g. a mug) — explicitly requested by Marker A — is not covered; the 3rd object is asymmetric/irregular (mustard bottle) instead, per an explicit scope trade-off made this session. (ii) The clutter arrangement is fixed (not randomised across trials), so Experiment B cannot separate "this particular arrangement" from "clutter in general" — a single clutter geometry, same class of limitation as Stage 5 originally had for single-object geometry. (iii) Two implementation bugs were found and fixed during this session that are worth tracking as risk items in their own right: the mustard bottle's merged V-HACD collision hull did not yield a stable resting pose (switched to a bounding-box collision primitive); the arm's fixed home pose overlapped the taller mustard bottle (arm now parks at a fixed clear Cartesian position before settling). (iv) GRASP_RADIUS (0.065 m) is not re-tuned per object — same threshold used for cylinder, box, and mustard, even though their footprints differ (0.036 m vs ~0.036 m vs ~0.029 m half-widths) — a documented simplification, not re-derived from Stage 10's threshold-sensitivity analysis for the new objects. (v) The 3 objects were chosen to have comparable footprint radii to the cylinder specifically so grasp-width feasibility stays roughly constant — this was a deliberate design constraint, not a property of "the box" / "the mustard bottle" categories in general.
- **Holds?** open — code built and physics/determinism-validated locally (no GL rendering available in the local sandbox); the actual trial batches have not been run. See `RUNPOD_SETUP.md`.
- **Severity** high (this redesign is the direct response to preliminary marking; until the batches run and are analysed, the thesis's core single-geometry limitation is only partially addressed)
- **Disposition** fix-now, in progress. Next actions: (1) run `run_experiments_v2.py` and `run_clutter_experiments.py` on RunPod (GPU); (2) refit the SCM with object identity as a moderator (interaction terms, or a stratified fit per object) to test Marker B's "consistent structure vs. interacts with geometry" question; (3) analyse `collision_with_neighbor` as a function of σ_d/ρ in the clutter data; (4) update the thesis DAG, methods chapter, and limitations section accordingly; (5) consider a 4th object (concavity/handle) if time remains, to close gap (i) above.
- **Evidence / notes:** `object_specs.py`, `sim_common.py`, `run_experiments_v2.py`, `run_clutter_experiments.py`, `RUNPOD_SETUP.md`, `assets/ycb/README.md`. Marker feedback quoted verbatim in this session's conversation (not yet a separate file — consider saving the raw feedback text somewhere durable, e.g. a `MARKER_FEEDBACK.md`, so the response can be checked against the original wording later).

## Stage 21 — Outcome-variable redesign: floating-gripper shake test (preliminary marking response, round 2, 4 Aug)

- **Design choice:** Replace `success = 1{e_pose < D_τ}` (Stage 10) with a physically-grounded floating-gripper shake test for every trial in Experiment A and B: teleport a free Panda gripper (no arm, no IK) directly to CGN's predicted 6-DoF pose (full orientation, not just position — the earlier pipeline dropped orientation), check the pre-grasp pose is collision-free, close the fingers, then apply a lift + shake disturbance under gravity. `success = 1` iff the object stays within `footprint_radius + 3cm` of the gripper's ee-site *and* reaches ≥40% of the intended 15cm lift, at every sampled step of the shake (not just the final frame). This is independent of `e_pose`/`q_grasp`/`n_grasps` (all still logged as mediators, but none of them mechanically determines the new outcome), directly answering Marker B's circularity objection, and matches the ACRONYM / 6-DOF-GraspNet evaluation protocol Marker A named.
- **Assumption(s):** (a) A momentumless, gravity-compensated free-floating gripper (no arm dynamics) is an acceptable idealisation of "the arm holds the end-effector rigidly at the commanded pose" — standard in the grasp-evaluation literature this design is copying. (b) A single fixed shake trajectory (deterministic, no RNG — 4Hz/5.3Hz sinusoidal xy jitter plus a ramped 15cm lift) is a sufficient disturbance to distinguish a secure grasp from a marginal one, without being so aggressive that it fails good grasps. (c) Realistic actuator/friction parameters (Franka Hand's real ~70N/finger rated force; explicit high-friction 5.0 rubberised-pad coefficient) are a defensible physical basis, not tuned to hit a target success rate (directly answering Marker B's third point about the old D_τ).
- **Strength:** Outcome is now genuinely physical (an object either stays gripped through a disturbance or it doesn't) and structurally independent of the SCM's own mediators — the SCM-vs-LLM comparison this metric feeds into can no longer be accused of being rigged in the SCM's favour by construction. Validated end-to-end: (i) a standalone smoke test (`smoke_test_floating_gripper.py`) confirms a known-good side-grasp pose on the cylinder succeeds (full 15cm lift), a known-empty pose 30cm away fails, and a pose planted through the object's centroid is correctly flagged as colliding before the shake even starts; (ii) a full `run_experiments_v2.py --test` pass (24 trials, real CGN-proposed poses, all 3 objects) produced genuine `SUCCESS` rows on both the cylinder and the mustard bottle with realistic lift heights (z rising from ~0.45-0.48m resting to ~0.60-0.63m, i.e. the full commanded 15cm), confirming the mechanism works with CGN's own proposals, not just hand-picked poses.
- **Weakness / risk:** (i) **A major implementation bug was found and fixed mid-session, worth recording because it nearly shipped silently**: the first implementation drove the gripper's `hand` body as a MuJoCo *mocap* body (teleported directly via `data.mocap_pos` each step, as is standard for "kinematically-held" end-effectors). This is a genuine MuJoCo semantics trap for this exact use case — mocap bodies contribute no rows to the contact Jacobian, so the constraint solver has no notion that a mocap body is *moving*; teleporting it changes where contacts are detected but not the relative velocity the friction constraint reacts to. Net effect, confirmed by direct instrumentation (per-step object height + raw contact-force tracing): an object squeezed by a teleported mocap hand never actually got dragged upward by friction, however large the squeeze force — it just sat on the table while the fingers slid past it (object height frozen to the mm for seconds of simulated lift time). This would have produced a metric that always returned `success=0` for essentially every real grasp, i.e. a silently-broken (not just noisy) outcome variable. **Fix:** `hand` is now a real dynamic free body (freejoint, `gravcomp="1"`) welded via a MuJoCo equality constraint to a separate invisible mocap target, which *is* teleported as before — the weld's tracking force is now resolved by the same Newton/constraint solver that handles contacts, so momentum genuinely couples through friction to a gripped object. (An intermediate attempt at a hand-rolled PD force via `xfrc_applied` was also tried and rejected — it is not integrated into the implicit solver the way equality constraints are, and diverged numerically (NaN qacc) the moment the finger-closing reaction force hit the gripper's small 0.73kg mass.) (ii) Object-agnostic validation is incomplete: the box (thin YCB GelatinBox, ~2.8cm short axis) showed 0/8 successes in the `--test` smoke run, split roughly evenly between real CGN collision failures and collision-free-but-no-lift outcomes. This may be a genuine geometry effect (thin flat objects are a known hard case for parallel-jaw grippers, and would itself be a relevant "geometry as moderator" finding for the thesis) rather than a residual test-mechanism bug, but it has not been root-caused with the same rigour as the mocap-vs-weld issue above, given time constraints this session — flagged as the top follow-up item. (iii) Shake trajectory parameters (amplitude, frequency, lift height, 40%-of-lift pass threshold) are fixed constants chosen once and not swept/justified via a sensitivity analysis analogous to Stage 10's D_τ sweep — a fast-follow, not a blocker, but should be done before quoting exact success-rate numbers in the final report. (iv) `forcerange` on the finger actuator was capped at ±70N/finger to match the real Franka Hand's rated grasp force and to keep the new dynamic-hand simulation numerically stable — this is *more* physically grounded than the old scene's arbitrary large values, but is a new parameter that wasn't present in the Stage 10-era pipeline and should be named in the methods section.
- **Holds?** partial — mechanism validated (mocap/weld bug fixed and confirmed via direct force/position tracing, not just outcome inspection; smoke tests pass; real CGN pipeline produces genuine successes for 2/3 objects). Not yet holds `yes` because (ii) above (box) is unresolved and the full-grid batches (thousands of trials) have not been run.
- **Severity** high (this is the metric the entire SCM-vs-LLM comparison depends on; getting it right is a precondition for every downstream causal claim)
- **Disposition** fix-now, in progress. Next actions: (1) root-cause the box's low success rate the same way the mocap bug was root-caused (contact-force/position tracing, not just outcome inspection) before assuming it is a genuine geometry effect; (2) run the full (non-`--test`) `run_experiments_v2.py` and `run_clutter_experiments.py` batches on RunPod now that the outcome mechanism is validated; (3) add a brief shake-trajectory sensitivity check analogous to Stage 10's `success_threshold_sensitivity.py`, time permitting; (4) update the thesis methods chapter to describe the floating-gripper protocol (replacing the D_τ description) and add a short "outcome variable was redesigned in response to examiner feedback" note for transparency in the viva.
- **Evidence / notes:** `floating_gripper_template.xml` (docstring records the mocap→weld fix in detail), `sim_common.py` (`teleport_hand_hard`, `teleport_mocap`, `run_floating_gripper_test`, `gripper_contacted_bodies`, `open_gripper`), `smoke_test_floating_gripper.py`, `run_experiments_v2.py` / `run_clutter_experiments.py` (`execute_grasp_floating` / `execute_grasp_clutter_floating`), `results/experiment_results_v2.csv` (24-trial `--test` run, 4 Aug). `MARKER_FEEDBACK.md` for the verbatim marker quotes that motivated this stage.

---

## Redesign-candidate shortlist (for a future experiment)

Filtering `Disposition = redesign-candidate`:

1. **Stage 1 & 17 — Multi-cause framing.** Adopt set-cause / joint attribution; the single-cause target mismatches 57.2% of the data. *(high severity)*
2. **Stage 8 — Viewpoint grid.** Densify around φ=60°, which sits on a pathological regime boundary. *(med)*
3. **Stage 11 — Seeds.** Increase per-condition replication to narrow CIs on ρ and φ-interaction effects. *(med)*
4. **Stage 12 — Non-parametric SCM.** Replace linear structural equations with a flexible (neural) parameterisation to remove the functional-form cost. *(high)*
5. **Stage 2 — Variable set.** Revisit whether the four chosen variables are the right axes for a redesigned study (e.g., add a contact-mechanics pathway). *(med)*

## Fix-now shortlist (before experiment freeze, 27 Jul)

1. **Stage 13 — Correct the DAG** in `msc_report.tex` Ch.3 (remove σ_d→C_pc, ρ→C_pc edges). *(high)*
2. **Stage 14 — Held-out L2 interventional check** of the fitted SCM vs re-simulated rates. *(high; highest-value test available)*
3. **Stage 15 — Name and defend the NPSEM-ie assumption** at `msc_report.tex:1009`. *(med)*
4. **Stage 16 — Reframe the fitted SCM as a surrogate** in §4.10, citing Rubenstein et al. *(high)*
5. ~~Stage 9 — Log IK reach success per trial~~ — **closed 3 Aug** via indirect agreement check (97.9% e_pose/success agreement, all disagreements non-harmful direction).
6. ~~Stage 10 — Threshold sensitivity sweep~~ — **closed 3 Aug**, see `success_threshold_sensitivity.py`; σ_d ranking is threshold-robust. Surfaced a new item: **fix the 85% clean-condition calibration claim at `msc_report.tex:978`** — actual value is 51.9% for the literal (σ_d=0, ρ=1.0) cell; the 85%/89% figures apply only when restricted to φ∈{30°,45°}. *(med — checkable inconsistency, fix before freeze)*
7. **Stage 18 — Finalise LLM prompt/rubric** with "none/joint" category, before any LLM trials. *(med)*

## Counts at last review (updated 4 Aug, second pass — outcome-variable redesign: Stage 10 superseded, new Stage 21 added)

- Total tracked stages: 21
- `Holds? = yes`: 0 · `partial`: 9 · `open`: 5 · `challenged`: 2 · `fixed`: 1 · `superseded`: 4
- `Disposition`: keep-for-MSc: 6 · fix-now: 9 (2 closed: Stage 7, and Stage 9/10-threshold-sweep from 3 Aug) · future-work: 3 · redesign-candidate: 5 (3 newly actioned: Stages 5, 8, 11 — code built, not yet run)
- `Severity high`: 8 · `med`: 8 · `low`/`low–med`: 5

*Update these counts whenever you edit a row.*
