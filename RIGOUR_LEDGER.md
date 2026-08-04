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
- **Holds?** yes (for the stated scope)
- **Severity** low
- **Disposition** keep-for-MSc; `future-work` complex geometries.
- **Evidence / notes:** Limitations §7.4 items 4, 6.

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
- **Holds?** partial
- **Severity** low–med
- **Disposition** keep-for-MSc; redesign could use deterministic voxel downsampling to remove the second random process.
- **Evidence / notes:** `msc_report.tex:773`-`796`.

## Stage 8 — Viewpoint parameterisation (φ, θ on fixed-radius sphere)

- **Design choice:** Camera on 0.8 m sphere, elevation × azimuth grid.
- **Assumption(s):** Fixed radius ⇒ distance effects constant; grid samples the relevant viewpoint space.
- **Strength:** Decouples viewpoint from distance; small grid is tractable.
- **Weakness / risk:** 3×3 azimuth/elevation grid is coarse; φ=60° turns out to be a pathological regime (98/167 "none" failures), suggesting the grid straddles a regime boundary rather than sampling it.
- **Holds?** partial
- **Severity** med
- **Disposition** redesign-candidate — densify near φ=60° or add intermediate elevations in a future design.
- **Evidence / notes:** Counterfactual ground truth, φ=60° cluster.

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
- **Holds?** yes — threshold sensitivity swept 3 Aug (`success_threshold_sensitivity.py`, D_τ ∈ [0.03, 0.12] m): σ_d ranking is perfectly rank-stable across the *entire* range (σ_d=0.04 pins at 0% for every threshold tested); φ=60° is the worst level at every threshold (dead-zone claim robust); θ=90° is the worst level at every threshold. ρ's ranking is *not* stable across thresholds — but this corroborates, not contradicts, the existing finding that ρ is statistically non-significant in Eq2A (Stage 12), so the instability is a real property of the data, not an artefact of D_τ=0.065.
- **Severity** low (downgraded from med — the headline claim, σ_d dominance, is now shown threshold-robust)
- **Disposition** keep-for-MSc, closed. Add one paragraph + figure (`results/figures/success_threshold_sensitivity.png`) to §5 reporting this check.
- **Evidence / notes:** `msc_report.tex:944`-`972`; Limitations §7.4 item 3; `results/success_threshold_sensitivity.csv`.
- **New issue surfaced:** the calibration text at `msc_report.tex:978` claims ~85% success under clean conditions (σ_d=0, ρ=1.0); the actual full-grid value is **51.9%** (14/27), because that cell includes φ=60° trials (0% success there) dragging the average down. The ~85%/89% figures in the text match σ_d=0 restricted to φ∈{30°,45°} only (`msc_report.tex:1665`), not the literal "σ_d=0, ρ=1.0" cell. **Action:** reword §5 calibration paragraph to state the correct scope (either report 51.9% honestly, or restate the constraint as "≈85% at favourable viewpoints (φ=30°/45°)" — do not leave the current wording, since it is checkable against your own CSV and currently wrong).

## Stage 11 — Experimental grid (432 trials, 3 seeds)

- **Design choice:** 4 σ_d × 4 ρ × 3 φ × 3 θ × 3 seeds = 432.
- **Assumption(s):** 3 seeds gives adequate residual df for OLS; per-condition replication sufficient for the effect sizes of interest.
- **Strength:** Full factorial ⇒ no aliasing among exogenous variables; clean main effects.
- **Weakness / risk:** 3 seeds is the *minimum* for 2-df residual ⇒ wide CIs on per-condition rates; the non-monotone ρ response and viewpoint interactions are underpowered.
- **Holds?** partial
- **Severity** med
- **Disposition** redesign-candidate — increase seeds (esp. for ρ and φ interactions) in any follow-up.
- **Evidence / notes:** Limitations §7.4 item 10; 6 incomplete trials (1.4%).

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

## Counts at last review (updated 3 Aug — Stage 9 & 10 closed)

- Total tracked stages: 19
- `Holds? = yes`: 2 · `partial`: 11 · `open`: 4 · `challenged`: 2
- `Disposition`: keep-for-MSc: 9 · fix-now: 6 · future-work: 3 · redesign-candidate: 5
- `Severity high`: 5 · `med`: 8 · `low`/`low–med`: 6

*Update these counts whenever you edit a row.*
