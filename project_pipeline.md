# MSc Project Pipeline — Living Document
## Causal Inference for Robotic Grasp Failure Diagnosis under Perceptual Degradation

**Project start:** 25 May 2026
**Preliminary report submitted:** 19 June 2026
**Experiment freeze:** 27 July 2026
**Report submission:** 14 August 2026
**Poster presentation:** 19 August 2026

This is a living document. Completed entries log what actually happened, including decisions made and questions that arose. Future entries are milestones only. The thinking process is tracked alongside the implementation process.

---

## Week 1 — May 25 to May 31

**25 May — First supervisor meeting (Dezong Zhao)**

- Introduced to the project area: causal inference for robotic reasoning
- Directed to read Dezong's published work, in particular Causal DiffuseVAE and related causal generative modelling
- Told to think about what "causal" means in a robotics context beyond image generation
- Project registered and timeline established

---

## Week 2 — June 1 to June 7

**Meeting with Jingzhi Ruan (PhD student)**

Key clarifications from this meeting:
- The robot must perceive through a camera, not use internal simulator state. This is the fundamental requirement of a perception-driven grasping pipeline.
- Contact-GraspNet identified as the right algorithm class: pre-trained, 6-DoF, point-cloud-driven
- Jingzhi exposed a blind spot: no clear algorithm had been chosen before this meeting

Critical question surfaced: How does the robot grasp autonomously? I had a simulator in mind but no grasp algorithm. This meeting forced the distinction between "the simulator knows where the object is" and "the robot must see the object and decide." That distinction turned out to be the core of the thesis.

---

## Week 3 — June 8 to June 14

**Second supervisor meeting (Dezong Zhao)**

- Discussed project direction after Jingzhi meeting
- Confirmed perception-driven pipeline as the right approach
- Began clarifying what the causal model would look like

**Simulator decision: MuJoCo vs PyBullet vs Isaac Sim**

Initial instinct was Isaac Sim on the basis that it is photorealistic and Contact-GraspNet was assumed to be well-integrated. After stress-testing that assumption:

- Isaac Sim is not a pip install. It requires Omniverse Launcher, Nucleus server setup, driver compatibility checks, and shader cache compilation. University HPC access can take days to weeks to approve.
- Contact-GraspNet has no native Isaac Sim integration. The integration advantage was a false assumption from early planning. The bridging work is identical to MuJoCo.
- MuJoCo: deterministic, fast, well-validated for manipulation research, CPU-runnable on a laptop, actively maintained.
- PyBullet: older, less actively maintained, worse contact dynamics for this class of problems.

Decision: MuJoCo.

Critical question: Does using MuJoCo and synthetic depth noise instead of Isaac Sim's photorealistic rendering make the project weaker?

Conclusion: No. The thesis is a causal inference study, not a perception fidelity study. The strength of a causal study is identifiability, not ecological validity. MuJoCo's determinism means the simulation is a randomised controlled trial by construction. A photorealistic renderer would make the data look more realistic but would introduce confounders into the causal structure that would require adjustment. Synthetic Gaussian noise injected directly onto the depth buffer gives a precisely known, independently controllable causal variable. That is what the project needs.

**Algorithm decision: Contact-GraspNet**

Three candidates evaluated:
- GraspNet-baseline: standard implementation, well-documented
- Contact-GraspNet: NVIDIA's version, open source, strongest community
- AnyGrasp: newer, SDK-gated, licence friction

Decision: Contact-GraspNet.

---

## Week 4 — June 15 to June 21

**Working on preliminary report**

- Drafting project objectives, background, Gantt chart, references
- Preliminary report due Thursday 19 June at noon

**Variable redesign**

Original plan used lighting intensity, lighting direction, and camera distance as exogenous variables. Problem identified: MuJoCo's depth buffer is purely geometric. It returns exact Z-distance unaffected by illumination. Lighting is not a controllable variable in this simulator. Redesign adopted:

| Symbol | Variable | Domain |
|---|---|---|
| sigma_d | Depth noise (Gaussian, injected on depth buffer post-render) | 0, 0.01, 0.02, 0.04 m |
| rho | Point cloud sparsity (random downsample fraction) | 1.0, 0.5, 0.25 |
| phi | Viewpoint elevation (camera on fixed-radius sphere) | 30, 45, 60, 75 degrees |
| theta | Viewpoint azimuth | 0, 60, 120 degrees |

Experimental grid: 4 x 3 x 4 x 3 = 144 unique conditions x 3 seeded noise draws = 432 trials. Fully exhaustive, no sampling strategy required. At roughly 2 seconds per trial in MuJoCo headless mode, the full dataset runs in under 15 minutes.

Physical constraints (mass, friction, gripper force) were considered as a second physics pathway in the SCM. Decision: removed for now. Perception pathway alone is sufficient for a focused MSc thesis. The physics pathway is a clean future extension.

Critical question noted: An implementation plan suggested heuristically computing a grasp from the object centroid and calling it simulating Contact-GraspNet. Is this acceptable?

No. If the grasp proposal does not consume the actual degraded point cloud, the causal chain "degraded point cloud to worse grasp proposal to failure" does not exist. The thesis claims this chain. Contact-GraspNet must actually run on the degraded point cloud.

**19 June — Preliminary report submitted**

Submitted before noon deadline. Contents: project objectives, background on causal inference and robotic grasping, Gantt chart, resources, initial references.

**20 June — Research journal: Doubts, framing, and aligning on deliverables**

*Taking time to step back and honestly evaluate the project's direction and my own commitment. A few critical struggles and turning points emerged:*

1. **Commitment and Rigor:**
   Realized that giving 50% only yields 50% of the results. To achieve a project that is potentially publishable and has a clear unique contribution, I need to commit 100% and ruthlessly optimize for the final goal. No more aiming for the bare minimum. Every step needs to be stress-tested for rigor.

2. **Doubt over "Methodology looking for a problem":**
   Struggled with whether applying causal inference to robotic failure is genuinely useful or just a forced intersection.
   *Resolution:* The strength of the project lies in the concept of **trustworthy autonomy**. Grasping isn't just the task; it's a mechanism that can be disrupted through the perception channel. We need explainable methods to reason about *why* a robot failed so we can trust it.

3. **Confusion over deliverables (Visuals vs. Control):**
   Questioned the earlier desire to use Isaac Sim. Admitted that Isaac Sim was appealing because the pipeline "looks really cool on the report."
   *Resolution:* Since the core task is feeding point clouds to Contact-GraspNet and building a causal model over structured variables, **controllable variation** is far more critical than visual realism. This cements the decision to use MuJoCo.

4. **Clarity on training and causal factors:**
   Grappled with confusion over whether I am training a new model and how to extract causal factors. 
   *Resolution:* The project is not about training a novel grasping model from scratch. It is about using a pre-trained model (Contact-GraspNet) and building a structural causal model on top of its outputs and controllable inputs to reason about failures.

*Central research question, crystallized from this session:* Can causal counterfactual reasoning identify the root cause of grasp failures more reliably than an LLM?

---

## Week 5 — June 22 to June 28

**22 June — MuJoCo environment setup**

- MuJoCo installed and tested (mujoco_test.py, falling box scene confirms physics running)
- Franka Panda arm loaded from MuJoCo Menagerie (visualize_panda.py, arm visible and joint-controllable)
- Contact-GraspNet downloaded (not yet integrated)
- Project documentation files updated to reflect current design
- ✅ **Completed**: Added table surface, graspable object (generic cylinder), and perception camera to the MuJoCo scene (`grasp_scene.xml`, `grasp_simulation.py`). The camera is parameterized on a fixed-radius sphere using $\phi$ (elevation) and $\theta$ (azimuth). A visual marker (green box) was added to the camera body so its position is visible in the viewer.

**Critical questions surfaced during scene setup:**
1. *Does using generic shapes instead of real household items weaken the project?*
   *Resolution*: No. The thesis is a causal inference study about perception degradation, not complex geometries. A failure caused by a noisy depth camera looks the same whether grasping a generic cylinder or a complex mug. In fact, simple objects remove physical confounders. However, to make the final report look professional, a standard YCB object (like a mustard bottle) should be swapped in later.
2. *Does the camera parameterization reflect the real world?*
   *Resolution*: The camera setup (moving on a perfect sphere) is actually *better* than the real world for a causal study. Real-world camera movement introduces confounders like changing background clutter and lighting angles. The simulation isolates the viewpoint variable cleanly. The 0.8m radius is realistic for a head-mounted or tripod-mounted robot camera.

**22 June — Depth capture and point cloud pipeline**

- Rewrote `grasp_simulation.py` with full documentation of the pinhole camera model and each pipeline step
- Implemented depth rendering using `mujoco.Renderer` with `enable_depth_rendering()`
- Implemented Gaussian noise injection on the depth buffer (`sigma_d` variable)
- Implemented standard pinhole back-projection to 3D point cloud using camera intrinsics derived from MuJoCo's `cam_fovy`
- Implemented random downsampling by fraction `rho`
- Implemented camera-to-world transform using `data.cam_xmat` and `data.cam_xpos`, with correct Z-flip for MuJoCo's camera convention
- Verified: clean point cloud centroid ~(0.39, -0.09, 0.35) vs expected target at (0.50, 0.00, 0.45) — offset expected because the camera sees the whole scene, not just the object

**23 June — Understanding the MuJoCo camera model**

Conflict: The camera in MuJoCo appears to work very differently from a real camera. I had expected a physical camera on a stabiliser looking at the scene, the way a robot vision system might be set up in a warehouse. In MuJoCo, the camera has no physical body — you cannot see it in the viewer, and it does not exist as an object in the simulation. This raised the question of whether the setup was rigorous enough for a thesis.

Resolution: The MuJoCo virtual camera is mathematically identical to a real RGB-D sensor. In MuJoCo, a camera is a virtual viewpoint defined by a position (x, y, z), an orientation (quaternion), and a vertical field-of-view angle. The renderer projects the scene from that viewpoint and returns a depth buffer — a 2D array where each pixel stores the perpendicular distance, in metres, from the camera to the nearest surface. This is exactly what a RealSense D435 or Azure Kinect returns.

The difference between the virtual camera and a physical one is not mathematical but mechanical: a real camera has vibration jitter, lens distortion, thermal drift, and ambient illumination effects. For a causal inference study, these are confounders. The virtual camera eliminates them by construction, meaning elevation (phi) and azimuth (theta) can be varied independently without introducing any side-effects. This is not a weakness of the simulation — it is the correct experimental design. The one property a real sensor has that the virtual camera lacks is measurement noise, which is addressed by injecting Gaussian noise with standard deviation sigma_d directly onto the depth buffer.

Written up as a subsection in `thesis_direction_example.tex` (Section: Camera Model), including the back-projection equation, reference placeholders for RealSense, Kinect, and Pearl 2009.

---

## Week 6 — June 29 to July 5

**30 June — Contact-GraspNet integration and full pipeline**

Major implementation day. Moved from Stage 1 (perception only) to a working end-to-end grasp pipeline and first full dataset collection.

**Contact-GraspNet integration (Stage 2)**
- Integrated `contact_graspnet_pytorch` (PyTorch port of NVlabs Contact-GraspNet) as a local submodule with pre-trained checkpoint loaded on CPU
- Built `mujoco_cgn_bridge.py`: MuJoCo depth + segmentation → CGN-compatible `.npz` → inference → world-frame grasp pose
- CGN runs with `local_regions=True` and `filter_grasps=True` using a segmentation map of the target object body
- Frame transform implemented: camera frame → world frame via `cam_xmat`/`cam_xpos` with MuJoCo Z-flip correction

**Scene rebuild (`grasp_scene_v2.xml`)**
- Rebuilt Panda scene using Menagerie kinematics (capsule geometry, no mesh dependencies)
- Added proper actuators, finger tendons, home keyframe, and `ee_site` for IK
- Target object: free-joint cylinder (radius 0.036 m, mass 0.03 kg, high friction) at (0.5, 0, 0.455)
- Also created `simple_grasp_scene.xml` for early CGN-only testing via the bridge script

**End-to-end grasp execution**
- Built `demo_grasp.py`: visual pipeline in MuJoCo viewer — camera capture → CGN inference → pre-grasp → descend → close gripper → lift → record outcome
- Built `run_experiments.py`: headless 432-trial batch runner with CSV logging to `results/experiment_results.csv`
- Implemented DLS Jacobian IK (`mj_jacSite`, arm columns `nv[6:13]`, joint updates in `qpos[7:14]`)
- Gripper control via tendon actuator `ctrl[7]` (255 = open, 0 = closed)

**Critical bug fixed: IK writing to wrong qpos slice**
- Early trials teleported the target object off the table instead of moving the arm
- Root cause: the target object's free joint occupies `qpos[0:7]`, but IK was incrementing `qpos[0:7]` instead of `qpos[7:14]`
- Fix confirmed in `run_experiments.py`, `demo_grasp.py`, and documented in `thesis_direction_example.tex` §Grasp Execution

**Gripper and wrist alignment debugging**
- `gripper_debug.py`: finger geometry, contact pairs, and gap-to-cylinder diagnostics
- `wrist_test.py`: joint7 (wrist) alignment so fingers straddle the cylinder along world-Y
- `final_lift_test.py`: scripted lift with `WRIST_TARGET=0.7853` and `Y_OFFSET=0.012` to centre the cylinder between finger inner faces
- Physical lift remains unreliable with the primitive capsule gripper; proximity fallback adopted for batch experiments (see below)

**432-trial dataset collected**
- Full factorial grid run and exported to `results/experiment_results.csv`
- Grid (4 × 4 × 3 × 3 × 3 = 432): `sigma_d` ∈ {0, 0.005, 0.02, 0.04}, `rho` ∈ {1.0, 0.75, 0.5, 0.25}, `phi` ∈ {30°, 45°, 60°}, `theta` ∈ {0°, 45°, 90°}, 3 seeds per condition
- Note: grid differs slightly from original thesis design (added σ_d=0.005 and ρ=0.75; dropped φ=75°; θ now {0, 45, 90} instead of {0, 60, 120}) — update thesis tables to match collected data
- Results: **134 / 432 successes (31.0%)**; 141 trials returned `no_grasps` from CGN (mostly at φ=60° where target visibility drops)
- Per-trial logged variables: `C_pc`, `q_grasp`, `e_pose`, `n_grasps`, `success`, `obj_z_final`

**Success criterion decision**
- Physical lift (object Z > 0.55 m) attempted first; gripper friction with primitive geometry insufficient for reliable lift across 432 headless trials
- Operationalised success as **proximity fallback**: end-effector XY within 0.065 m of object centroid after approach (`D_τ = 0.065` m), preserving the causal chain (degraded perception → large e_pose → arm misses → failure)
- Documented in `thesis_direction_example.tex` §Grasp Success Criterion with calibration rationale (≈85% success under clean conditions, ≈0% under maximum degradation)

**Thesis document updated**
- Added grasp execution section (DLS IK, qpos layout, gripper control)
- Added success criterion section and updated experimental grid table to 432 trials
- Preliminary results placeholders populated with observed failure modes (CGN lateral bias at low φ, no_grasps at high φ)

**Critical questions surfaced:**
1. *Is proximity-based success acceptable for the causal claim?*
   *Resolution*: Yes, for this thesis. The causal mechanism is perceptual degradation causing the proposed grasp position to drift away from the true object. Whether the gripper physically lifts or merely reaches the correct XY position is secondary to whether perception drove the arm to the right place. Physical lift remains a stretch goal; proximity preserves the signal for SCM fitting.
2. *Should we revert to the original 144-condition grid (σ_d without 0.005, θ = {0, 60, 120})?*
   *Open*: Current 432-trial CSV uses the expanded grid. Either re-run with the original design or update the thesis experimental design section to match what was actually collected.
   *Observation*: ~3 s model load + ~1–2 s per trial on CPU. Full 432-trial batch completed in one session. GPU would speed individual trials but is not blocking.

**1 July — CGN Visualisation Engine and Gripper Geometry**

- **Visualisation Engine Built**: Implemented `visualize_cgn_grasps.py` and an annotation script to generate high-quality figures for the thesis. The pipeline now saves:
  - 4-panel perception summaries (RGB, depth, segmentation, score distribution)
  - 3D grasp distributions (the "crescent/croissant" shape of grasp candidates on the cylinder)
  - Top-10 grasp candidate tables
  - World-frame and camera-frame views
- **Headless Rendering**: Resolved Open3D GUI blocking issues on Windows by switching to Open3D's headless `OffscreenRenderer` and matplotlib fallback scatters, allowing automated batch generation of thesis figures.
- **Physical Execution Refinement**: (In Progress) Moving from top-down position-only grasping to full 6-DoF side-approach execution.
- **Gripper Geometry Clarification**: Addressed a discrepancy between the initial Panda gripper visual and the current simulation. The original MuJoCo Menagerie load used high-fidelity visual meshes. The current `grasp_scene_v2.xml` uses simplified capsule/primitive geometries. This is an intentional design choice for simulation stability: visual meshes have complex concavities that cause physics engines to struggle with contact detection, whereas primitives ensure robust, fast, and deterministic contact dynamics necessary for a batch causal study.

---

## Week 7 — June 30 to July 6

**3 July — Supervisor Meeting 3 (Dezong Zhao)**

Meeting was positively received. Strong visuals were the highlight and the supervisor said so explicitly. Key exchanges:

- **Gripper appearance**: Supervisor noticed the disjointed primitive gripper and asked why it looks different from the real Panda. Explanation given on the spot was adequate but not fully satisfying — the capsule/box approach was described as a deliberate physics stability choice. Supervisor accepted this but expects a cleaner answer or a fix.
- **Point cloud definition**: Supervisor asked "what is a point cloud?" directly. The answer given ("mathematical equivalent of an RGB image but using points instead of pixels") was accepted conversationally but did not rigorously explain what the points *are*, why the representation matters, or how back-projection from a depth buffer produces them. This was identified as a gap — the point cloud is the fundamental input to the entire pipeline and needs a rigorous, defended definition in the thesis.
- **SCM concerns**: Supervisor flagged that structural causal models can be tricky and "don't always work." He asked to see a first attempt quickly. This needs to be implemented soon and rigorously justified (choice over alternatives such as Causal VAE).
- **Action items from meeting**: (1) Write a rigorous point cloud section for the thesis. (2) Begin SCM fitting on collected data. (3) Revisit gripper geometry — fix or formally defend.

*Takeaway: The meeting pattern that worked well — bring strong visuals every time, not just at the end. This is now a standing rule.*

---

**6 July — Post-meeting working session**

Two sessions. Major decisions and writing work.

**Context file created**

- Created `CONTEXT.md` to solve the AI working memory problem. Every AI session (Cursor or Antigravity, Claude or Gemini) now has a single-file briefing document. The solution: paste or attach `CONTEXT.md` at the start of every new chat session. `project_pipeline.md` remains the deep journal; `CONTEXT.md` is the concise briefing.

**Gripper geometry investigation (reopened)**

- Re-examined the decision to use capsule primitives instead of the original Menagerie mesh geometry.
- The original reason (simulation stability during batch experiments) was re-evaluated. The Contact-GraspNet authors themselves used the full Panda URDF with complete mesh geometry — they did not simplify to primitives. The Menagerie's own `panda.xml` already uses simplified STL collision meshes for most links, so the contact detection concern is less severe than originally assumed.
- Conclusion reached: the capsule chain was overly cautious. The Menagerie's mesh-based setup is workable and would produce a gripper that matches the real Panda visually.
- Decision: revert `grasp_scene_v2.xml` to use Menagerie's native mesh-based body chain. This keeps our scene elements (table, cylinder, perception camera, actuator remapping, home keyframe) while using STL/OBJ geometry for the arm and hand.
- *Status: In progress. `grasp_scene_v2.xml` being rebuilt from the Menagerie `panda.xml` as base.*

**Point cloud — thesis writing**

- Wrote the point cloud section for the thesis. The key insight that was missing: a point cloud is not just "a list of 3D points." Each point is a *spatial measurement* obtained by back-projecting a pixel's depth value through the pinhole camera model using calibrated intrinsics (focal length, principal point). The result is a Euclidean coordinate in camera frame, transformable to world frame via the camera extrinsics. This is the same representation used by RGB-D sensors (RealSense, Kinect). The reason it matters: Contact-GraspNet was trained on point clouds and operates entirely in 3D space — it never sees a 2D image. The degradation variables (σ_d, ρ) act directly on the 3D point cloud, not on a 2D projection.

**432-trial dataset review**

- Reviewed whether the collected dataset is defensible. Decision: yes.
- Justification: 4 × 4 × 3 × 3 × 3 = 432 is a complete factorial design over all four causal variables. There is no sampling — every condition is represented. The design is better described as a randomised controlled trial than a dataset in the ML sense. For an MSc causal inference study, a complete factorial grid is the correct experimental design. The relatively small number (compared to ML datasets) is a feature, not a limitation: it enables exact computation of marginal and conditional effects without approximation.
- *To do: write this justification into the thesis methods section explicitly.*

**Why this problem matters — resolved**

- The question of whether this is "methodology looking for a problem" was examined directly.
- Resolution: the core question is whether a *structured* causal model can diagnose failures more reliably and more interpretably than a zero-shot LLM given the same observational data. That is a falsifiable methodological comparison with a known ground truth (the experimenter set the perturbation, so the true cause is always known). The practical motivation is trustworthy autonomy: in safety-critical manipulation tasks, knowing *why* a robot failed is as important as knowing *that* it failed. The SCM provides an interpretable audit trail; the LLM provides a natural-language hypothesis. Comparing them rigorously is the contribution.

**Compute resources**

- Evaluated whether the project would benefit from university HPC access.
- Current status: all 432 trials ran on CPU in under 15 minutes. CGN inference is ~1–2 s per trial on CPU. SCM fitting on 432 rows is computationally trivial.
- Decision: no blocking need for HPC at this stage. If LLM baseline requires inference calls (e.g., running a local language model), HPC may become useful. Keep the offer in reserve.

---

## Upcoming milestones

| Milestone | Target date | Status |
|---|---|---|
| Contact-GraspNet running on MuJoCo point cloud | ~June 27 | ✅ Done (30 June) |
| First end-to-end grasp attempt | ~June 28 | ✅ Done (30 June) |
| Full 432-trial dataset collected | ~30 June | ✅ Done (30 June) |
| CGN visualisation engine (thesis figures) | ~1 July | ✅ Done (1 July) |
| Supervisor Meeting 3 | 3 July | ✅ Done — strong visuals, good reception |
| Point cloud thesis section written | ~6 July | ✅ Done (6 July) |
| Gripper geometry revert to Menagerie meshes | ~7 July | 🔄 In progress |
| SCM fitting on collected data | ~7 July | ⬜ Active (starting now) |
| Counterfactual diagnosis implementation | ~14 July | ⬜ Pending |
| LLM baseline comparison | ~21 July | ⬜ Pending |
| Experiment freeze | 27 July 2026 | |
| Report submission | 14 August 2026 | |
| Poster presentation | 19 August 2026 | |

---

## Open questions (to revisit during writing)

These came up during the project and do not have final answers yet. They should appear in the thesis discussion and limitations sections.

1. What do we do with the diagnosis? Real-time recovery versus accumulated learning for future failure prevention are both valid uses. This thesis covers diagnosis only. Recovery is explicitly future work and should be framed as such.

2. Does the Gaussian noise model miss structured sensor failures? Real depth sensors fail differently near specular surfaces and at steep incidence angles. The Gaussian model is a first-order approximation. This should be acknowledged as a limitation.

3. What does a good attribution accuracy score actually mean? If the SCM explains 65% of variance, attribution accuracy will follow from that. The target cannot be set arbitrarily. Revisit once the SCM is fitted.

4. Can the LLM be given a fair test? The prompt and scoring rubric mapping LLM natural-language responses to the four variable names must be written before any LLM trials are run. Post-hoc interpretation would compromise the comparison.

5. Is this genuinely a sharp research question or is it methodology looking for a problem? The answer arrived at: the question is whether a structured causal model can diagnose failures more reliably and more interpretably than a zero-shot language model. That is a methodological comparison with a clear, falsifiable answer. The ground truth is known exactly because the perturbations are set by the experimenter.
