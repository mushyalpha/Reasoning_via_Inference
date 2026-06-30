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

## Week 5 — June 22 to June 28 (current week)

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

## Upcoming milestones

| Milestone | Target date |
|---|---|
| Contact-GraspNet running on MuJoCo point cloud | ~June 27 |
| First end-to-end grasp attempt | ~June 28 |
| Experiment freeze | 27 July 2026 |
| Report submission | 14 August 2026 |
| Poster presentation | 19 August 2026 |

---

## Open questions (to revisit during writing)

These came up during the project and do not have final answers yet. They should appear in the thesis discussion and limitations sections.

1. What do we do with the diagnosis? Real-time recovery versus accumulated learning for future failure prevention are both valid uses. This thesis covers diagnosis only. Recovery is explicitly future work and should be framed as such.

2. Does the Gaussian noise model miss structured sensor failures? Real depth sensors fail differently near specular surfaces and at steep incidence angles. The Gaussian model is a first-order approximation. This should be acknowledged as a limitation.

3. What does a good attribution accuracy score actually mean? If the SCM explains 65% of variance, attribution accuracy will follow from that. The target cannot be set arbitrarily. Revisit once the SCM is fitted.

4. Can the LLM be given a fair test? The prompt and scoring rubric mapping LLM natural-language responses to the four variable names must be written before any LLM trials are run. Post-hoc interpretation would compromise the comparison.

5. Is this genuinely a sharp research question or is it methodology looking for a problem? The answer arrived at: the question is whether a structured causal model can diagnose failures more reliably and more interpretably than a zero-shot language model. That is a methodological comparison with a clear, falsifiable answer. The ground truth is known exactly because the perturbations are set by the experimenter.
