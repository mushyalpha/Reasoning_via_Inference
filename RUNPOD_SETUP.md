# Running the multi-object experiments on RunPod (GPU)

This is the handoff note for running `run_experiments_v2.py` (Experiment A,
isolated objects) and `run_clutter_experiments.py` (Experiment B, clutter)
on a rented RunPod GPU instance, instead of the local (CPU-only, sandboxed)
Mac used for development. Everything below was written and validated for
correctness on the Mac using CPU/no-render checks (see "What was validated
locally" at the end) -- the full trial batches themselves have **not**
been run yet; that's what this note is for.

## Why RunPod instead of local

MuJoCo's offscreen renderer on macOS needs a CGL/WindowServer connection
that isn't available in a sandboxed/headless context. On Linux with an
NVIDIA GPU, MuJoCo renders headlessly via EGL -- no display server needed
at all -- which sidesteps that problem entirely, and CGN inference will
also be much faster on GPU than the CPU numbers quoted below.

## One-time setup on the pod

```bash
# 1. Pick a RunPod template with an NVIDIA GPU + CUDA already installed
#    (any recent "PyTorch" template works). Clone/rsync this repo onto it.

cd Reasoning_via_Inference
pip install -r requirements.txt

# 2. Headless EGL rendering (Linux + NVIDIA only -- do NOT set this on macOS)
export MUJOCO_GL=egl

# 3. Sanity check MuJoCo can render headlessly
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('generated_scenes/scene_cylinder.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=480, width=640)
r.update_scene(d)
img = r.render()
print('EGL render OK:', img.shape)
"

# 4. Confirm CGN picks up the GPU (contact_grasp_estimator.py already
#    auto-selects cuda if available -- no code change needed)
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If step 3 fails with an EGL error, the template may be missing the NVIDIA
EGL driver libraries -- try a template explicitly advertised for
OpenGL/rendering workloads (e.g. ones used for Isaac Sim / robotics), or
fall back to `MUJOCO_GL=osmesa` (CPU software rendering, slower but no GPU
driver dependency) if `libosmesa6-dev` can be apt-installed on the pod.

## Running the experiments

```bash
# Experiment A -- isolated objects (redesigned grid: dense sigma_d and
# phi, deterministic rho, 5 seeds). Full grid is 2520 trials/object x 3
# objects = 7560 trials total.
python3 run_experiments_v2.py --object cylinder box mustard

# Faster first pass (reduced grid, ~1296 trials/object = 3888 total):
python3 run_experiments_v2.py --lean --object cylinder box mustard

# One object at a time (useful for splitting across multiple pods):
python3 run_experiments_v2.py --object mustard

# Resume after an interruption (skips (object, trial_id) rows already
# in results/experiment_results_v2.csv):
python3 run_experiments_v2.py --resume

# Smoke test only (8 trials, no CGN weight caching concerns):
python3 run_experiments_v2.py --test
```

```bash
# Experiment B -- clutter (all 3 objects together, 504-trial targeted grid;
# see run_clutter_experiments.py docstring for the exact grid + rationale)
python3 run_clutter_experiments.py
python3 run_clutter_experiments.py --resume
python3 run_clutter_experiments.py --test
```

Outputs:
- `results/experiment_results_v2.csv` -- Experiment A (has an `object`
  column; `results/experiment_results.csv`, the original 432-trial
  single-cylinder dataset, is left untouched for comparison).
- `results/clutter_results.csv` -- Experiment B (has `target_object` and
  `collision_with_neighbor` columns).

## Expected runtime

Locally on CPU (macOS), a single trial (render + CGN inference + IK +
grasp execution) took ~9-19s depending on point-cloud size / grasp count.
GPU inference should cut the CGN-inference component substantially (the
dominant per-trial cost); rendering/physics/IK stay CPU-bound regardless
of GPU. Recommend timing ~20 trials first (`--lean` is good for this) to
get a pod-specific ETA before committing to the full grid, and using
`--resume` liberally so a pod restart doesn't lose progress.

## What was validated locally (Mac, no GL / synthetic-input checks only)

- All 3 scene XMLs (`generated_scenes/scene_{cylinder,box,mustard}.xml`,
  `scene_clutter.xml`) build and load in MuJoCo.
- All objects settle stably on the table (drift < 2mm over 500 physics
  steps) after fixing two bugs found during this session:
    1. The mustard bottle's V-HACD collision mesh (merged to a single
       MuJoCo convex hull) didn't preserve a flat base -> switched its
       collision geometry to a bounding-box primitive (visual mesh keeps
       the real YCB geometry; only collision is simplified -- same
       pattern already used for the Panda arm's visual/collision split).
    2. The arm's fixed "home" pose (calibrated for the short cylinder)
       physically overlapped the taller mustard bottle -> arm now parks
       at a fixed clear position (`PARK_POS`) with collision temporarily
       disabled during that transient, before the object is allowed to
       settle.
- The IK/grasp-approach code path runs without exploding contacts for all
  3 objects (tested by moving to a pre-grasp-like position after park+settle).
- CGN's inference call path (`run_cgn`, deterministic rho downsampling,
  `best_grasp_overall`) was validated end-to-end with a synthetic depth
  map (no MuJoCo rendering needed for this check) -- confirmed the
  deterministic-rho helper gives exact-fraction, reproducible indices.
- **Determinism was verified bit-exact**: seeding both numpy and torch's
  global RNG (`sim_common.seed_cgn_global_random`) plus pinning to
  single-threaded CPU execution (`sim_common.configure_determinism`)
  gives identical CGN outputs (score matched to 8 decimal places) across
  3 repeated calls with the same seed. Before this fix, seeding only
  numpy left residual non-determinism (0.207879 vs 0.208227 on a repeat
  run) -- traced to torch's own internal RNG use inside the model
  forward pass, not just CGN's numpy-based preprocessing.
- A `np.isin` patch was required in
  `contact_graspnet_pytorch/contact_graspnet_pytorch/contact_grasp_estimator.py`
  for numpy>=2.0 compatibility (`np.in1d` was removed).
- **Not yet run**: any full trial through the real CGN model with real
  MuJoCo-rendered depth input (blocked locally by the sandboxed GL
  renderer). The original single-cylinder `run_experiments.py --test`
  *was* run successfully end-to-end locally (with the numpy patch), which
  validates the underlying CGN/MuJoCo integration pattern that
  `run_experiments_v2.py` and `run_clutter_experiments.py` both reuse via
  `sim_common.py` -- but the new object-specific code paths (box mesh,
  mustard mesh + collision-primitive override, clutter multi-body
  segmentation, `finger_nontarget_collision`) have only been validated
  with physics-only and synthetic-CGN-input checks, not a real end-to-end
  render+CGN trial. Recommend running `--test` first on the pod for both
  scripts and skimming the console output before launching the full grid.
