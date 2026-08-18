# Causal Inference for Robotic Grasp Failure Diagnosis under Perceptual Degradation

**Bonolo Masima** — MSc Thesis, University of Glasgow, 2026  
Supervisor: Dr Dezong Zhao

---

## Submission files

| File | Description |
|---|---|
| `3175764M_Bonolo_Masima_report.pdf` | Final thesis report |
| `Scaled_Poster_Final_Bonolo_Masima.pdf` | A1 poster (presentation, 19 Aug 2026) |
| This repository | Code, simulation assets, and pre-computed experiment data |

If the compressed archive is too large for email, attach the **report** and **poster** directly and share the code/data folder via a cloud link (Google Drive, OneDrive, etc.).

---

## What this project does (30-second summary)

A MuJoCo simulator perturbs a robot's depth perception along four controlled axes — depth noise (σ_d), point-cloud sparsity (ρ), camera elevation (φ), and azimuth (θ). Contact-GraspNet proposes grasps; the simulator executes them and records success or failure. A **Structural Causal Model (SCM)** then diagnoses *why* a grasp failed using Pearl's counterfactual reasoning, and is compared against a **Gemini LLM baseline**.

---

## Recommended path for your supervisor (~15 minutes)

These steps let you explore the work without re-running the full experiment grid (432+ trials, several hours on GPU).

### 1. One-time setup

```bash
cd Reasoning_via_Inference
bash setup.sh
```

This creates symlinks inside `Report code scripts/` (so Python scripts find `results/`, `assets/`, etc.), installs dependencies from `Report code scripts/requirements.txt`, and runs basic sanity checks.

**Requirements:** Python 3.9+, ~2 GB free disk. GPU optional (CPU works for demos; experiments are faster on CUDA).

**macOS note:** Interactive MuJoCo viewer requires `mjpython` (bundled with `pip install mujoco`):

```bash
pip install mujoco
mjpython --version
```

**Linux / headless GPU note:** set `export MUJOCO_GL=egl` before running batch experiments. See `Report code scripts/RUNPOD_SETUP.md`.

### 2. Watch the grasp simulation (interactive)

```bash
cd "Report code scripts"
mjpython demo_floating_gripper.py --object cylinder   # macOS — opens MuJoCo viewer
python3 demo_floating_gripper.py --object cylinder    # Linux
```

Shows approach → finger close → lift + shake for each object (cylinder, box, mustard bottle). Uses hand-tuned poses by default (guaranteed success). Add `--cgn` to use Contact-GraspNet instead.

**Record a video without opening a window:**

```bash
mjpython demo_floating_gripper.py --record_dir results/figures/pickup_demo --no_viewer
```

### 3. Visualise Contact-GraspNet grasp proposals

```bash
cd "Report code scripts"
python3 visualize_cgn_grasps.py --save
```

Produces point-cloud + grasp wireframe screenshots in `results/figures/`. Control the perceptual conditions to match the thesis experiments:

```bash
python3 visualize_cgn_grasps.py --sigma_d 0.02 --rho 0.5 --phi 45 --theta 0 --save
```

Re-load saved predictions without re-running CGN:

```bash
python3 visualize_cgn_grasps.py --from_file results/cgn_predictions.npz
```

### 4. Inspect pre-computed results (no simulation needed)

All main experiment outputs are already in `results/`:

| File | Contents |
|---|---|
| `experiment_results.csv` | Primary 432-trial dataset (single cylinder; used for SCM + LLM baseline) |
| `counterfactual_groundtruth.csv` | Ground-truth single-variable interventions for 292 failed trials |
| `scm_nonparametric_report.md` | Nonparametric SCM summary (primary causal estimator) |
| `algorithm2_summary.json` | Pearl Algorithm 2 diagnosis accuracy vs ground truth |
| `llm_baseline_summary.json` | Gemini baseline accuracy (3 prompt tiers) |
| `experiment_results_v2.csv` | Multi-object extension (cylinder + box + mustard, 7560 trials) |
| `results/figures/` | All thesis figures (heatmaps, DAG, Sankey, LLM comparison, etc.) |

**Re-generate analysis figures from existing CSVs:**

```bash
cd "Report code scripts"
python3 scm_nonparametric.py          # nonparametric SCM tables (seconds)
python3 score_algorithm2.py           # Pearl counterfactual diagnosis scoring
python3 plot_llm_baseline.py          # LLM vs SCM comparison figures
python3 scm_fit.py                    # linear SCM coefficients + DAG figure
```

---

## Full reproduction pipeline

Run these only if you want to regenerate data from scratch. Pre-computed outputs are included.

### Step A — Main experiment grid (432 trials, ~2–4 h on GPU)

```bash
cd "Report code scripts"
python3 run_experiments.py --test     # smoke test (8 trials)
python3 run_experiments.py            # full 1296-trial densified grid
python3 run_experiments.py --resume   # resume after interruption
```

Output: `results/experiment_results.csv` (original 432 trials preserved; densified run writes to `experiment_results_densified.csv`).

### Step B — Counterfactual ground truth (~35–50 min)

Re-runs simulation with each perceptual variable reset to its clean baseline for every failed trial:

```bash
python3 run_counterfactual_groundtruth.py
python3 run_counterfactual_groundtruth.py --resume
```

Output: `results/counterfactual_groundtruth.csv`

### Step C — SCM fitting and diagnosis

```bash
python3 scm_nonparametric.py    # primary: stratified empirical estimators (no shape assumptions)
python3 scm_fit.py              # supplementary: linear/logistic fits + DAG figure
python3 score_algorithm2.py     # Algorithm 2 (abduction → action → prediction) scoring
```

### Step D — LLM baseline (requires Google Gemini API key)

```bash
export GEMINI_API_KEY="your-key-here"
python3 run_llm_baseline.py --dry-run --n-sample 3   # inspect prompts first
python3 run_llm_baseline.py --tier T1 T2 T3           # full run (~3500 API calls)
python3 plot_llm_baseline.py
```

Outputs: `results/llm_baseline_raw.jsonl`, `llm_baseline_results.csv`, `llm_baseline_summary.json`

### Step E — Multi-object extension (optional, GPU recommended)

```bash
python3 run_experiments_v2.py --test
python3 run_experiments_v2.py --object cylinder box mustard
python3 run_clutter_experiments.py                     # cluttered scene variant
```

See `Report code scripts/RUNPOD_SETUP.md` for GPU cloud setup.

---

## Repository layout

```
Reasoning_via_Inference/
├── 3175764M_Bonolo_Masima_report.pdf      # Final report
├── Scaled_Poster_Final_Bonolo_Masima.pdf    # Final poster
├── README.md                                 # This file
├── setup.sh                                  # One-time environment setup
│
├── Report code scripts/                      # All Python scripts (run from here)
│   ├── requirements.txt
│   ├── demo_floating_gripper.py              # Interactive grasp demo ★ start here
│   ├── visualize_cgn_grasps.py               # CGN grasp visualisation ★
│   ├── run_experiments.py                    # Main 432-trial batch runner
│   ├── run_counterfactual_groundtruth.py     # Counterfactual interventions
│   ├── scm_nonparametric.py                  # Primary SCM analysis
│   ├── score_algorithm2.py                   # Diagnosis algorithm scoring
│   ├── run_llm_baseline.py                     # Gemini LLM baseline
│   ├── sim_common.py / object_specs.py       # Shared simulation helpers
│   ├── CAUSAL_DAG_PREREGISTRATION.md         # Pre-registered causal graph
│   ├── RIGOUR_LEDGER.md                      # Design choices & assumptions log
│   └── RUNPOD_SETUP.md                       # GPU cloud experiment notes
│
├── results/                                  # Pre-computed data & figures ★
│   ├── experiment_results.csv
│   ├── counterfactual_groundtruth.csv
│   ├── scm_nonparametric_report.md
│   ├── algorithm2_summary.json
│   ├── llm_baseline_summary.json
│   └── figures/                              # All thesis figures
│
├── contact_graspnet_pytorch/                 # Contact-GraspNet (pre-trained, not retrained)
│   └── checkpoints/contact_graspnet/checkpoints/model.pt
├── mujoco_menagerie/                         # Franka Panda robot model
├── assets/ycb/                               # YCB object meshes (box, mustard bottle)
├── generated_scenes/                         # Auto-generated MuJoCo scene XMLs
└── figures/                                  # Static report/poster figures
```

> **Note:** After unzipping, run `bash setup.sh` once. It creates symlinks inside `Report code scripts/` pointing to the sibling folders above (`results/`, `assets/`, etc.).

---

## Causal variables (experiment design)

| Symbol | Name | Values |
|---|---|---|
| σ_d | Depth noise (Gaussian std dev) | 0, 0.005, 0.02, 0.04 m |
| ρ | Point-cloud keep fraction | 1.0, 0.75, 0.50, 0.25 |
| φ | Camera elevation | 30°, 45°, 60° |
| θ | Camera azimuth | 0°, 45°, 90° |

**Measured mediators:** point-cloud completeness (C_pc), grasp confidence (q_grasp), pose error (e_pose)  
**Outcome:** grasp success Y ∈ {0, 1}

Full grid: 4 × 4 × 3 × 3 conditions × 3 random seeds = **432 trials** (primary dataset).

---

## Creating the submission archive

Suggested contents for the zip sent to Dr Zhao:

**Include:**
- Both PDFs (report + poster)
- `Report code scripts/` (all `.py` scripts and `.md` docs)
- `results/` (CSVs, JSON, figures)
- `contact_graspnet_pytorch/` (includes ~26 MB model checkpoint)
- `mujoco_menagerie/`, `assets/`, `generated_scenes/`, `figures/`
- `README.md`, `setup.sh`, `msc_report.tex` (LaTeX source)

**Exclude to save space:**
- `.git/` (~history)
- `Research Papers/`, `Dezong Papers/` (reference material, not needed to run code)
- `__pycache__/`, `.DS_Store`, `~$*.pptx` (temp/lock files)
- `Report code scripts/*.pptx`, large poster working files

Approximate size after exclusions: **~450 MB** (may exceed email attachment limits — use a cloud link for the zip and email the two PDFs separately, as Dr Zhao suggested).

```bash
# Example archive command (run from parent directory):
zip -r Bonolo_Masima_MSc_submission.zip Reasoning_via_Inference \
  -x "*/.git/*" "*__pycache__/*" "*/.DS_Store" "*~$*" \
     "*/Research Papers/*" "*/Dezong Papers/*" \
     "*/Report code scripts/*.pptx"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: results/experiment_results.csv` | Run `bash setup.sh` from repo root to create symlinks |
| MuJoCo viewer won't open on macOS | Use `mjpython` instead of `python3` |
| Headless render fails on Linux | `export MUJOCO_GL=egl` (or `osmesa` as fallback) |
| CGN import error with numpy 2.x | Already patched: `np.in1d` → `np.isin` in `contact_grasp_estimator.py` |
| LLM baseline fails | Requires `GEMINI_API_KEY` env var; pre-computed results are in `results/llm_baseline_*` |

---

## Further reading (inside the repo)

- `Report code scripts/CAUSAL_DAG_PREREGISTRATION.md` — pre-registered causal graph and identifiability arguments
- `Report code scripts/RIGOUR_LEDGER.md` — complete log of design choices, assumptions, and known limitations
- `Report code scripts/RUNPOD_SETUP.md` — GPU cloud instructions for the multi-object experiment extension
- `results/scm_nonparametric_report.md` — primary SCM analysis summary with key effect sizes
