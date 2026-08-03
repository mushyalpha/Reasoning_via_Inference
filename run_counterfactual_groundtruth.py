"""
run_counterfactual_groundtruth.py
=================================
Compute ground-truth counterfactuals for all failed trials by re-running
the MuJoCo+CGN simulation with each perceptual variable reset to its clean
baseline while the other three are held at observed values.

For each failed trial t with observed (sigma_d, rho, phi, theta):
  Intervention 1: do(sigma_d = 0.0)   -- zero depth noise
  Intervention 2: do(rho = 1.0)       -- full point cloud density
  Intervention 3: do(phi = 45.0)      -- mid-range camera elevation
  Intervention 4: do(theta = 0.0)     -- reference azimuth

primary_cause(t) = variable whose clean-baseline intervention converts Y=0->Y=1
                   (if multiple fix it: all listed joined by +; none if nothing works)

Output:
  results/counterfactual_groundtruth.csv

Fields:
  trial_id, sigma_d, rho, phi, theta, seed, observed_success,
  cf_sigma_d_success, cf_rho_success, cf_phi_success, cf_theta_success,
  n_cf_successes, primary_cause

Usage:
    python run_counterfactual_groundtruth.py            # full run
    python run_counterfactual_groundtruth.py --dry-run  # print plan, no sim
    python run_counterfactual_groundtruth.py --resume   # skip completed rows

Runtime: ~292 failed trials x 4 interventions x ~2-3s = ~35-50 min.
         Run overnight or as a background job.

Note on unit-level vs marginal counterfactuals:
  - 285 has_grasps trials: U_3, U_4 recoverable as scm_fit.py residuals.
    Unit-level counterfactuals fully supported.
  - 141 no_grasps trials: U_3, U_4 unobserved (mediators never generated).
    This script provides marginal counterfactuals via fresh sim runs.
    Both are valid; distinction noted in thesis limitations section.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import pandas as pd

_PROJECT = os.path.dirname(os.path.abspath(__file__))
_CGN_REPO = os.path.join(_PROJECT, "contact_graspnet_pytorch")
_CGN_SRC  = os.path.join(_CGN_REPO, "contact_graspnet_pytorch")
for _p in [_CGN_REPO, _CGN_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_DIR = os.path.join(_PROJECT, "results")
INPUT_CSV   = os.path.join(RESULTS_DIR, "experiment_results.csv")
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "counterfactual_groundtruth.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Clean-baseline values for each single-variable intervention
CLEAN = {"sigma_d": 0.0, "rho": 1.0, "phi": 45.0, "theta": 0.0}

GT_FIELDS = [
    "trial_id", "sigma_d", "rho", "phi", "theta", "seed",
    "observed_success",
    "cf_sigma_d_success", "cf_rho_success",
    "cf_phi_success",     "cf_theta_success",
    "n_cf_successes", "primary_cause",
]


# ======================================================================
#  Run a single trial via the existing pipeline
# ======================================================================
def _run_single(sigma_d, rho, phi, theta, seed, estimator):
    """
    Re-run one MuJoCo+CGN trial with overridden parameters.
    Returns int (0 or 1) or None on error.
    Mirrors run_experiments.py::run_trial exactly.
    """
    try:
        import mujoco
        from run_experiments import (
            set_camera, render_depth_seg, run_cgn,
            best_grasp_cam, cam_to_world, execute_grasp,
            settle, SCENE_XML,
        )
        rng   = np.random.default_rng(int(seed))
        model = mujoco.MjModel.from_xml_path(SCENE_XML)
        data  = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        set_camera(model, float(phi), float(theta))
        mujoco.mj_forward(model, data)
        settle(model, data, 200)

        depth, K, seg_map = render_depth_seg(
            model, data, sigma_d=float(sigma_d), rng=rng)
        pred_grasps, scores = run_cgn(
            depth, K, seg_map, estimator, rho=float(rho), rng=rng)

        n_grasps = sum(len(scores[k]) for k in scores)
        if n_grasps == 0:
            return 0

        pose_cam, _  = best_grasp_cam(pred_grasps, scores)
        pose_world   = cam_to_world(pose_cam, model, data)
        grasp_pos    = pose_world[:3, 3]
        success, _   = execute_grasp(model, data, grasp_pos)
        return int(success)

    except Exception as e:
        print(f"      [WARN] {type(e).__name__}: {e}")
        return None


# ======================================================================
#  Main counterfactual loop
# ======================================================================
def run_counterfactuals(dry_run=False, resume=False):
    df = pd.read_csv(INPUT_CSV)
    for col in ["sigma_d", "rho", "phi", "theta", "success"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_winerror"]  = df["error"].str.contains("WinError", na=False)
    df["causal_valid"] = ~df["is_winerror"]
    failed = df[(df["success"] == 0) & df["causal_valid"]].copy()

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  Counterfactual Ground Truth Runner")
    print(SEP)
    print(f"  Failed trials (causal_valid): {len(failed)}")
    print(f"  Interventions per trial:      4")
    total_runs = len(failed) * 4
    print(f"  Total simulation re-runs:     ~{total_runs}")
    est_min = total_runs * 2.5 / 60
    print(f"  Estimated time:               ~{est_min:.0f} minutes")
    print(SEP + "\n")

    if dry_run:
        print("DRY RUN -- simulation not started. First 10 failed trials:")
        for _, row in failed.head(10).iterrows():
            print(f"  trial={int(row.trial_id)}  sigma_d={row.sigma_d:.3f}  "
                  f"rho={row.rho:.2f}  phi={row.phi:.0f}  theta={row.theta:.0f}")
        return

    # Resume: load already-completed trial_ids
    done = set()
    if resume and os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done.add(int(row["trial_id"]))
        print(f"Resuming: {len(done)} trials already completed.")

    remaining = failed[~failed["trial_id"].isin(done)]
    print(f"Remaining: {len(remaining)} trials\n")
    if len(remaining) == 0:
        print("All done. Output:", OUTPUT_CSV)
        return

    print("Loading Contact-GraspNet (loaded once for all trials)...")
    from run_experiments import load_cgn
    estimator = load_cgn()
    print("CGN ready.\n")

    mode     = "a" if (resume and os.path.exists(OUTPUT_CSV)) else "w"
    csv_file = open(OUTPUT_CSV, mode, newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=GT_FIELDS)
    if mode == "w":
        writer.writeheader()
    csv_file.flush()

    t0 = time.time()
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        tid  = int(row.trial_id)
        seed = int(row.seed) if not pd.isna(row.seed) else 0
        obs  = {
            "sigma_d": float(row.sigma_d),
            "rho":     float(row.rho),
            "phi":     float(row.phi),
            "theta":   float(row.theta),
        }
        eta = ((time.time() - t0) / i * (len(remaining) - i) / 60
               if i > 1 else 0.0)
        print(f"[{i:>3}/{len(remaining)}] trial={tid}  "
              f"sigma_d={obs['sigma_d']:.3f}  rho={obs['rho']:.2f}  "
              f"phi={obs['phi']:.0f}  theta={obs['theta']:.0f}  "
              f"ETA={eta:.1f}min")

        # Four single-variable counterfactuals
        cf_results = {}
        for var in ["sigma_d", "rho", "phi", "theta"]:
            cf = dict(obs)
            cf[var] = CLEAN[var]
            s = _run_single(
                cf["sigma_d"], cf["rho"], cf["phi"], cf["theta"],
                seed, estimator)
            cf_results[var] = s
            clean_val = CLEAN[var]
            print(f"    do({var}={clean_val}) -> success={s}")

        # Determine primary cause
        successes = [k for k, v in cf_results.items() if v == 1]
        if not successes:
            primary = "none"
        elif len(successes) == 1:
            primary = successes[0]
        else:
            primary = "+".join(sorted(successes))

        n_succ = sum(v for v in cf_results.values() if v is not None)

        writer.writerow({
            "trial_id":           tid,
            "sigma_d":            obs["sigma_d"],
            "rho":                obs["rho"],
            "phi":                obs["phi"],
            "theta":              obs["theta"],
            "seed":               seed,
            "observed_success":   0,
            "cf_sigma_d_success": cf_results["sigma_d"],
            "cf_rho_success":     cf_results["rho"],
            "cf_phi_success":     cf_results["phi"],
            "cf_theta_success":   cf_results["theta"],
            "n_cf_successes":     n_succ,
            "primary_cause":      primary,
        })
        csv_file.flush()
        fixes = "+".join(successes) if successes else "none"
        print(f"  -> primary_cause={primary}  fixes=[{fixes}]")

    csv_file.close()
    elapsed = (time.time() - t0) / 60
    print(f"\n{SEP}")
    print(f"  Done. {len(remaining)} trials in {elapsed:.1f} minutes.")
    print(f"  Output: {OUTPUT_CSV}")
    print(SEP + "\n")

    # Summary
    out_df = pd.read_csv(OUTPUT_CSV)
    print("Primary cause distribution:")
    print(out_df["primary_cause"].value_counts())
    n_none = (out_df["primary_cause"] == "none").sum()
    pct_none = 100 * n_none / len(out_df)
    print(f"\nNo single-variable fix: {n_none}/{len(out_df)} ({pct_none:.1f}%)")
    print("  (Require joint intervention or irreducible failure)")


# ======================================================================
#  Entry point
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ground-truth counterfactuals via MuJoCo re-simulation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print trial list without running simulations")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip trials already in the output CSV")
    args = parser.parse_args()
    run_counterfactuals(dry_run=args.dry_run, resume=args.resume)
