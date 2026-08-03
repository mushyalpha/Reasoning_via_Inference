"""
scm_fit.py
==========
Structural Causal Model fitting for the MSc thesis:
  Causal Inference for Robotic Grasp Failure Diagnosis under Perceptual Degradation

Fits four structural equations to the 432-trial dataset.

Verified DAG (see implementation_plan.md for full justification):
  phi, theta              --> C_pc          (viewpoint -> silhouette, Eq 1)
  sigma_d, rho, phi, theta --> has_grasps  (Eq 2A -- PRIMARY CAUSAL CLAIM)
  sigma_d, rho, phi, theta --> n_grasps    (Eq 2B -- NegBin count)
  sigma_d, rho, phi, theta, log(n) --> q_grasp  (Eq 3)
  sigma_d, rho, phi, theta, q_grasp --> e_pose  (Eq 4, mediation test)
  (e_pose < 0.065) --> Y=1  (97.9% agreement -- definitional, not modelled)

Key audit findings (all empirically verified):
  - C_pc is INDEPENDENT of sigma_d and rho (seg_map before noise/downsample)
  - 141 rows: error=='no_grasps' (CGN perceptual collapse -- causal censoring)
  - 6 rows:   WinError (infra failure, excluded): trial_ids 63,141,349,354,355,357
  - 97.9% agreement: (e_pose < 0.065) == success  -> Y is near-deterministic threshold
  - C_pc slope per degree phi: -0.000258  (over 30 deg = -0.0077)
  - Success rates: sigma_d=0->71%, 0.005->43%, 0.02->11%, 0.04->0%

Usage:
    python scm_fit.py

Outputs:
    results/scm_coefficients.csv
    results/scm_model.json
    results/figures/scm_dag.png
    results/figures/scm_heatmap_sigma_rho.png
    results/figures/scm_heatmap_phi_theta.png
    results/figures/scm_has_grasps_calibration.png
    results/figures/scm_coefficients.png
    results/figures/scm_binned_residuals.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

warnings.filterwarnings("ignore")

_PROJECT   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(_PROJECT, "results", "experiment_results.csv")
FIG_DIR    = os.path.join(_PROJECT, "results", "figures")
COEFF_CSV  = os.path.join(_PROJECT, "results", "scm_coefficients.csv")
MODEL_JSON = os.path.join(_PROJECT, "results", "scm_model.json")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    "sigma_d": "#E76F51", "rho": "#2A9D8F",
    "phi": "#457B9D",     "theta": "#9B5DE5",
    "n_grasps": "#F4A261", "q_grasp": "#264653",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 11, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150,
})

SEP = "=" * 60


# ======================================================================
#  Step 1: Load and audit
# ======================================================================
def load_and_audit(path):
    df = pd.read_csv(path)
    for col in ["sigma_d", "rho", "phi", "theta", "C_pc", "q_grasp",
                "e_pose", "n_grasps", "success"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_winerror"]  = df["error"].str.contains("WinError", na=False)
    df["is_no_grasps"] = df["error"] == "no_grasps"
    # Causally valid = not a Windows infrastructure failure
    df["causal_valid"] = ~df["is_winerror"]
    # has_grasps = CGN returned at least one proposal AND no infra failure
    df["has_grasps"]   = (~df["is_no_grasps"]) & (~df["is_winerror"])

    print(f"\n{SEP}")
    print("  SCM FIT -- Data Audit")
    print(SEP)
    we_ids = sorted(df[df.is_winerror].trial_id.tolist())
    print(f"  Total rows             : {len(df)}")
    print(f"  WinError rows (excl.)  : {df.is_winerror.sum()}  -> trial_ids {we_ids}")
    print(f"  no_grasps rows         : {df.is_no_grasps.sum()}  (CGN perceptual collapse)")
    print(f"  Causally valid rows    : {df.causal_valid.sum()}")
    print(f"  has_grasps rows        : {df.has_grasps.sum()}")
    succ_rate = df[df.causal_valid].success.mean()
    print(f"  Successes              : {int(df.success.sum())} / "
          f"{df.causal_valid.sum()} valid ({100*succ_rate:.1f}%)")
    print()

    # Confirm C_pc independence from sigma_d / rho
    cv = df.dropna(subset=["C_pc"])
    cs = cv.groupby("sigma_d")["C_pc"].mean()
    cr = cv.groupby("rho")["C_pc"].mean()
    print(f"  C_pc range by sigma_d: [{cs.min():.5f}, {cs.max():.5f}]  (should be ~ constant)")
    print(f"  C_pc range by rho:     [{cr.min():.5f}, {cr.max():.5f}]  (should be ~ constant)")
    cp = cv.groupby("phi")["C_pc"].mean()
    print(f"  C_pc by phi: {dict(zip([int(k) for k in cp.index], [round(v,5) for v in cp.values]))}")
    print()

    # e_pose vs Y circularity check
    valid = df[df.has_grasps].dropna(subset=["e_pose", "success"])
    agree = ((valid["e_pose"] < 0.065).astype(int) == valid["success"].astype(int)).mean()
    print(f"  (e_pose<0.065)==success agreement: {agree:.1%}  (Y is near-deterministic threshold)")
    print(f"  -> Logistic Y~e_pose excluded from primary analysis (see sanity_check_outcome)")
    print(SEP + "\n")
    return df


# ======================================================================
#  Helper: print OLS summary
# ======================================================================
def _print_ols(res, label):
    print(f"\n-- {label} --")
    print(f"  n={int(res.nobs)}  R2={res.rsquared:.4f}  "
          f"adj-R2={res.rsquared_adj:.4f}  AIC={res.aic:.1f}")
    ci = res.conf_int()
    print(f"  {'Variable':<22} {'Coef':>10} {'CI_lo':>10} {'CI_hi':>10} {'p':>8}")
    for var in res.params.index:
        sig = "*" if res.pvalues[var] < 0.05 else " "
        print(f"  {var:<22} {res.params[var]:>10.5f} "
              f"{ci.loc[var,0]:>10.5f} {ci.loc[var,1]:>10.5f} "
              f"{res.pvalues[var]:>8.4f}{sig}")


# ======================================================================
#  Equation 1: C_pc ~ phi + theta
# ======================================================================
def fit_eq1_cpc(df):
    d = df[df["causal_valid"]].dropna(subset=["C_pc", "phi", "theta"])
    X = sm.add_constant(d[["phi", "theta"]])
    res = sm.OLS(d["C_pc"], X).fit(cov_type="HC3")
    _print_ols(res, "Eq 1: C_pc ~ phi + theta  [sigma_d, rho excluded by construction]")
    print(f"  alpha_phi per degree: {res.params['phi']:.6f}  (expected ~ -0.000258)")
    print(f"  alpha_theta:          {res.params['theta']:.6f}  (expected ~ 0, cylinder symmetry)")
    return res


# ======================================================================
#  Equation 2A: has_grasps ~ sigma_d+rho+phi+theta  [PRIMARY CAUSAL CLAIM]
# ======================================================================
def fit_eq2a_has_grasps(df):
    d = df[df["causal_valid"]].dropna(subset=["sigma_d", "rho", "phi", "theta"])
    xcols = ["sigma_d", "rho", "phi", "theta"]
    X = sm.add_constant(d[xcols])
    res = sm.Logit(d["has_grasps"].astype(int), X).fit(disp=0)

    print(f"\n-- Eq 2A: has_grasps ~ sigma_d+rho+phi+theta  [PRIMARY CAUSAL CLAIM] --")
    print(f"  n={int(res.nobs)}  pseudo-R2={res.prsquared:.4f}  "
          f"AIC={res.aic:.1f}  LLR p={res.llr_pvalue:.2e}")
    ci = res.conf_int()
    print(f"  {'Variable':<22} {'Coef(logit)':>12} {'OddsRatio':>11} "
          f"{'OR_lo':>8} {'OR_hi':>8} {'p':>8}")
    for var in res.params.index:
        OR    = np.exp(res.params[var])
        OR_lo = np.exp(ci.loc[var, 0])
        OR_hi = np.exp(ci.loc[var, 1])
        sig   = "*" if res.pvalues[var] < 0.05 else " "
        print(f"  {var:<22} {res.params[var]:>12.4f} {OR:>11.3f} "
              f"{OR_lo:>8.3f} {OR_hi:>8.3f} {res.pvalues[var]:>8.4f}{sig}")

    y_prob = res.predict(X)
    auc = roc_auc_score(d["has_grasps"].astype(int), y_prob)
    print(f"  ROC AUC: {auc:.4f}")
    return res, d, xcols


# ======================================================================
#  Equation 2B: n_grasps ~ sigma_d+rho+phi+theta  [NegBin, n_grasps>0]
# ======================================================================
def fit_eq2b_n_grasps(df):
    d = df[df["has_grasps"]].dropna(
        subset=["sigma_d", "rho", "phi", "theta", "n_grasps"])
    X = sm.add_constant(d[["sigma_d", "rho", "phi", "theta"]])
    res = sm.GLM(d["n_grasps"], X,
                 family=sm.families.NegativeBinomial()).fit(disp=0)
    print(f"\n-- Eq 2B: n_grasps ~ sigma_d+rho+phi+theta  [NegBin, n={int(res.nobs)}] --")
    print(f"  Deviance={res.deviance:.2f}  AIC={res.aic:.1f}")
    ci = res.conf_int()
    print(f"  {'Variable':<22} {'Coef(log)':>10} {'exp(Coef)':>10} "
          f"{'CI_lo':>8} {'CI_hi':>8} {'p':>8}")
    for var in res.params.index:
        sig = "*" if res.pvalues[var] < 0.05 else " "
        ec  = np.exp(res.params[var])
        print(f"  {var:<22} {res.params[var]:>10.4f} {ec:>10.3f} "
              f"{np.exp(ci.loc[var,0]):>8.3f} {np.exp(ci.loc[var,1]):>8.3f} "
              f"{res.pvalues[var]:>8.4f}{sig}")
    return res


# ======================================================================
#  Equation 3: q_grasp ~ sigma_d+rho+phi+theta+log(n_grasps)
# ======================================================================
def fit_eq3_q_grasp(df):
    d = df[df["has_grasps"]].dropna(
        subset=["sigma_d", "rho", "phi", "theta", "n_grasps", "q_grasp"]).copy()
    d["log_n_grasps"] = np.log(d["n_grasps"])
    X = sm.add_constant(d[["sigma_d", "rho", "phi", "theta", "log_n_grasps"]])
    res = sm.OLS(d["q_grasp"], X).fit(cov_type="HC3")
    _print_ols(res, "Eq 3: q_grasp ~ sigma_d+rho+phi+theta+log(n_grasps)")
    print("  NOTE: conditional on has_grasps=1; log(n_grasps) corrects "
          "order-statistic selection bias")
    return res


# ======================================================================
#  Equation 4: e_pose ~ sigma_d+rho+phi+theta+q_grasp  (+mediation test)
# ======================================================================
def fit_eq4_e_pose(df):
    d = df[df["has_grasps"]].dropna(
        subset=["sigma_d", "rho", "phi", "theta", "q_grasp", "e_pose"])
    # Total effects (without q_grasp)
    X_tot = sm.add_constant(d[["sigma_d", "rho", "phi", "theta"]])
    res_tot = sm.OLS(d["e_pose"], X_tot).fit(cov_type="HC3")
    # Direct effects (with q_grasp as mediator)
    X_dir = sm.add_constant(d[["sigma_d", "rho", "phi", "theta", "q_grasp"]])
    res_dir = sm.OLS(d["e_pose"], X_dir).fit(cov_type="HC3")

    _print_ols(res_tot, "Eq 4a: e_pose ~ sigma_d+rho+phi+theta  [total effects]")
    _print_ols(res_dir, "Eq 4b: e_pose ~ sigma_d+rho+phi+theta+q_grasp  [direct effects]")

    print("\n  Mediation via q_grasp -- coefficient change:")
    for var in ["sigma_d", "rho", "phi", "theta"]:
        ct  = res_tot.params.get(var, 0)
        cd  = res_dir.params.get(var, 0)
        pct = (ct - cd) / (abs(ct) + 1e-12) * 100
        print(f"    {var:<12}: total={ct:+.5f}  direct={cd:+.5f}  mediated={pct:.1f}%")
    return res_dir, res_tot


# ======================================================================
#  Sanity check: Y ~ e_pose  (definitional -- NOT a causal result)
# ======================================================================
def sanity_check_outcome(df):
    d = df[df["has_grasps"]].dropna(subset=["e_pose", "success"])
    X = sm.add_constant(d[["e_pose"]])
    res = sm.Logit(d["success"].astype(int), X).fit(disp=0)
    y_prob = res.predict(X)
    auc = roc_auc_score(d["success"].astype(int), y_prob)
    print(f"\n-- SANITY CHECK ONLY: Y ~ e_pose  [DO NOT report as causal result] --")
    print(f"  AUC = {auc:.4f}")
    print(f"  High AUC expected by construction: Y = (xy_dist < 0.065), e_pose ~ xy_dist")
    print(f"  This equation is excluded from the primary SCM.")
    return res


# ======================================================================
#  Step 8: Binned residual plots
# ======================================================================
def plot_binned_residuals(df, eq1, eq3, eq4_dir):
    d_full  = df[df["causal_valid"]].dropna(subset=["C_pc", "phi", "theta"])
    d_valid = df[df["has_grasps"]].dropna(
        subset=["q_grasp", "e_pose", "n_grasps",
                "sigma_d", "rho", "phi", "theta"]).copy()
    d_valid["log_n_grasps"] = np.log(d_valid["n_grasps"])

    resid_e1 = (d_full["C_pc"]
                - eq1.predict(sm.add_constant(d_full[["phi", "theta"]])))
    resid_e3 = (d_valid["q_grasp"]
                - eq3.predict(sm.add_constant(
                    d_valid[["sigma_d", "rho", "phi", "theta", "log_n_grasps"]])))
    resid_e4 = (d_valid["e_pose"]
                - eq4_dir.predict(sm.add_constant(
                    d_valid[["sigma_d", "rho", "phi", "theta", "q_grasp"]])))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Binned Residual Plots -- Structural Equation Misspecification Check",
                 fontsize=12, fontweight="bold", y=1.01)

    def _bin(ax, x_series, resid_series, xlabel, title, color):
        df_r = pd.DataFrame({"x": x_series.values, "r": resid_series.values})
        grp  = df_r.groupby("x")["r"]
        mn   = grp.mean()
        se   = grp.sem().fillna(0)
        bars = list(mn.index)
        w    = (bars[1] - bars[0]) * 0.6 if len(bars) > 1 else 0.5
        ax.bar(bars, mn.values, color=color, alpha=0.75, width=w)
        ax.errorbar(bars, mn.values, yerr=1.96 * se.values,
                    fmt="none", color="#1D3557", capsize=4, lw=1.5)
        ax.axhline(0, color="#888", lw=1, ls="--")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.set_ylabel("Mean residual")

    _bin(axes[0, 0], d_full["phi"],      resid_e1, "phi (deg)",
         "Eq1: C_pc by phi",    PALETTE["phi"])
    _bin(axes[0, 1], d_full["theta"],    resid_e1, "theta (deg)",
         "Eq1: C_pc by theta",  PALETTE["theta"])
    _bin(axes[0, 2], d_valid["sigma_d"], resid_e3, "sigma_d",
         "Eq3: q_grasp by sigma_d", PALETTE["sigma_d"])
    _bin(axes[1, 0], d_valid["phi"],     resid_e3, "phi (deg)",
         "Eq3: q_grasp by phi",  PALETTE["phi"])
    _bin(axes[1, 1], d_valid["sigma_d"], resid_e4, "sigma_d",
         "Eq4: e_pose by sigma_d", PALETTE["sigma_d"])
    _bin(axes[1, 2], d_valid["phi"],     resid_e4, "phi (deg)",
         "Eq4: e_pose by phi",  PALETTE["phi"])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "scm_binned_residuals.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
#  Step 9: Save coefficient CSV and model JSON
# ======================================================================
def save_outputs(eq1, eq2a, eq2a_xcols, eq2b, eq3, eq4_dir):
    rows = []

    def _add(res, eqname, outcome, mtype):
        ci = res.conf_int()
        for var in res.params.index:
            if mtype == "OLS":
                fit_stat = round(float(res.rsquared), 4)
            elif mtype == "Logit":
                fit_stat = round(float(res.prsquared), 4)
            else:
                fit_stat = None
            rows.append({
                "equation":       eqname,
                "outcome":        outcome,
                "predictor":      var,
                "model_type":     mtype,
                "coef":           round(float(res.params[var]), 6),
                "ci_lo":          round(float(ci.loc[var, 0]), 6),
                "ci_hi":          round(float(ci.loc[var, 1]), 6),
                "se":             round(float(res.bse[var]), 6),
                "p_value":        round(float(res.pvalues[var]), 6),
                "n_obs":          int(res.nobs),
                "r2_or_pseudo":   fit_stat,
            })

    _add(eq1,     "Eq1",  "C_pc",       "OLS")
    _add(eq2a,    "Eq2A", "has_grasps", "Logit")
    _add(eq2b,    "Eq2B", "n_grasps",   "NegBin")
    _add(eq3,     "Eq3",  "q_grasp",    "OLS")
    _add(eq4_dir, "Eq4",  "e_pose",     "OLS")

    pd.DataFrame(rows).to_csv(COEFF_CSV, index=False)
    print(f"  Saved: {COEFF_CSV}")

    def _cov(res):
        try:
            return {str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                    for k, v in res.cov_params().to_dict().items()}
        except Exception:
            return {}

    model_dict = {
        "description": "SCM structural equations -- 432-trial robotic grasp dataset",
        "threshold_m": 0.065,
        "winerror_trial_ids": [63, 141, 349, 354, 355, 357],
        "audit": {
            "no_grasps_rows": 141,
            "winerror_rows": 6,
            "e_pose_y_agreement": 0.979,
            "C_pc_slope_per_degree_phi": -0.000258,
        },
        "Eq1_C_pc": {
            "type": "OLS", "outcome": "C_pc",
            "coef": {str(k): float(v) for k, v in eq1.params.items()},
            "cov": _cov(eq1),
            "r2": float(eq1.rsquared), "n": int(eq1.nobs),
            "note": "C_pc independent of sigma_d, rho -- verified empirically",
        },
        "Eq2A_has_grasps": {
            "type": "Logit", "outcome": "has_grasps",
            "coef": {str(k): float(v) for k, v in eq2a.params.items()},
            "cov": _cov(eq2a),
            "pseudo_r2": float(eq2a.prsquared), "n": int(eq2a.nobs),
            "note": "PRIMARY CAUSAL CLAIM -- probability of CGN producing any grasp",
        },
        "Eq2B_n_grasps": {
            "type": "NegativeBinomial",
            "outcome": "n_grasps (conditional on has_grasps=1)",
            "coef": {str(k): float(v) for k, v in eq2b.params.items()},
            "cov": _cov(eq2b),
            "n": int(eq2b.nobs),
            "note": "Required for counterfactual pipeline: feeds log(n_grasps) into Eq3",
        },
        "Eq3_q_grasp": {
            "type": "OLS", "outcome": "q_grasp",
            "coef": {str(k): float(v) for k, v in eq3.params.items()},
            "cov": _cov(eq3),
            "r2": float(eq3.rsquared), "n": int(eq3.nobs),
            "note": "Conditional on has_grasps=1; log(n_grasps) corrects order-statistic bias",
        },
        "Eq4_e_pose": {
            "type": "OLS", "outcome": "e_pose",
            "coef": {str(k): float(v) for k, v in eq4_dir.params.items()},
            "cov": _cov(eq4_dir),
            "r2": float(eq4_dir.rsquared), "n": int(eq4_dir.nobs),
            "note": "Direct effects after q_grasp mediation; covariance matrix for counterfactual uncertainty",
        },
    }
    with open(MODEL_JSON, "w") as f:
        json.dump(model_dict, f, indent=2, default=str)
    print(f"  Saved: {MODEL_JSON}")


# ======================================================================
#  Step 10a: DAG figure
# ======================================================================
def fig_dag():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    nodes = {
        "sigma_d":    (0.7, 5.8),
        "rho":        (0.7, 4.5),
        "phi":        (0.7, 3.2),
        "theta":      (0.7, 1.9),
        "C_pc":       (3.2, 0.9),
        "has_grasps": (3.2, 3.9),
        "n_grasps":   (5.2, 3.9),
        "q_grasp":    (7.1, 3.9),
        "e_pose":     (8.9, 3.9),
        "Y":          (10.4, 3.9),
    }
    labels = {
        "sigma_d":    "sigma_d\n(depth noise)",
        "rho":        "rho\n(sparsity)",
        "phi":        "phi\n(elevation)",
        "theta":      "theta\n(azimuth)",
        "C_pc":       "C_pc\n(coverage)",
        "has_grasps": "has_grasps\n(any proposals?)",
        "n_grasps":   "n_grasps\n(count)",
        "q_grasp":    "q_grasp\n(best score)",
        "e_pose":     "e_pose\n(pose error)",
        "Y":          "Y\n(success)",
    }
    colors = {
        "sigma_d": "#E76F51", "rho": "#2A9D8F",
        "phi": "#457B9D",     "theta": "#9B5DE5",
        "C_pc": "#A8DADC",    "has_grasps": "#F4A261",
        "n_grasps": "#E9C46A", "q_grasp": "#264653",
        "e_pose": "#6B9BC3",  "Y": "#2A9D8F",
    }
    dark_nodes = {"#E76F51", "#2A9D8F", "#9B5DE5", "#457B9D", "#264653"}
    r = 0.50

    for n, (x, y) in nodes.items():
        c = plt.Circle((x, y), r, color=colors[n], zorder=3,
                        linewidth=1.5, edgecolor="#1D3557")
        ax.add_patch(c)
        tc = "white" if colors[n] in dark_nodes else "#1D3557"
        ax.text(x, y, labels[n], ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=tc, zorder=4)

    def _arr(src, dst, color="#1D3557", lw=1.5, rad=0.0):
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        dx, dy  = x1 - x0, y1 - y0
        L       = np.sqrt(dx**2 + dy**2)
        ux, uy  = dx / L, dy / L
        xs, ys  = x0 + ux * r, y0 + uy * r
        xe, ye  = x1 - ux * r, y1 - uy * r
        cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    connectionstyle=cs))

    # phi, theta -> C_pc only (sigma_d, rho do NOT affect C_pc)
    _arr("phi",   "C_pc", color="#457B9D", lw=2.0)
    _arr("theta", "C_pc", color="#9B5DE5", lw=2.0)
    # all exogenous -> has_grasps
    for n in ["sigma_d", "rho", "phi", "theta"]:
        _arr(n, "has_grasps", color=colors[n], lw=1.8)
    # main mediator chain
    _arr("has_grasps", "n_grasps", lw=2.2)
    _arr("n_grasps",   "q_grasp",  lw=2.2)
    _arr("q_grasp",    "e_pose",   lw=2.2)
    _arr("e_pose",     "Y",        lw=2.5)
    # phi, theta -> e_pose direct path (dashed)
    ax.annotate("", xy=(nodes["e_pose"][0] - r, nodes["e_pose"][1] - 0.3),
                xytext=(nodes["phi"][0] + r, nodes["phi"][1] - 0.2),
                arrowprops=dict(arrowstyle="->", color="#457B9D",
                                lw=1.1, linestyle="dashed",
                                connectionstyle="arc3,rad=-0.25"))
    # no_grasps -> Y=0 annotation
    hg = nodes["has_grasps"]
    ax.annotate("has_grasps=0\n-> Y=0 directly",
                xy=(hg[0], hg[1] - r - 0.05),
                xytext=(hg[0] + 0.6, hg[1] - 1.8),
                ha="center", fontsize=8.5, color="#C84B31", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C84B31", lw=1.3))

    ax.set_title(
        "Structural Causal Model for Robotic Grasp Failure Diagnosis\n"
        "C_pc depends on phi, theta only  |  "
        "sigma_d and rho bypass C_pc and act directly on CGN",
        fontsize=11, fontweight="bold", pad=12)

    patches = [
        mpatches.Patch(color="#E76F51", label="sigma_d: depth noise"),
        mpatches.Patch(color="#2A9D8F", label="rho: sparsity"),
        mpatches.Patch(color="#457B9D", label="phi: elevation"),
        mpatches.Patch(color="#9B5DE5", label="theta: azimuth"),
        mpatches.Patch(color="#A8DADC", label="C_pc: viewpoint mediator only"),
        mpatches.Patch(color="#F4A261", label="has_grasps: primary causal node"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=8.5,
              framealpha=0.92, ncol=2)

    out = os.path.join(FIG_DIR, "scm_dag.png")
    plt.savefig(out, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
#  Step 10b: Heatmaps
# ======================================================================
def fig_heatmaps(df):
    d = df[df["causal_valid"]]

    def _heat(pivot, title, xlabel, ylabel, outpath):
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if v < 0.35 else "#1D3557")
        plt.colorbar(im, ax=ax, label="Success rate", shrink=0.85)
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {outpath}")

    pivot1 = d.groupby(["sigma_d", "rho"])["success"].mean().unstack()
    _heat(pivot1, "Grasp Success Rate: sigma_d x rho",
          "Point cloud sparsity (rho)", "Depth noise (sigma_d)",
          os.path.join(FIG_DIR, "scm_heatmap_sigma_rho.png"))

    pivot2 = d.groupby(["phi", "theta"])["success"].mean().unstack()
    pivot2.index  = [f"phi={int(v)}deg" for v in pivot2.index]
    pivot2.columns = [f"theta={int(v)}deg" for v in pivot2.columns]
    _heat(pivot2, "Grasp Success Rate: phi x theta",
          "Camera azimuth (theta)", "Camera elevation (phi)",
          os.path.join(FIG_DIR, "scm_heatmap_phi_theta.png"))


# ======================================================================
#  Step 10c: Calibration plot
# ======================================================================
def fig_calibration(df, eq2a, xcols):
    d = df[df["causal_valid"]].dropna(subset=["sigma_d", "rho", "phi", "theta"])
    X = sm.add_constant(d[xcols])
    y_prob = eq2a.predict(X)
    y_true = d["has_grasps"].astype(int).values

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10,
                                             strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Eq 2A Calibration -- P(has_grasps) Logistic Regression",
                 fontsize=12, fontweight="bold")

    axes[0].plot(mean_pred, frac_pos, "o-", color="#E76F51", lw=2, label="SCM Eq 2A")
    axes[0].plot([0, 1], [0, 1], "--", color="#888", label="Perfect calibration")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].text(0.05, 0.92, f"ROC AUC = {auc:.3f}",
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="#E8F4FD", alpha=0.8))

    axes[1].hist(y_prob[y_true == 1], bins=20, alpha=0.7, color="#2A9D8F",
                 label="has_grasps=1", density=True)
    axes[1].hist(y_prob[y_true == 0], bins=20, alpha=0.7, color="#E76F51",
                 label="has_grasps=0", density=True)
    axes[1].set_xlabel("Predicted P(has_grasps)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score Distribution by Class")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "scm_has_grasps_calibration.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
#  Step 10d: Coefficient forest plot
# ======================================================================
def fig_coefficients(eq1, eq2a, eq2b, eq3, eq4_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 7))
    fig.suptitle("SCM Structural Equation Coefficients (+/- 95% CI)",
                 fontsize=13, fontweight="bold")

    def _forest(ax, res, title, vars_to_plot):
        ci         = res.conf_int()
        valid_vars = [v for v in vars_to_plot if v in res.params.index]
        coefs  = [float(res.params[v])   for v in valid_vars]
        ci_los = [float(ci.loc[v, 0])    for v in valid_vars]
        ci_his = [float(ci.loc[v, 1])    for v in valid_vars]
        pvals  = [float(res.pvalues[v])  for v in valid_vars]

        y     = list(range(len(valid_vars)))
        clrs  = ["#E76F51" if p < 0.05 else "#AAAAAA" for p in pvals]
        ax.barh(y, coefs, color=clrs, alpha=0.8, height=0.5)
        for i, (lo, hi) in enumerate(zip(ci_los, ci_his)):
            ax.plot([lo, hi], [i, i], color="#1D3557", lw=2, zorder=4)
            ax.plot([lo, lo, hi, hi], [i, i, i, i], "o", color="#1D3557",
                    ms=4, zorder=5)
        ax.axvline(0, color="#888", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(valid_vars, fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Coefficient", fontsize=9)
        sig_p = mpatches.Patch(color="#E76F51", alpha=0.8, label="p<0.05")
        ns_p  = mpatches.Patch(color="#AAAAAA", alpha=0.8, label="n.s.")
        ax.legend(handles=[sig_p, ns_p], fontsize=7, loc="lower right")

    _forest(axes[0], eq1,
            "Eq1: C_pc\n~ phi + theta",
            ["phi", "theta"])
    _forest(axes[1], eq2a,
            "Eq2A: has_grasps\n~ sigma_d+rho+phi+theta",
            ["sigma_d", "rho", "phi", "theta"])
    _forest(axes[2], eq3,
            "Eq3: q_grasp\n~ all + log(n)",
            ["sigma_d", "rho", "phi", "theta", "log_n_grasps"])
    _forest(axes[3], eq4_dir,
            "Eq4: e_pose\n~ all + q_grasp",
            ["sigma_d", "rho", "phi", "theta", "q_grasp"])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "scm_coefficients.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ======================================================================
#  Main
# ======================================================================
def main():
    print(f"\n{SEP}")
    print("  SCM Fitting -- MSc Thesis Causal Inference Pipeline")
    print(SEP)

    df = load_and_audit(CSV_PATH)

    eq1                         = fit_eq1_cpc(df)
    eq2a, eq2a_data, eq2a_cols  = fit_eq2a_has_grasps(df)
    eq2b                        = fit_eq2b_n_grasps(df)
    eq3                         = fit_eq3_q_grasp(df)
    eq4_dir, _eq4_tot           = fit_eq4_e_pose(df)
    sanity_check_outcome(df)

    print("\n-- Generating binned residual plots --")
    plot_binned_residuals(df, eq1, eq3, eq4_dir)

    print("\n-- Saving coefficient table and model JSON --")
    save_outputs(eq1, eq2a, eq2a_cols, eq2b, eq3, eq4_dir)

    print("\n-- Generating thesis figures --")
    fig_dag()
    fig_heatmaps(df)
    fig_calibration(df, eq2a, eq2a_cols)
    fig_coefficients(eq1, eq2a, eq2b, eq3, eq4_dir)

    print(f"\n{SEP}")
    print("  SCM fitting complete.")
    print(f"  Coefficient table : results/scm_coefficients.csv")
    print(f"  Model JSON        : results/scm_model.json")
    print(f"  Figures           : results/figures/scm_*.png")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
