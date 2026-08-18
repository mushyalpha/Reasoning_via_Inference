"""
test_dag_edges.py
==================
Interventional falsification tests for the pre-registered causal DAG
(see CAUSAL_DAG_PREREGISTRATION.md, dated 5 Aug 2026).

This script does NOT derive the DAG. The DAG comes from reading the
pipeline's dataflow (dataflow fact, not a statistical fit). What this
script does is the second, independent step the pre-registration document
promises: for every edge and non-edge claimed in that document, test
whether the data behave the way that graph predicts. Because sigma_d,
rho, phi, theta are independently randomized by the grid design
(run_experiments.py's itertools.product grid + independent RNG seed per
trial), conditioning on them IS an intervention -- do(sigma_d = s) is
identical to "the subset of rows where sigma_d = s", with no backdoor
adjustment required. This is exactly the "fix phi, theta, vary sigma_d,
check whether C_pc moves" procedure described in the pre-registration.

If any test below fails, the correct response is to revisit the DAG (or
the measurement definition of the node), not to quietly re-fit around it
-- that is the whole point of pre-registering the graph before looking at
these numbers.

Usage:
    python test_dag_edges.py

Outputs:
    results/dag_edge_falsification_report.md
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

_PROJECT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_PROJECT, "results", "experiment_results.csv")
OUT_PATH = os.path.join(_PROJECT, "results", "dag_edge_falsification_report.md")

ALPHA = 0.05          # significance threshold for "moved detectably"
PRACTICAL_TOL = 0.02  # a coefficient smaller than this, in C_pc's own units
                       # (C_pc ranges ~0.005-0.09), is "practically zero"
                       # even if formally significant with n=432


def load():
    df = pd.read_csv(CSV_PATH)
    for c in ["sigma_d", "rho", "phi", "theta", "C_pc", "q_grasp",
              "e_pose", "n_grasps", "success"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_winerror"] = df["error"].astype(str).str.contains("WinError", na=False)
    df["causal_valid"] = ~df["is_winerror"]
    df["has_grasps"] = (df["error"] != "no_grasps") & df["causal_valid"]
    return df


def fmt_row(name, coef, se, p, verdict):
    sig = "*" if p < ALPHA else " "
    return f"| {name:<10} | {coef:+.6f} | {se:.6f} | {p:.4f}{sig} | {verdict} |"


# ═══════════════════════════════════════════════════════════════════════
#  Test 1 & 2: C_pc's parents. Claim: phi, theta -> C_pc; sigma_d, rho -/->
# ═══════════════════════════════════════════════════════════════════════
def test_cpc_parents(df, lines):
    lines.append("## Test 1/2 — `C_pc`'s parents: claim is {phi, theta} -> C_pc, "
                  "sigma_d and rho excluded\n")
    lines.append("Regression includes all four exogenous variables (not just "
                  "the claimed parents) so that a violated non-edge would show "
                  "up as a non-zero, significant coefficient on sigma_d or rho.\n")

    d = df[df.causal_valid].dropna(subset=["C_pc", "sigma_d", "rho", "phi", "theta"])
    X = sm.add_constant(d[["sigma_d", "rho", "phi", "theta"]])
    res = sm.OLS(d["C_pc"], X).fit(cov_type="HC3")

    lines.append(f"n={int(res.nobs)}  R2={res.rsquared:.4f}\n")
    lines.append("| Variable | Coef | SE | p | Verdict |")
    lines.append("|---|---|---|---|---|")

    verdicts = {}
    for var in ["sigma_d", "rho", "phi", "theta"]:
        coef, se, p = res.params[var], res.bse[var], res.pvalues[var]
        if var in ("sigma_d", "rho"):
            moved = (p < ALPHA) and (abs(coef) * d[var].std() > PRACTICAL_TOL)
            verdict = "FAIL (edge exists, DAG wrong)" if moved else "PASS (non-edge holds)"
        else:
            verdict = "PASS (edge confirmed)" if p < ALPHA else "WEAK (edge not detected)"
        verdicts[var] = verdict
        lines.append(fmt_row(var, coef, se, p, verdict))
    lines.append("")
    return verdicts


# ═══════════════════════════════════════════════════════════════════════
#  Test 1b: literal "fix phi,theta, vary sigma_d/rho" stratified check
# ═══════════════════════════════════════════════════════════════════════
def test_cpc_stratified(df, lines):
    lines.append("## Test 1b — literal stratified version: within each fixed "
                  "(phi, theta) cell, does C_pc move across sigma_d / rho levels?\n")
    d = df[df.causal_valid].dropna(subset=["C_pc", "sigma_d", "rho", "phi", "theta"])

    lines.append("### Fixing (phi, theta), varying sigma_d\n")
    lines.append("| phi | theta | C_pc range across sigma_d levels | max-min |")
    lines.append("|---|---|---|---|")
    max_spread_sigma = 0.0
    for (phi, theta), g in d.groupby(["phi", "theta"]):
        means = g.groupby("sigma_d")["C_pc"].mean()
        spread = means.max() - means.min()
        max_spread_sigma = max(max_spread_sigma, spread)
        vals = ", ".join(f"{v:.5f}" for v in means.values)
        lines.append(f"| {phi:.0f} | {theta:.0f} | {vals} | {spread:.5f} |")
    lines.append(f"\nMax spread across sigma_d, holding (phi,theta) fixed: "
                 f"**{max_spread_sigma:.5f}** (tolerance {PRACTICAL_TOL}) -> "
                 f"{'FAIL' if max_spread_sigma > PRACTICAL_TOL else 'PASS'}\n")

    lines.append("### Fixing (phi, theta), varying rho\n")
    lines.append("| phi | theta | C_pc range across rho levels | max-min |")
    lines.append("|---|---|---|---|")
    max_spread_rho = 0.0
    for (phi, theta), g in d.groupby(["phi", "theta"]):
        means = g.groupby("rho")["C_pc"].mean()
        spread = means.max() - means.min()
        max_spread_rho = max(max_spread_rho, spread)
        vals = ", ".join(f"{v:.5f}" for v in means.values)
        lines.append(f"| {phi:.0f} | {theta:.0f} | {vals} | {spread:.5f} |")
    lines.append(f"\nMax spread across rho, holding (phi,theta) fixed: "
                 f"**{max_spread_rho:.5f}** (tolerance {PRACTICAL_TOL}) -> "
                 f"{'FAIL' if max_spread_rho > PRACTICAL_TOL else 'PASS'}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Test 3: q_grasp / e_pose sibling claim
# ═══════════════════════════════════════════════════════════════════════
def test_qgrasp_epose_siblings(df, lines):
    lines.append("## Test 3 — `q_grasp` / `e_pose`: siblings under common parent "
                  "`S`, NOT a mediation edge\n")
    lines.append("Dataflow reading (CAUSAL_DAG_PREREGISTRATION.md Sec.4) already "
                  "rules out a `q_grasp -> e_pose` edge: `e_pose`'s code never "
                  "reads `q_grasp`'s value. A statistical test alone cannot "
                  "distinguish a chain (mediation) from a fork (common cause) -- "
                  "both predict a non-zero residual link -- so this test is run "
                  "to confirm the *link exists* (ruling out 'no relationship at "
                  "all'), not to adjudicate its direction (that was already "
                  "settled by reading the code).\n")

    d = df[df.has_grasps].dropna(
        subset=["sigma_d", "rho", "phi", "theta", "n_grasps", "q_grasp", "e_pose"]).copy()
    d["log_n_grasps"] = np.log(d["n_grasps"])

    X_wo = sm.add_constant(d[["sigma_d", "rho", "phi", "theta", "log_n_grasps"]])
    res_wo = sm.OLS(d["e_pose"], X_wo).fit(cov_type="HC3")

    X_w = sm.add_constant(d[["sigma_d", "rho", "phi", "theta", "log_n_grasps", "q_grasp"]])
    res_w = sm.OLS(d["e_pose"], X_w).fit(cov_type="HC3")

    coef, se, p = res_w.params["q_grasp"], res_w.bse["q_grasp"], res_w.pvalues["q_grasp"]
    lines.append(f"n={int(res_w.nobs)}")
    lines.append(f"R2 without q_grasp: {res_wo.rsquared:.4f}  |  "
                 f"R2 with q_grasp: {res_w.rsquared:.4f}  "
                 f"(delta R2 = {res_w.rsquared - res_wo.rsquared:.4f})\n")
    lines.append(f"q_grasp coefficient on e_pose, controlling for "
                 f"{{sigma_d, rho, phi, theta, log(n_grasps)}}: "
                 f"**{coef:+.5f}** (SE={se:.5f}, p={p:.4g})\n")
    verdict = ("CONFIRMED residual link (expected under sibling/shared-idx "
               "reading -- do not reinterpret as q_grasp causing e_pose)"
               if p < ALPHA else
               "NOT confirmed -- residual link is weaker than expected; "
               "worth re-checking the shared-idx argument in Sec.4")
    lines.append(f"Verdict: {verdict}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Test 4: exogenous mutual independence (design check)
# ═══════════════════════════════════════════════════════════════════════
def test_exogenous_independence(df, lines):
    lines.append("## Test 4 — exogenous mutual independence (grid-design check)\n")
    lines.append("sigma_d, rho, phi, theta should be mutually uncorrelated: they "
                 "are assigned by a full-factorial grid (`itertools.product`), "
                 "not sampled, so any residual correlation would indicate the "
                 "grid itself (or row exclusions) introduced an accidental "
                 "confound.\n")
    exog = ["sigma_d", "rho", "phi", "theta"]
    corr = df[exog].corr()
    lines.append("| | " + " | ".join(exog) + " |")
    lines.append("|---|" + "---|" * len(exog))
    for v in exog:
        lines.append("| " + v + " | " + " | ".join(f"{corr.loc[v,w]:+.4f}" for w in exog) + " |")
    max_off_diag = corr.values[~np.eye(len(exog), dtype=bool)]
    max_abs = np.max(np.abs(max_off_diag))
    lines.append(f"\nMax |pairwise correlation| among exogenous variables: "
                 f"**{max_abs:.4f}** -> {'PASS' if max_abs < 0.05 else 'FAIL (grid imbalance)'}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Test 5: has_grasps / n_grasps respond to sigma_d, rho (primary causal claim)
# ═══════════════════════════════════════════════════════════════════════
def test_has_grasps_moves(df, lines):
    lines.append("## Test 5 — primary causal claim: sigma_d, rho, phi, theta "
                 "-> has_grasps actually moves has_grasps\n")
    d = df[df.causal_valid].dropna(subset=["sigma_d", "rho", "phi", "theta"])
    X = sm.add_constant(d[["sigma_d", "rho", "phi", "theta"]])
    res = sm.Logit(d["has_grasps"].astype(int), X).fit(disp=0)
    lines.append("| Variable | Coef (logit) | p | Verdict |")
    lines.append("|---|---|---|---|")
    for var in ["sigma_d", "rho", "phi", "theta"]:
        coef, p = res.params[var], res.pvalues[var]
        verdict = "PASS (edge confirmed)" if p < ALPHA else "WEAK (edge not detected at n=432)"
        lines.append(f"| {var} | {coef:+.4f} | {p:.4g} | {verdict} |")
    lines.append("")


def main():
    df = load()
    lines = []
    lines.append("# DAG Edge Falsification Report\n")
    lines.append(f"Generated by `test_dag_edges.py` against "
                 f"`{os.path.relpath(CSV_PATH, _PROJECT)}` "
                 f"({len(df)} rows, {int(df.causal_valid.sum())} causally valid).\n")
    lines.append("Tests the graph pre-registered in `CAUSAL_DAG_PREREGISTRATION.md` "
                 "(dated 5 Aug 2026) against the 432-trial dataset. Because "
                 "sigma_d/rho/phi/theta are independently randomized by the grid "
                 "design, every conditional test below is an interventional test, "
                 "not merely an observational one.\n")
    lines.append("---\n")

    test_cpc_parents(df, lines)
    test_cpc_stratified(df, lines)
    test_qgrasp_epose_siblings(df, lines)
    test_exogenous_independence(df, lines)
    test_has_grasps_moves(df, lines)

    lines.append("---\n")
    lines.append("*This file is generated. Do not hand-edit — re-run "
                 "`python test_dag_edges.py` after any change to "
                 "`experiment_results.csv` or the DAG claims it tests.*")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {OUT_PATH}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
