#!/usr/bin/env python3
"""
Score Algorithm 2 (Pearl abduction-action-prediction) against
results/counterfactual_groundtruth.csv.

This is the marker's fix #1: a scripting task, not a new experiment.
No MuJoCo, no CGN, no LLM API.

Pilot Y is not a fitted regression. Under the proximity criterion it is
    Y = 1{has_grasps} * 1{e_pose < D_tau},   D_tau = 0.065 m
so the Algorithm-2 prediction is
    P_hat(Y=1) = p_gate(x) * 1{e_pose_cf < D_tau}
with e_pose_cf = f_e(x_cf) + U_e  (U_e abducted when e_pose is observed,
U_e = 0 on no_grasps trials where the mediator was never generated).

U_n is not abducted: under the sibling specification, n_grasps does not
enter the pilot Y-step (it feeds q_grasp, and q_grasp is not a parent of
e_pose). U_C and U_q are recovered for the NPSEM-ie residual check but
do not enter P_hat(Y=1).

Outputs:
  results/algorithm2_scored.csv
  results/algorithm2_summary.json
  results/scm_residual_correlations.csv
  results/scm_kfold_metrics.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXP_CSV = RESULTS / "experiment_results.csv"
GT_CSV = RESULTS / "counterfactual_groundtruth.csv"
LLM_CSV = RESULTS / "llm_baseline_results.csv"

D_TAU = 0.065
CLEAN = {"sigma_d": 0.0, "rho": 1.0, "phi": 45.0, "theta": 0.0}
VARS = ["sigma_d", "rho", "phi", "theta"]
SINGLE_CAUSES = set(VARS)


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    s = pd.Series(y_score)
    ranks = s.rank(method="average").to_numpy()
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def kfold_indices(n, n_splits=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train, test


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_pilot():
    df = pd.read_csv(EXP_CSV)
    for col in ["sigma_d", "rho", "phi", "theta", "C_pc", "q_grasp",
                "e_pose", "n_grasps", "success"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_winerror"] = df["error"].str.contains("WinError", na=False)
    df["is_no_grasps"] = df["error"] == "no_grasps"
    df["causal_valid"] = ~df["is_winerror"]
    df["has_grasps"] = (~df["is_no_grasps"]) & (~df["is_winerror"])
    return df


def fit_surrogate(df):
    """Fit the thesis equations, including sibling e_pose (no q_grasp)."""
    cv = df[df["causal_valid"]].copy()
    hg = df[df["has_grasps"]].copy()
    hg["log_n_grasps"] = np.log(hg["n_grasps"])

    eq1 = sm.OLS(cv["C_pc"], sm.add_constant(cv[["phi", "theta"]])).fit(cov_type="HC3")
    eq2a = sm.Logit(
        cv["has_grasps"].astype(int),
        sm.add_constant(cv[VARS]),
    ).fit(disp=0)
    eq2b = sm.GLM(
        hg["n_grasps"],
        sm.add_constant(hg[VARS]),
        family=sm.families.NegativeBinomial(),
    ).fit(disp=0)
    eq3 = sm.OLS(
        hg["q_grasp"],
        sm.add_constant(hg[VARS + ["log_n_grasps"]]),
    ).fit(cov_type="HC3")
    # Sibling specification used by Algorithm 2 / Eq. (eq:ep)
    eq4 = sm.OLS(
        hg["e_pose"],
        sm.add_constant(hg[VARS]),
    ).fit(cov_type="HC3")
    # Mediation spec retained only to match Table 6.8's historical R^2
    eq4_med = sm.OLS(
        hg["e_pose"],
        sm.add_constant(hg[VARS + ["q_grasp"]]),
    ).fit(cov_type="HC3")

    Xg = sm.add_constant(cv[VARS])
    auc = roc_auc_score(cv["has_grasps"].astype(int), eq2a.predict(Xg))
    return {
        "eq1": eq1, "eq2a": eq2a, "eq2b": eq2b, "eq3": eq3,
        "eq4": eq4, "eq4_med": eq4_med, "auc_gate": float(auc),
        "n_cv": int(eq1.nobs), "n_hg": int(eq4.nobs),
    }


def linpred(res, xrow, extra=None):
    names = [n for n in res.params.index if n != "const"]
    val = float(res.params["const"])
    for n in names:
        if extra is not None and n in extra:
            val += float(res.params[n]) * float(extra[n])
        else:
            val += float(res.params[n]) * float(xrow[n])
    return val


def sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))


def p_gate(models, xrow):
    return sigmoid(linpred(models["eq2a"], xrow))


def e_mean(models, xrow):
    return linpred(models["eq4"], xrow)


def p_y(models, xrow, u_e):
    """Pilot prediction: gate probability times hard threshold on e_pose."""
    e_cf = e_mean(models, xrow) + u_e
    return p_gate(models, xrow) * float(e_cf < D_TAU), e_cf


def abduct_row(models, row):
    """Recover unit-level residuals. Missing mediators -> U = 0."""
    u_c = np.nan
    u_q = np.nan
    u_e = 0.0
    u_n = np.nan
    u_e_observed = False

    if pd.notna(row.get("C_pc")):
        u_c = float(row["C_pc"]) - linpred(models["eq1"], row)

    if bool(row["has_grasps"]) and pd.notna(row.get("e_pose")):
        u_e = float(row["e_pose"]) - e_mean(models, row)
        u_e_observed = True
        mu_n = np.exp(linpred(models["eq2b"], row))
        u_n = float(row["n_grasps"]) - mu_n
        logn = np.log(float(row["n_grasps"]))
        u_q = float(row["q_grasp"]) - linpred(
            models["eq3"], row, extra={"log_n_grasps": logn}
        )
    return {
        "U_C": u_c, "U_q": u_q, "U_e": u_e, "U_n": u_n,
        "U_e_observed": u_e_observed,
    }


def intervene(row, var):
    x = {v: float(row[v]) for v in VARS}
    x[var] = CLEAN[var]
    return x


def gt_label(primary_cause):
    pc = str(primary_cause)
    if pc in SINGLE_CAUSES:
        return pc
    if pc == "none":
        return "none"
    return "joint"


def diagnose(models, row, u_e):
    p_fact, e_fact = p_y(models, {v: float(row[v]) for v in VARS}, u_e)
    deltas = {}
    p_cfs = {}
    e_cfs = {}
    for v in VARS:
        p_cf, e_cf = p_y(models, intervene(row, v), u_e)
        deltas[v] = p_cf - p_fact
        p_cfs[v] = p_cf
        e_cfs[v] = e_cf

    max_d = max(deltas.values())
    # If nothing improves the factual prediction, the failure is irreducible
    # under the surrogate (mirrors the ground-truth "none" class).
    if max_d <= 1e-12:
        argmax = "none"
        ties = []
    else:
        ties = [v for v in VARS if abs(deltas[v] - max_d) <= 1e-12]
        argmax = ties[0]  # documented order: sigma_d, rho, phi, theta

    # Discrete GT-mirror: which single resets would flip Ŷ to 1?
    def yhat(x):
        p, e = p_y(models, x, u_e)
        return int((p_gate(models, x) > 0.5) and (e < D_TAU))

    x_fact = {v: float(row[v]) for v in VARS}
    y_fact = yhat(x_fact)
    savers = [v for v in VARS if yhat(intervene(row, v)) == 1 and y_fact == 0]
    if len(savers) == 0:
        discrete = "none" if y_fact == 0 else "none"
    elif len(savers) == 1:
        discrete = savers[0]
    else:
        discrete = "joint"

    return {
        "p_fact": p_fact,
        "e_fact_cf": e_fact,
        "argmax": argmax,
        "n_argmax_ties": len(ties) if max_d > 1e-12 else 0,
        "discrete": discrete,
        "n_savers": len(savers),
        "savers": "+".join(savers) if savers else "",
        **{f"delta_{v}": deltas[v] for v in VARS},
        **{f"p_cf_{v}": p_cfs[v] for v in VARS},
        **{f"e_cf_{v}": e_cfs[v] for v in VARS},
    }


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    k = int((y_true == y_pred).sum())
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "acc": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi}


def per_cause_acc(sub, pred_col, true_col="gt_label"):
    out = {}
    for cause, g in sub.groupby(true_col):
        out[str(cause)] = accuracy(g[true_col], g[pred_col])
    return out


def kfold_metrics(df, n_splits=5, seed=0):
    cv = df[df["causal_valid"]].reset_index(drop=True)
    rows = []

    def rmse(y, yhat):
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        return float(np.sqrt(np.mean((y - yhat) ** 2)))

    def r2(y, yhat):
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # in-sample (full 426 / 285)
    models = fit_surrogate(df)
    cpc_hat = models["eq1"].predict(sm.add_constant(cv[["phi", "theta"]]))
    gate_hat = models["eq2a"].predict(sm.add_constant(cv[VARS]))
    hg = cv[cv["has_grasps"]].copy()
    hg["log_n_grasps"] = np.log(hg["n_grasps"])
    q_hat = models["eq3"].predict(sm.add_constant(hg[VARS + ["log_n_grasps"]]))
    e_hat = models["eq4"].predict(sm.add_constant(hg[VARS]))

    in_sample = {
        "C_pc_R2": r2(cv["C_pc"], cpc_hat),
        "C_pc_RMSE": rmse(cv["C_pc"], cpc_hat),
        "has_grasps_AUC": roc_auc_score(cv["has_grasps"].astype(int), gate_hat),
        "q_grasp_R2": r2(hg["q_grasp"], q_hat),
        "q_grasp_RMSE": rmse(hg["q_grasp"], q_hat),
        "e_pose_R2": r2(hg["e_pose"], e_hat),
        "e_pose_RMSE": rmse(hg["e_pose"], e_hat),
        "e_pose_med_R2": float(models["eq4_med"].rsquared),
    }

    fold_stats = {k: [] for k in [
        "C_pc_R2", "C_pc_RMSE", "has_grasps_AUC",
        "q_grasp_R2", "q_grasp_RMSE", "e_pose_R2", "e_pose_RMSE",
    ]}
    for train_idx, test_idx in kfold_indices(len(cv), n_splits=n_splits, seed=seed):
        tr = cv.iloc[train_idx]
        te = cv.iloc[test_idx]
        tr_hg = tr[tr["has_grasps"]].copy()
        te_hg = te[te["has_grasps"]].copy()
        tr_hg["log_n_grasps"] = np.log(tr_hg["n_grasps"])
        te_hg["log_n_grasps"] = np.log(te_hg["n_grasps"])

        eq1 = sm.OLS(tr["C_pc"], sm.add_constant(tr[["phi", "theta"]])).fit()
        eq2a = sm.Logit(tr["has_grasps"].astype(int), sm.add_constant(tr[VARS])).fit(disp=0)
        eq3 = sm.OLS(tr_hg["q_grasp"], sm.add_constant(tr_hg[VARS + ["log_n_grasps"]])).fit()
        eq4 = sm.OLS(tr_hg["e_pose"], sm.add_constant(tr_hg[VARS])).fit()

        cpc = eq1.predict(sm.add_constant(te[["phi", "theta"]]))
        gate = eq2a.predict(sm.add_constant(te[VARS]))
        fold_stats["C_pc_R2"].append(r2(te["C_pc"], cpc))
        fold_stats["C_pc_RMSE"].append(rmse(te["C_pc"], cpc))
        fold_stats["has_grasps_AUC"].append(
            roc_auc_score(te["has_grasps"].astype(int), gate)
        )
        if len(te_hg) > 5:
            qh = eq3.predict(sm.add_constant(te_hg[VARS + ["log_n_grasps"]]))
            eh = eq4.predict(sm.add_constant(te_hg[VARS]))
            fold_stats["q_grasp_R2"].append(r2(te_hg["q_grasp"], qh))
            fold_stats["q_grasp_RMSE"].append(rmse(te_hg["q_grasp"], qh))
            fold_stats["e_pose_R2"].append(r2(te_hg["e_pose"], eh))
            fold_stats["e_pose_RMSE"].append(rmse(te_hg["e_pose"], eh))

    kfold = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
             for k, v in fold_stats.items() if v}
    return in_sample, kfold


def residual_correlations(df, models):
    hg = df[df["has_grasps"]].copy()
    recs = []
    for _, row in hg.iterrows():
        recs.append(abduct_row(models, row))
    u = pd.DataFrame(recs)
    cols = ["U_C", "U_n", "U_q", "U_e"]
    corr = u[cols].corr()
    return corr, u


def load_llm_primary():
    """Scored T=0 LLM attribution, if the CSV is present."""
    if not LLM_CSV.exists():
        return None
    llm = pd.read_csv(LLM_CSV)
    # Flexible column names from run_llm_baseline.py
    cols = {c.lower(): c for c in llm.columns}
    tid = cols.get("trial_id")
    # scored attribution
    pred = None
    for key in ["primary_code", "scored_code", "attribution", "pred",
                "llm_code", "parsed_code"]:
        if key in cols:
            pred = cols[key]
            break
    tier = cols.get("tier")
    temp = None
    for key in ["temperature", "temp"]:
        if key in cols:
            temp = cols[key]
            break
    if tid is None or pred is None:
        return llm, None
    sub = llm
    if tier is not None:
        sub = sub[sub[tier].astype(str).str.upper().str.contains("T1")]
    if temp is not None:
        sub = sub[pd.to_numeric(sub[temp], errors="coerce").fillna(0) <= 0.01]
    return sub[[tid, pred]].rename(columns={tid: "trial_id", pred: "llm_pred"})


def main():
    df = load_pilot()
    models = fit_surrogate(df)
    gt = pd.read_csv(GT_CSV)
    gt["gt_label"] = gt["primary_cause"].map(gt_label)
    gt["trial_id"] = gt["trial_id"].astype(int)

    # join observed mediators
    keep = df[["trial_id", "C_pc", "q_grasp", "e_pose", "n_grasps",
               "has_grasps", "success", "error"]].copy()
    keep["trial_id"] = keep["trial_id"].astype(int)
    work = gt.merge(keep, on="trial_id", how="left")

    rows = []
    for _, row in work.iterrows():
        ab = abduct_row(models, row)
        diag = diagnose(models, row, ab["U_e"])
        rec = {
            "trial_id": int(row["trial_id"]),
            "sigma_d": float(row["sigma_d"]),
            "rho": float(row["rho"]),
            "phi": float(row["phi"]),
            "theta": float(row["theta"]),
            "has_grasps": bool(row["has_grasps"]),
            "gt_label": row["gt_label"],
            "primary_cause": row["primary_cause"],
            **ab, **diag,
        }
        rec["argmax_correct"] = int(rec["argmax"] == rec["gt_label"])
        rec["discrete_correct"] = int(rec["discrete"] == rec["gt_label"])
        rows.append(rec)

    scored = pd.DataFrame(rows)
    scored.to_csv(RESULTS / "algorithm2_scored.csv", index=False)

    single = scored[scored["gt_label"].isin(SINGLE_CAUSES)].copy()
    # Pre-registered primary: argmax Δ_v on 95 single-cause trials.
    # Recode "none" argmax as incorrect (it is not the GT label).
    primary = accuracy(single["gt_label"], single["argmax"])
    primary_cause = per_cause_acc(single, "argmax")
    discrete_95 = accuracy(single["gt_label"], single["discrete"])
    discrete_292 = accuracy(scored["gt_label"], scored["discrete"])
    argmax_292 = accuracy(scored["gt_label"], scored["argmax"])

    # Confusion (argmax, 95)
    labels_95 = VARS + ["none"]
    cm95 = pd.crosstab(
        single["gt_label"], single["argmax"],
        rownames=["gt"], colnames=["scm"], dropna=False,
    ).reindex(index=VARS, columns=labels_95, fill_value=0)

    cm292 = pd.crosstab(
        scored["gt_label"], scored["discrete"],
        rownames=["gt"], colnames=["scm"], dropna=False,
    )

    in_sample, kfold = kfold_metrics(df)
    corr, _u = residual_correlations(df, models)
    corr.to_csv(RESULTS / "scm_residual_correlations.csv")

    kfold_df = pd.DataFrame(kfold).T
    kfold_df.to_csv(RESULTS / "scm_kfold_metrics.csv")

    # Pilot success by sigma_d (the 70.4% → 0.0% claim)
    cv = df[df["causal_valid"]]
    by_sd = (
        cv.groupby("sigma_d")["success"]
        .agg(n="size", k="sum", rate="mean")
        .reset_index()
    )
    by_sd["pct"] = 100 * by_sd["rate"]

    # LLM numbers from summary json (authoritative) plus per-trial if present
    llm_sum_path = RESULTS / "llm_baseline_summary.json"
    llm_sum = json.loads(llm_sum_path.read_text()) if llm_sum_path.exists() else {}
    llm_t1 = llm_sum.get("T1", {})
    llm_mean_95 = llm_t1.get("primary_accuracy")
    llm_per = llm_t1.get("primary_per_cause", {})

    h31 = primary["acc"] > 0.5
    h32 = (primary["acc"] > llm_mean_95) if llm_mean_95 is not None else None

    summary = {
        "n_failed_gt": int(len(scored)),
        "n_single_cause": int(len(single)),
        "n_joint": int((scored.gt_label == "joint").sum()),
        "n_none": int((scored.gt_label == "none").sum()),
        "primary_argmax_95": primary,
        "primary_per_cause_argmax": primary_cause,
        "discrete_95": discrete_95,
        "discrete_292": discrete_292,
        "argmax_292": argmax_292,
        "confusion_argmax_95": cm95.to_dict(),
        "confusion_discrete_292": cm292.to_dict(),
        "n_argmax_ties_95": int((single["n_argmax_ties"] > 1).sum()),
        "n_argmax_none_on_single": int((single["argmax"] == "none").sum()),
        "n_U_e_observed": int(scored["U_e_observed"].sum()),
        "in_sample": in_sample,
        "kfold_5": kfold,
        "residual_corr": corr.round(4).to_dict(),
        "pilot_success_by_sigma_d": by_sd.to_dict(orient="records"),
        "llm_t1_primary_accuracy": llm_mean_95,
        "llm_t1_per_cause": llm_per,
        "llm_t1_full_accuracy": llm_t1.get("full_accuracy"),
        "llm_t1_agreement": llm_t1.get("mean_agreement_rate"),
        "H3_1_supported": bool(h31),
        "H3_2_supported": (bool(h32) if h32 is not None else None),
        "eq4_sibling_R2": float(models["eq4"].rsquared),
        "eq4_mediation_R2": float(models["eq4_med"].rsquared),
        "eq4_sibling_RMSE_insample": in_sample["e_pose_RMSE"],
        "gate_AUC_insample": in_sample["has_grasps_AUC"],
        "D_tau": D_TAU,
        "note": (
            "Primary metric is argmax Δ_v on 95 single-cause trials. "
            "P_hat(Y=1) = p_gate * 1{e_pose_cf < 0.065}. "
            "U_e abducted iff e_pose observed; else 0. "
            "If max Δ_v <= 0, argmax is 'none'."
        ),
    }
    (RESULTS / "algorithm2_summary.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )

    print("=" * 64)
    print("Algorithm 2 scored against counterfactual_groundtruth.csv")
    print("=" * 64)
    print(f"  Failed trials:     {len(scored)}")
    print(f"  Single-cause (95): {len(single)}")
    print(f"  U_e abducted:      {int(scored['U_e_observed'].sum())} / {len(scored)}")
    print()
    print(f"  PRIMARY argmax Δ  on 95: "
          f"{primary['k']}/{primary['n']} = {100*primary['acc']:.1f}%  "
          f"(Wilson {100*primary['ci_lo']:.1f}–{100*primary['ci_hi']:.1f}%)")
    for cause in VARS:
        a = primary_cause.get(cause, {})
        if a:
            print(f"      {cause:<8} {a['k']}/{a['n']} = {100*a['acc']:.1f}%")
    print(f"  Discrete GT-mirror on 95: "
          f"{discrete_95['k']}/{discrete_95['n']} = {100*discrete_95['acc']:.1f}%")
    print(f"  Discrete GT-mirror on 292: "
          f"{discrete_292['k']}/{discrete_292['n']} = {100*discrete_292['acc']:.1f}%")
    print()
    print("  Confusion (gt \\ scm argmax) on 95:")
    print(cm95.to_string())
    print()
    print("  Confusion (gt \\ scm discrete) on 292:")
    print(cm292.to_string())
    print()
    llm_pct = 100 * llm_mean_95 if llm_mean_95 is not None else float("nan")
    print(f"  LLM T1 mean on 95: {llm_pct:.1f}%")
    print(f"  H3.1 (SCM > 50%):  {h31}")
    print(f"  H3.2 (SCM > LLM):  {h32}")
    print()
    print("  In-sample sibling e_pose R2 / RMSE:",
          f"{in_sample['e_pose_R2']:.3f} / {in_sample['e_pose_RMSE']:.4f}")
    print("  5-fold e_pose R2 mean±sd:",
          f"{kfold['e_pose_R2']['mean']:.3f} ± {kfold['e_pose_R2']['std']:.3f}")
    print("  Residual corr U_q vs U_e:",
          f"{corr.loc['U_q','U_e']:.3f}")
    print()
    print("  Pilot success by sigma_d:")
    print(by_sd.to_string(index=False))
    print()
    print("  Wrote results/algorithm2_scored.csv")
    print("  Wrote results/algorithm2_summary.json")


if __name__ == "__main__":
    main()
