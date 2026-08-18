"""
scm_nonparametric.py
=====================
Layer-2 (functional form) implementation of "Option A" from the
5 Aug 2026 SCM redesign: a nonparametric structural causal model.

The graph (Layer 1) is fixed and pre-registered in
CAUSAL_DAG_PREREGISTRATION.md, independently of this file and independently
of any result computed here. This script does NOT fit a linear or logistic
model for any structural equation. Every f in

    C_pc     = f_C(phi, theta, U_C)
    has_grasps = f_h(sigma_d, rho, phi, theta, U_h)
    n_grasps = f_n(sigma_d, rho, phi, theta, U_n)          | has_grasps=1
    q_grasp  = f_q(sigma_d, rho, phi, theta, n_grasps, U_q) | has_grasps=1
    e_pose   = f_e(sigma_d, rho, phi, theta, U_e)           | has_grasps=1
                 (NOT a function of q_grasp -- see DAG doc Sec.4)
    Y        = f_Y(has_grasps, sigma_d, rho, phi, theta, U_Y)

is left UNSPECIFIED. Because the experimental design is a complete
factorial grid over independently randomized sigma_d/rho/phi/theta,
every causal estimand this thesis needs is identified directly as a
stratified empirical statistic -- no shape assumption, nothing to defend:

  - "total effect of do(v=x)"      -> mean/rate of the outcome among rows
                                       with v=x (marginalizing the other
                                       three exogenous vars, which is valid
                                       because they are independently
                                       randomized -- conditioning IS
                                       intervening here, no adjustment set
                                       needed).
  - "path-specific effect"          -> law-of-total-probability decomposition
                                       through has_grasps (the one binary
                                       gate in the graph), using the same
                                       stratified rates on each side.
  - "moderated effect"              -> the same total-effect computation,
                                       run within strata of a second
                                       exogenous variable instead of
                                       marginalizing over it.

This is the only fully correct choice for a system whose true mechanism
(CGN's forward pass) is a deep network -- there is no reason to believe
its confidence or pose outputs are any nice parametric function of
(sigma_d, rho, phi, theta).

Usage:
    python scm_nonparametric.py

Outputs:
    results/scm_nonparametric_node_tables.csv
    results/scm_nonparametric_total_effects.csv
    results/scm_nonparametric_moderated_sigma_by_phi.csv
    results/scm_nonparametric_path_decomposition.csv
    results/scm_nonparametric_report.md
    results/figures/scm_dag_corrected.png
    results/figures/scm_nonparametric_total_effects.png
    results/figures/scm_nonparametric_moderation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_PROJECT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_PROJECT, "results", "experiment_results.csv")
FIG_DIR = os.path.join(_PROJECT, "results", "figures")
RES_DIR = os.path.join(_PROJECT, "results")
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {
    "sigma_d": "#E76F51", "rho": "#2A9D8F",
    "phi": "#457B9D", "theta": "#9B5DE5",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})
SEP = "=" * 68


# ══════════════════════════════════════════════════════════════════════
#  Nonparametric estimators (no functional form, ever)
# ══════════════════════════════════════════════════════════════════════
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion. No normal-shape
    assumption needed even at small n (unlike Wald), which matters here
    because several grid cells have only n=3 seeds."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, centre - half, centre + half


def mean_ci(x, z=1.96):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - z * (se if n > 1 else 0), m + z * (se if n > 1 else 0), n


def load(path):
    df = pd.read_csv(path)
    for c in ["sigma_d", "rho", "phi", "theta", "C_pc", "q_grasp",
              "e_pose", "n_grasps", "success"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_winerror"] = df["error"].astype(str).str.contains("WinError", na=False)
    df["causal_valid"] = ~df["is_winerror"]
    df["has_grasps"] = ((df["error"] != "no_grasps") & df["causal_valid"]).astype(int)
    return df


# ══════════════════════════════════════════════════════════════════════
#  Node-level nonparametric structural equations (stratified tables)
# ══════════════════════════════════════════════════════════════════════
def node_table_C_pc(df):
    """f_C(phi, theta): empirical distribution of C_pc per (phi, theta) cell."""
    d = df[df.causal_valid].dropna(subset=["C_pc", "phi", "theta"])
    rows = []
    for (phi, theta), g in d.groupby(["phi", "theta"]):
        m, lo, hi, n = mean_ci(g["C_pc"].values)
        rows.append(dict(node="C_pc", parents="phi,theta", phi=phi, theta=theta,
                          stat="mean", value=m, ci_lo=lo, ci_hi=hi, n=n))
    return pd.DataFrame(rows)


def node_table_has_grasps(df):
    """f_h(sigma_d, rho, phi, theta): empirical P(has_grasps=1) per full grid cell."""
    d = df[df.causal_valid].dropna(subset=["sigma_d", "rho", "phi", "theta"])
    rows = []
    for (s, r, p, t), g in d.groupby(["sigma_d", "rho", "phi", "theta"]):
        k, n = int(g["has_grasps"].sum()), len(g)
        prop, lo, hi = wilson_ci(k, n)
        rows.append(dict(node="has_grasps", parents="sigma_d,rho,phi,theta",
                          sigma_d=s, rho=r, phi=p, theta=t,
                          stat="proportion", value=prop, ci_lo=lo, ci_hi=hi, n=n))
    return pd.DataFrame(rows)


def node_table_conditional(df, col, label):
    """f(sigma_d, rho, phi, theta) for a continuous node, conditional on
    has_grasps=1 (n_grasps, q_grasp, e_pose all only exist once CGN returns
    a candidate)."""
    d = df[df.has_grasps == 1].dropna(subset=["sigma_d", "rho", "phi", "theta", col])
    rows = []
    for (s, r, p, t), g in d.groupby(["sigma_d", "rho", "phi", "theta"]):
        m, lo, hi, n = mean_ci(g[col].values)
        rows.append(dict(node=label, parents="sigma_d,rho,phi,theta | has_grasps=1",
                          sigma_d=s, rho=r, phi=p, theta=t,
                          stat="mean", value=m, ci_lo=lo, ci_hi=hi, n=n))
    return pd.DataFrame(rows)


def node_table_Y(df):
    """f_Y: empirical P(success=1) per full grid cell (unconditional --
    has_grasps=0 rows contribute their success=0 rows directly, so this
    already reflects the has_grasps gate without needing a separate term)."""
    d = df[df.causal_valid].dropna(subset=["sigma_d", "rho", "phi", "theta", "success"])
    rows = []
    for (s, r, p, t), g in d.groupby(["sigma_d", "rho", "phi", "theta"]):
        k, n = int(g["success"].sum()), len(g)
        prop, lo, hi = wilson_ci(k, n)
        rows.append(dict(node="Y", parents="has_grasps,sigma_d,rho,phi,theta",
                          sigma_d=s, rho=r, phi=p, theta=t,
                          stat="proportion", value=prop, ci_lo=lo, ci_hi=hi, n=n))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  Total effects: do(v = x), marginalizing over the other 3 exogenous vars
# ══════════════════════════════════════════════════════════════════════
def total_effect_binary(df, var, outcome, valid_mask):
    """P(outcome=1 | do(var=x)) for each level x of var, marginal over the
    other exogenous variables. Valid because they are independently
    randomized -- this is do(), not just conditioning-with-confounding."""
    d = df[valid_mask].dropna(subset=[var, outcome])
    rows = []
    for x, g in d.groupby(var):
        k, n = int(g[outcome].sum()), len(g)
        prop, lo, hi = wilson_ci(k, n)
        rows.append(dict(variable=var, level=x, outcome=outcome,
                          estimate=prop, ci_lo=lo, ci_hi=hi, n=n))
    return pd.DataFrame(rows)


def total_effects_table(df):
    exog = ["sigma_d", "rho", "phi", "theta"]
    parts = []
    for v in exog:
        parts.append(total_effect_binary(df, v, "has_grasps", df.causal_valid))
        parts.append(total_effect_binary(df, v, "success", df.causal_valid))
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
#  Path-specific decomposition through has_grasps
#  P(Y=1 | do(sigma_d=s)) = P(has_grasps=1|do(s)) * P(Y=1|has_grasps=1, do(s))
#  (exact law of total probability -- since has_grasps=0 forces Y=0, the
#  second term for has_grasps=0 is identically 0 and drops out)
# ══════════════════════════════════════════════════════════════════════
def path_decomposition(df, var="sigma_d"):
    d = df[df.causal_valid].dropna(subset=[var, "has_grasps", "success"])
    rows = []
    for x, g in d.groupby(var):
        n = len(g)
        p_has, has_lo, has_hi = wilson_ci(int(g["has_grasps"].sum()), n)
        g1 = g[g["has_grasps"] == 1]
        n1 = len(g1)
        p_y_given_has, y_lo, y_hi = wilson_ci(int(g1["success"].sum()), n1) if n1 else (0.0, np.nan, np.nan)
        p_y_total_direct, tot_lo, tot_hi = wilson_ci(int(g["success"].sum()), n)
        p_y_total_reconstructed = p_has * p_y_given_has
        rows.append(dict(
            variable=var, level=x, n=n,
            P_has_grasps=p_has,
            P_success_given_has_grasps=p_y_given_has,
            P_success_total_direct=p_y_total_direct,
            P_success_total_reconstructed=p_y_total_reconstructed,
            reconstruction_error=abs(p_y_total_direct - p_y_total_reconstructed),
        ))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  Moderated effect: does phi moderate sigma_d's effect on success?
# ══════════════════════════════════════════════════════════════════════
def moderated_effect(df, var="sigma_d", moderator="phi", outcome="success"):
    d = df[df.causal_valid].dropna(subset=[var, moderator, outcome])
    rows = []
    for (m, x), g in d.groupby([moderator, var]):
        k, n = int(g[outcome].sum()), len(g)
        prop, lo, hi = wilson_ci(k, n)
        rows.append(dict(moderator=moderator, moderator_level=m,
                          variable=var, level=x, outcome=outcome,
                          estimate=prop, ci_lo=lo, ci_hi=hi, n=n))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════
def fig_total_effects(total_df):
    exog = ["sigma_d", "rho", "phi", "theta"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="row")
    for row_i, outcome in enumerate(["has_grasps", "success"]):
        for col_i, v in enumerate(exog):
            ax = axes[row_i, col_i]
            d = total_df[(total_df.variable == v) & (total_df.outcome == outcome)].sort_values("level")
            ax.errorbar(d.level, d.estimate,
                        yerr=[d.estimate - d.ci_lo, d.ci_hi - d.estimate],
                        marker="o", color=PALETTE[v], capsize=4, lw=2)
            ax.set_title(f"do({v}) -> {outcome}", fontsize=10, fontweight="bold")
            ax.set_xlabel(v)
            if col_i == 0:
                ax.set_ylabel(f"P({outcome}=1)")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
    fig.suptitle("Nonparametric Total Effects — P(outcome | do(v)), marginal over "
                 "the other 3 randomized exogenous variables (Wilson 95% CI)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "scm_nonparametric_total_effects.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_moderation(mod_df):
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, g in mod_df.groupby("moderator_level"):
        g = g.sort_values("level")
        ax.errorbar(g.level, g.estimate,
                    yerr=[g.estimate - g.ci_lo, g.ci_hi - g.estimate],
                    marker="o", capsize=3, lw=2, label=f"phi={int(m)} deg")
    ax.set_xlabel("sigma_d (depth noise, m)")
    ax.set_ylabel("P(success=1 | do(sigma_d), do(phi))")
    ax.set_title("Nonparametric Moderated Effect:\nphi changes the shape of sigma_d's "
                 "effect on success", fontsize=11, fontweight="bold")
    ax.legend(title="Moderator")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "scm_nonparametric_moderation.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_dag_corrected():
    """Redraws the DAG with the Sec.4 correction: q_grasp and e_pose are
    siblings under a common (unobserved) node S, not a mediation chain."""
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    nodes = {
        "sigma_d": (0.7, 5.8), "rho": (0.7, 4.5),
        "phi": (0.7, 3.2), "theta": (0.7, 1.9),
        "C_pc": (3.2, 0.9),
        "S": (3.6, 3.9),
        "n_grasps": (5.8, 4.9),
        "has_grasps": (5.8, 3.2),
        "q_grasp": (7.9, 5.6),
        "e_pose": (7.9, 2.2),
        "Y": (10.4, 3.9),
    }
    labels = {
        "sigma_d": "sigma_d\n(depth noise)", "rho": "rho\n(sparsity)",
        "phi": "phi\n(elevation)", "theta": "theta\n(azimuth)",
        "C_pc": "C_pc\n(coverage)",
        "S": "S\n(CGN raw scored\ncandidates, unobs.)",
        "n_grasps": "n_grasps\n(count)",
        "has_grasps": "has_grasps\n(n_grasps>0)",
        "q_grasp": "q_grasp\n(argmax score)",
        "e_pose": "e_pose\n(pose @ same idx)",
        "Y": "Y\n(success)",
    }
    colors = {
        "sigma_d": "#E76F51", "rho": "#2A9D8F", "phi": "#457B9D", "theta": "#9B5DE5",
        "C_pc": "#A8DADC", "S": "#BFBFBF",
        "n_grasps": "#E9C46A", "has_grasps": "#F4A261",
        "q_grasp": "#264653", "e_pose": "#6B9BC3", "Y": "#2A9D8F",
    }
    dark_nodes = {"#E76F51", "#2A9D8F", "#9B5DE5", "#457B9D", "#264653"}
    r = 0.52

    for n, (x, y) in nodes.items():
        c = plt.Circle((x, y), r, color=colors[n], zorder=3, linewidth=1.5, edgecolor="#1D3557")
        ax.add_patch(c)
        tc = "white" if colors[n] in dark_nodes else "#1D3557"
        ax.text(x, y, labels[n], ha="center", va="center", fontsize=7, fontweight="bold", color=tc, zorder=4)

    def _arr(src, dst, color="#1D3557", lw=1.6, rad=0.0, style="-"):
        x0, y0 = nodes[src]; x1, y1 = nodes[dst]
        dx, dy = x1 - x0, y1 - y0
        L = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / L, dy / L
        xs, ys = x0 + ux * r, y0 + uy * r
        xe, ye = x1 - ux * r, y1 - uy * r
        cs = f"arc3,rad={rad}" if rad else "arc3,rad=0"
        ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    linestyle=style, connectionstyle=cs))

    _arr("phi", "C_pc", color=PALETTE["phi"], lw=2.0)
    _arr("theta", "C_pc", color=PALETTE["theta"], lw=2.0)
    for n in ["sigma_d", "rho", "phi", "theta"]:
        _arr(n, "S", color=PALETTE[n], lw=1.8)
    _arr("S", "n_grasps", lw=2.0)
    _arr("n_grasps", "has_grasps", lw=2.0)
    _arr("S", "q_grasp", lw=2.0, rad=0.15)
    _arr("n_grasps", "q_grasp", lw=1.4, rad=-0.15)
    _arr("S", "e_pose", lw=2.0, rad=-0.15)
    _arr("phi", "e_pose", color=PALETTE["phi"], lw=1.1, style="dashed", rad=0.3)
    _arr("theta", "e_pose", color=PALETTE["theta"], lw=1.1, style="dashed", rad=0.35)
    _arr("has_grasps", "Y", lw=2.4)
    _arr("e_pose", "Y", color="#888888", lw=1.1, style="dashed", rad=-0.1)

    # correlated-error double arrow between q_grasp and e_pose (NOT a causal edge)
    x0, y0 = nodes["q_grasp"]; x1, y1 = nodes["e_pose"]
    ax.annotate("", xy=(x1 + 0.1, y1 + r), xytext=(x0 + 0.1, y0 - r),
                arrowprops=dict(arrowstyle="<->", color="#C84B31", lw=1.6,
                                linestyle="dotted", connectionstyle="arc3,rad=0.25"))
    ax.text(9.15, 3.9, "correlated errors\n(shared idx = argmax\nof S, NOT causal)",
            ha="center", fontsize=7, color="#C84B31", fontweight="bold")

    hg = nodes["has_grasps"]
    ax.annotate("has_grasps=0\n-> Y=0 directly", xy=(hg[0]+0.3, hg[1]),
                xytext=(hg[0] + 1.3, hg[1] - 1.6), ha="center", fontsize=8,
                color="#C84B31", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C84B31", lw=1.2))

    ax.set_title(
        "Corrected Causal DAG (pre-registered 5 Aug 2026)\n"
        "q_grasp and e_pose are siblings under common parent S, not a mediation chain "
        "(see CAUSAL_DAG_PREREGISTRATION.md Sec.4)",
        fontsize=11, fontweight="bold", pad=12)

    patches = [
        mpatches.Patch(color="#E76F51", label="sigma_d: depth noise"),
        mpatches.Patch(color="#2A9D8F", label="rho: sparsity"),
        mpatches.Patch(color="#457B9D", label="phi: elevation"),
        mpatches.Patch(color="#9B5DE5", label="theta: azimuth"),
        mpatches.Patch(color="#BFBFBF", label="S: CGN raw output (unobserved compound node)"),
        mpatches.Patch(color="#F4A261", label="has_grasps: primary causal node"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7.5, framealpha=0.92, ncol=2)

    out = os.path.join(FIG_DIR, "scm_dag_corrected.png")
    plt.savefig(out, bbox_inches="tight", facecolor="#F8F9FA")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════
def write_report(df, total_df, path_df, mod_df):
    lines = []
    lines.append("# Nonparametric SCM — Layer 2 (functional forms), Option A\n")
    lines.append("Graph is pre-registered in `CAUSAL_DAG_PREREGISTRATION.md` "
                 "(5 Aug 2026), independent of this file. Every number below is a "
                 "stratified empirical statistic (mean or Wilson-score proportion) "
                 "over the complete factorial grid — no line was fit to it.\n")
    lines.append("---\n")

    lines.append("## Total effects — P(outcome | do(v)), marginal over the other "
                 "3 exogenous variables\n")
    for outcome in ["has_grasps", "success"]:
        lines.append(f"### outcome = {outcome}\n")
        lines.append("| variable | level | estimate | 95% CI | n |")
        lines.append("|---|---|---|---|---|")
        d = total_df[total_df.outcome == outcome]
        for _, r in d.iterrows():
            lines.append(f"| {r.variable} | {r.level:g} | {r.estimate:.3f} | "
                         f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {int(r.n)} |")
        lines.append("")

    lines.append("## Path decomposition through `has_grasps` (sigma_d)\n")
    lines.append("Exact identity: P(success=1|do(sigma_d)) = "
                 "P(has_grasps=1|do(sigma_d)) x P(success=1|has_grasps=1,do(sigma_d)). "
                 "`reconstruction_error` should be ~0 — it is a sanity check on the "
                 "law of total probability, not a fitted quantity.\n")
    lines.append("| sigma_d | n | P(has_grasps) | P(success\\|has_grasps=1) | "
                 "P(success) direct | P(success) reconstructed | recon. error |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in path_df.iterrows():
        lines.append(f"| {r.level:g} | {int(r.n)} | {r.P_has_grasps:.3f} | "
                     f"{r.P_success_given_has_grasps:.3f} | "
                     f"{r.P_success_total_direct:.3f} | "
                     f"{r.P_success_total_reconstructed:.3f} | "
                     f"{r.reconstruction_error:.4f} |")
    lines.append("")

    lines.append("## Moderated effect: does phi change sigma_d's effect on success?\n")
    lines.append("| phi | sigma_d | P(success) | 95% CI | n |")
    lines.append("|---|---|---|---|---|")
    for _, r in mod_df.sort_values(["moderator_level", "level"]).iterrows():
        lines.append(f"| {r.moderator_level:g} | {r.level:g} | {r.estimate:.3f} | "
                     f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}] | {int(r.n)} |")
    lines.append("")

    lines.append("---\n")
    lines.append("*Generated by `scm_nonparametric.py`. Do not hand-edit — "
                 "re-run after any change to `experiment_results.csv`.*")

    out = os.path.join(RES_DIR, "scm_nonparametric_report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{SEP}\n  Nonparametric SCM (Option A) — Layer 2 fit\n{SEP}")
    df = load(CSV_PATH)
    print(f"  Loaded {len(df)} rows ({int(df.causal_valid.sum())} causally valid, "
          f"{int(df.has_grasps.sum())} has_grasps=1)")

    print("\n-- Node-level stratified structural equations --")
    node_tables = pd.concat([
        node_table_C_pc(df),
        node_table_has_grasps(df),
        node_table_conditional(df, "n_grasps", "n_grasps"),
        node_table_conditional(df, "q_grasp", "q_grasp"),
        node_table_conditional(df, "e_pose", "e_pose"),
        node_table_Y(df),
    ], ignore_index=True)
    out = os.path.join(RES_DIR, "scm_nonparametric_node_tables.csv")
    node_tables.to_csv(out, index=False)
    print(f"  Saved: {out}  ({len(node_tables)} stratum rows across 6 nodes)")

    print("\n-- Total effects (do-calculus via randomization) --")
    total_df = total_effects_table(df)
    out = os.path.join(RES_DIR, "scm_nonparametric_total_effects.csv")
    total_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(total_df.to_string(index=False))

    print("\n-- Path-specific decomposition through has_grasps --")
    path_df = path_decomposition(df, "sigma_d")
    out = os.path.join(RES_DIR, "scm_nonparametric_path_decomposition.csv")
    path_df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(path_df.to_string(index=False))
    max_recon_err = path_df["reconstruction_error"].max()
    print(f"  Max reconstruction error across sigma_d levels: {max_recon_err:.6f} "
          f"(should be ~0 -- sanity check on law of total probability)")

    print("\n-- Moderated effect: phi x sigma_d on success --")
    mod_df = moderated_effect(df, "sigma_d", "phi", "success")
    out = os.path.join(RES_DIR, "scm_nonparametric_moderated_sigma_by_phi.csv")
    mod_df.to_csv(out, index=False)
    print(f"  Saved: {out}")

    print("\n-- Figures --")
    fig_total_effects(total_df)
    fig_moderation(mod_df)
    fig_dag_corrected()

    print("\n-- Report --")
    write_report(df, total_df, path_df, mod_df)

    print(f"\n{SEP}\n  Nonparametric SCM fit complete.\n{SEP}\n")


if __name__ == "__main__":
    main()
