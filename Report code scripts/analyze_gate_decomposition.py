"""
Gate/execution decomposition tables for the v2 floating-gripper dataset.

Each trial passes through three stages, so marginal success factorises as

    P(Y=1) = P(has grasps) . P(gate pass | has grasps) . P(Y=1 | gate pass)

where the "gate" is the open-hand collision check on the selected pose.
Reporting the factors separately distinguishes variables that act by making
poses geometrically invalid from variables that act by making valid poses
physically bad; a variable can move the two in opposite directions and leave
the marginal almost unchanged.

Outputs machine-readable tables for the results chapter plus the numbers
quoted in prose. Figures are produced by plot_gate_decomposition.py.

Input:  results/experiment_results_v2.csv
Output: results/gate_decomposition_tables.csv
        results/gate_intermediate_by_sigma.csv
"""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import numpy as np
import pandas as pd

CSV = "results/experiment_results_v2.csv"
OUT_MAIN = "results/gate_decomposition_tables.csv"
OUT_INTER = "results/gate_intermediate_by_sigma.csv"

FACTORS = ["object", "sigma_d", "rho", "phi", "theta"]


def wilson(k, n, z=1.96):
    """Wilson score interval; degrades gracefully at k=0 and k=n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))


def decompose(df, key):
    rows = []
    for level, g in df.groupby(key, sort=True):
        has = g[g["collision_free"].notna()]
        gate = has[has["collision_free"] == 1]
        k_succ = int(gate["success"].sum())
        lo_pg, hi_pg = wilson(k_succ, len(gate))
        lo_m, hi_m = wilson(int(g["success"].sum()), len(g))
        rows.append({
            "variable": key,
            "level": level,
            "n": len(g),
            "has_grasps_pct": 100 * len(has) / len(g),
            "n_has_grasps": len(has),
            "gate_pass_pct": 100 * len(gate) / len(has) if len(has) else np.nan,
            "n_gate_pass": len(gate),
            "post_gate_pct": 100 * k_succ / len(gate) if len(gate) else np.nan,
            "post_gate_lo": lo_pg,
            "post_gate_hi": hi_pg,
            "marginal_pct": 100 * g["success"].mean(),
            "marginal_lo": lo_m,
            "marginal_hi": hi_m,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CSV)
    n = len(df)

    print(f"=== Dataset: {n:,} trials")
    design = {k: sorted(df[k].unique()) for k in FACTORS}
    cells = df.groupby(FACTORS).size()
    print("  design:", {k: len(v) for k, v in design.items()},
          f"x {cells.iloc[0]} seeds = {cells.size * cells.iloc[0]:,}")
    print("  cells all equal size:", cells.nunique() == 1)

    print("\n=== Outcome composition")
    comp = df["failure_mode"].value_counts()
    for mode, k in comp.items():
        print(f"  {mode:<20} {k:>5}  {100 * k / n:5.1f}%")

    has = df[df["collision_free"].notna()]
    gate = has[has["collision_free"] == 1]
    print(f"\n=== Stage yields")
    print(f"  has grasps            {len(has):>5}/{n} = {100 * len(has) / n:.1f}%")
    print(f"  gate pass | grasps    {len(gate):>5}/{len(has)} = "
          f"{100 * len(gate) / len(has):.1f}%")
    print(f"  success   | gate pass {int(gate['success'].sum()):>5}/{len(gate)} = "
          f"{100 * gate['success'].mean():.1f}%  "
          f"CI {wilson(int(gate['success'].sum()), len(gate))[0]:.1f}-"
          f"{wilson(int(gate['success'].sum()), len(gate))[1]:.1f}")
    print(f"  marginal success      {int(df['success'].sum()):>5}/{n} = "
          f"{100 * df['success'].mean():.1f}%")

    tables = pd.concat([decompose(df, k) for k in FACTORS], ignore_index=True)
    tables.to_csv(OUT_MAIN, index=False)

    for k in FACTORS:
        t = tables[tables["variable"] == k]
        print(f"\n=== By {k}")
        print(t[["level", "n", "has_grasps_pct", "gate_pass_pct",
                 "post_gate_pct", "marginal_pct"]]
              .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # The clean-condition gap that motivates the audit.
    clean = df[df["sigma_d"] == 0.0]
    clean_has = clean[clean["collision_free"].notna()]
    coll = (clean_has["collision_free"] == 0).mean()
    print(f"\n=== Clean condition (sigma_d = 0), no perceptual degradation")
    print(f"  candidate-producing trials whose top-1 pose collides: {100 * coll:.1f}%")
    print(f"  marginal success: {100 * clean['success'].mean():.1f}%")

    # Does noise buy gate yield at the cost of pose quality?
    inter = has.groupby("sigma_d").agg(
        n=("success", "size"),
        C_pc=("C_pc", "mean"),
        q_grasp=("q_grasp", "mean"),
        e_pose=("e_pose", "mean"),
        e_pose_sd=("e_pose", "std"),
        n_grasps=("n_grasps", "mean"),
    ).reset_index()
    inter.to_csv(OUT_INTER, index=False)
    print("\n=== Intermediate variables by sigma_d (candidate-producing trials)")
    print(inter.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Object x noise, the interaction that dominates everything else.
    print("\n=== Post-gate success (%) by object x sigma_d")
    piv = (gate.groupby(["object", "sigma_d"])["success"]
           .agg(["mean", "size"]).reset_index())
    piv["pct"] = 100 * piv["mean"]
    print(piv.pivot(index="object", columns="sigma_d", values="pct")
          .to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n  n per cell:")
    print(piv.pivot(index="object", columns="sigma_d", values="size")
          .to_string(float_format=lambda v: f"{v:.0f}"))

    print(f"\nWrote {OUT_MAIN}, {OUT_INTER}")


if __name__ == "__main__":
    main()
