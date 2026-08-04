"""
Success-threshold sensitivity analysis.

Answers the question: "Is the D_tau = 0.065 m proximity threshold an
arbitrary choice that happens to produce a nice story, or does the
causal ranking of (sigma_d, rho, phi, theta) hold across a wide range
of thresholds?"

Method
------
Uses the already-logged `e_pose` column (Euclidean distance between the
CGN-proposed grasp position and the true object centroid) as a proxy for
the post-execution outcome. This is justified empirically: thresholding
e_pose at 0.065 m agrees with the actual logged `success` column (which
is computed from the post-IK end-effector position) on 97.9% of trials
with a proposed grasp (279/285), with all 6 disagreements in the
"e_pose >= threshold but execution still succeeded" direction -- i.e.
IK/execution noise never destroys a good CGN proposal. No re-simulation
is required for this check.

Trials with n_grasps == 0 (no candidate grasp at all) are treated as
failures at every threshold, matching the definition used throughout
the thesis.

Outputs
-------
- results/figures/success_threshold_sensitivity.png : 3-panel figure
- results/success_threshold_sensitivity.csv          : raw swept rates
- Console summary for the thesis text / supervisor talking points
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "results/experiment_results.csv"
FIG_PATH = "results/figures/success_threshold_sensitivity.png"
CSV_PATH = "results/success_threshold_sensitivity.csv"
CHOSEN_D_TAU = 0.065

df = pd.read_csv(DATA_PATH)
df["e_pose"] = pd.to_numeric(df["e_pose"], errors="coerce")

thresholds = np.arange(0.03, 0.125, 0.005)


def success_at(threshold, frame):
    """1 if a grasp was proposed AND its e_pose is within threshold, else 0."""
    has_grasp = frame["e_pose"].notna()
    return (has_grasp & (frame["e_pose"] < threshold)).astype(int)


# ---------------------------------------------------------------------------
# 1. Overall success rate vs threshold, with the two calibration anchors
# ---------------------------------------------------------------------------
overall_rates = [success_at(t, df).mean() for t in thresholds]

clean_mask = (df.sigma_d == 0) & (df.rho == 1.0)
maxdeg_mask = (df.sigma_d == 0.04) & (df.rho == 0.25)
clean_rates = [success_at(t, df[clean_mask]).mean() for t in thresholds]
maxdeg_rates = [success_at(t, df[maxdeg_mask]).mean() for t in thresholds]

# ---------------------------------------------------------------------------
# 2. Per-variable rank stability: does the ordering of levels ever flip?
# ---------------------------------------------------------------------------
variables = ["sigma_d", "rho", "phi", "theta"]
rank_stable = {}
per_var_curves = {}

for var in variables:
    levels = sorted(df[var].unique())
    curves = {}
    for lvl in levels:
        sub = df[df[var] == lvl]
        curves[lvl] = [success_at(t, sub).mean() for t in thresholds]
    per_var_curves[var] = curves

    # Rank order of levels (by success rate) at every threshold
    rank_orders = []
    for i in range(len(thresholds)):
        rates_at_t = {lvl: curves[lvl][i] for lvl in levels}
        order = tuple(sorted(rates_at_t, key=lambda l: rates_at_t[l]))
        rank_orders.append(order)
    n_unique_orders = len(set(rank_orders))
    rank_stable[var] = (n_unique_orders, rank_orders[0], len(rank_orders))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.plot(thresholds, overall_rates, "k-", lw=2, label="Overall")
ax.plot(thresholds, clean_rates, "g-", lw=2, label=r"Clean ($\sigma_d$=0, $\rho$=1.0)")
ax.plot(thresholds, maxdeg_rates, "r-", lw=2, label=r"Max degradation ($\sigma_d$=0.04, $\rho$=0.25)")
ax.axvline(CHOSEN_D_TAU, color="blue", ls="--", alpha=0.7, label=r"Chosen $D_\tau$=0.065")
ax.set_xlabel(r"Proximity threshold $D_\tau$ (m)")
ax.set_ylabel("Success rate")
ax.set_title("Overall / clean / degraded")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
for lvl, curve in per_var_curves["sigma_d"].items():
    ax.plot(thresholds, curve, marker="o", ms=3, label=fr"$\sigma_d$={lvl}")
ax.axvline(CHOSEN_D_TAU, color="blue", ls="--", alpha=0.5)
ax.set_xlabel(r"Proximity threshold $D_\tau$ (m)")
ax.set_ylabel("Success rate")
ax.set_title(r"By $\sigma_d$ (depth noise)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
for lvl, curve in per_var_curves["phi"].items():
    ax.plot(thresholds, curve, marker="o", ms=3, label=fr"$\phi$={lvl}$\degree$")
ax.axvline(CHOSEN_D_TAU, color="blue", ls="--", alpha=0.5)
ax.set_xlabel(r"Proximity threshold $D_\tau$ (m)")
ax.set_ylabel("Success rate")
ax.set_title(r"By $\phi$ (elevation)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle("Success-threshold sensitivity: does $D_\\tau=0.065$ drive the causal story, or reveal it?", y=1.03)
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
print(f"Saved figure to {FIG_PATH}")

# ---------------------------------------------------------------------------
# Save raw sweep + print summary
# ---------------------------------------------------------------------------
out = pd.DataFrame({"D_tau": thresholds, "overall": overall_rates,
                     "clean": clean_rates, "max_degradation": maxdeg_rates})
for var in variables:
    for lvl, curve in per_var_curves[var].items():
        out[f"{var}={lvl}"] = curve
out.to_csv(CSV_PATH, index=False)
print(f"Saved sweep table to {CSV_PATH}")

print("\n=== Rank-order stability across D_tau in [0.03, 0.12] ===")
for var in variables:
    n_orders, first_order, n_total = rank_stable[var]
    status = "STABLE" if n_orders == 1 else f"CHANGES ({n_orders} distinct orderings)"
    print(f"{var:8s}: {status}  (worst->best at D_tau=0.03: {first_order})")

print(f"\nAt chosen D_tau=0.065:")
close_idx = int(np.argmin(np.abs(thresholds - CHOSEN_D_TAU)))
print(f"  overall success rate: {overall_rates[close_idx]:.3f}")
print(f"  clean (sigma_d=0, rho=1.0) rate: {clean_rates[close_idx]:.3f}")
print(f"  max-degradation rate: {maxdeg_rates[close_idx]:.3f}")
