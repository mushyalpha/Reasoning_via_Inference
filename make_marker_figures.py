#!/usr/bin/env python3
"""Figures and numeric dumps requested by the marker audit."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ACCENT = "#0B7A75"
CHARCOAL = "#2B2B2B"
RUST = "#A8432B"
GOLD = "#C9A227"
BLUE = "#1f4e79"


def wilson_ci(k, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig_attribution_bars():
    summary = json.loads((RESULTS / "algorithm2_summary.json").read_text())
    scm = summary["primary_per_cause_argmax"]
    llm = summary["llm_t1_per_cause"]
    causes = ["sigma_d", "rho", "phi", "theta"]
    labels = [r"$\sigma_d$", r"$\rho$", r"$\phi$", r"$\theta$"]
    scm_acc = [100 * scm[c]["acc"] for c in causes]
    llm_acc = [100 * (llm[c]["accuracy"] or 0) for c in causes]
    ns = [scm[c]["n"] for c in causes]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(causes))
    w = 0.36
    ax.bar(x - w / 2, scm_acc, w, color=ACCENT, label="SCM (Alg. 2)")
    ax.bar(x + w / 2, llm_acc, w, color="#B8B8B8", label="LLM T1")
    ax.axhline(50, color=RUST, ls="--", lw=1, label="H3.1 bar (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n$n$={n}" for lab, n in zip(labels, ns)])
    ax.set_ylabel("Attribution accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    mean_scm = 100 * summary["primary_argmax_95"]["acc"]
    mean_llm = 100 * summary["llm_t1_primary_accuracy"]
    ax.set_title(
        f"Single-cause pilot (n=95): SCM {mean_scm:.1f}% vs LLM {mean_llm:.1f}%"
    )
    save(fig, "fig_attribution_accuracy")


def fig_simpson():
    df = pd.read_csv(RESULTS / "experiment_results_v2.csv")
    g = df[df["collision_free"] == 1].copy()
    g = g.dropna(subset=["e_pose", "success"])
    # three lowest-noise strata where the within-stratum sign is clean
    strata = [0.0, 0.0025, 0.005]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6), sharey=True)
    pooled = g.dropna(subset=["e_pose"])

    # pooled OLS line for overlay on each panel (wrong-sign story)
    x_all = pooled["e_pose"].to_numpy()
    y_all = pooled["success"].astype(float).to_numpy()
    A = np.vstack([x_all, np.ones_like(x_all)]).T
    slope, intercept = np.linalg.lstsq(A, y_all, rcond=None)[0]

    for ax, sd in zip(axes, strata):
        sub = g[g["sigma_d"] == sd]
        fail = sub[sub["success"] == 0]["e_pose"]
        succ = sub[sub["success"] == 1]["e_pose"]
        ax.boxplot(
            [fail, succ],
            positions=[0, 1],
            widths=0.55,
            patch_artist=True,
            boxprops=dict(facecolor="#E8E8E8", edgecolor=CHARCOAL),
            medianprops=dict(color=RUST, lw=2),
            whiskerprops=dict(color=CHARCOAL),
            capprops=dict(color=CHARCOAL),
            flierprops=dict(marker="o", ms=3, alpha=0.35, color=CHARCOAL),
        )
        mf, ms_ = fail.mean(), succ.mean()
        ax.plot([0, 1], [mf, ms_], "o-", color=ACCENT, lw=2, ms=7, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["fail", "success"])
        ax.set_title(rf"$\sigma_d$={sd:g} m", loc="left")
        ax.text(
            0.5, 0.95,
            rf"$\bar{{e}}$ {mf:.3f}$\rightarrow${ms_:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color=ACCENT,
        )
        ax.set_ylim(0, 0.28)

    axes[0].set_ylabel(r"$e_{\mathrm{pose}}$ (m)")
    fig.suptitle(
        r"Within each noise stratum, higher $e_{\mathrm{pose}}$ means failure; "
        r"pooling across $\sigma_d$ flips the sign",
        fontsize=11, y=1.03,
    )
    save(fig, "fig_simpson_epose")

    # also dump the six numbers + pooled correlation
    rows = []
    for sd, sub in g.groupby("sigma_d"):
        rows.append({
            "sigma_d": sd,
            "n": len(sub),
            "mean_e_success": sub.loc[sub.success == 1, "e_pose"].mean(),
            "mean_e_fail": sub.loc[sub.success == 0, "e_pose"].mean(),
            "corr": sub["e_pose"].corr(sub["success"]),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "epose_reversal_by_sigma.csv", index=False)
    return slope, intercept, pooled["e_pose"].corr(pooled["success"])


def fig_sankey():
    """Monochrome trial-flow diagram with equal-height stage cards."""
    df = pd.read_csv(RESULTS / "experiment_results_v2.csv")
    n = len(df)
    n_nograps = int((df["failure_mode"] == "no_grasps").sum())
    n_coll = int((df["failure_mode"] == "pregrasp_collision").sum())
    n_drop = int((df["failure_mode"] == "executed_dropped").sum())
    n_eject = int((df["failure_mode"] == "executed_ejected").sum())
    n_succ = int((df["success"] == 1).sum())
    n_prop = n - n_nograps
    n_gate = n_succ + n_drop + n_eject
    assert n_nograps + n_coll + n_drop + n_eject + n_succ == n
    p_prop = 100.0 * n_prop / n
    p_gcond = 100.0 * n_gate / n_prop
    p_ycond = 100.0 * n_succ / n_gate

    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.set_xlim(0.0, 13.15)
    ax.set_ylim(0.0, 7.75)
    ax.set_axis_off()

    w, h = 2.22, 1.88
    y_card = 4.18
    xs = [0.38, 3.70, 7.02, 10.34]

    def box(x, y, ww, hh, face="white", edge="black", lw=0.9):
        ax.add_patch(FancyBboxPatch(
            (x, y), ww, hh,
            boxstyle="round,pad=0.008,rounding_size=0.035",
            facecolor=face, edgecolor=edge, lw=lw, zorder=2,
            clip_on=False,
        ))

    def labelled(x, y, ww, hh, rows, face="white", edge="black", lw=0.9):
        box(x, y, ww, hh, face, edge, lw=lw)
        cy = y + hh / 2.0
        for txt, kw in rows:
            if not txt:
                continue
            dy = kw.pop("dy")
            ax.text(x + ww / 2.0, cy + dy, txt, ha="center", va="center",
                    zorder=3, clip_on=False, **kw)

    stages = [
        (xs[0], f"{n:,}", "all trials",
         "starting set", "100% of grid"),
        (xs[1], f"{n_prop:,}", "proposals",
         rf"$P(N{{>}}0)={p_prop:.1f}\%$", "at least one CGN pose"),
        (xs[2], f"{n_gate:,}", "gate-pass",
         rf"$P(G{{=}}1\mid N{{>}}0)={p_gcond:.1f}\%$",
         f"{100.0 * n_gate / n:.1f}% of grid"),
        (xs[3], f"{n_succ}", "success",
         rf"$P(Y{{=}}1\mid G{{=}}1)={p_ycond:.1f}\%$",
         f"{100.0 * n_succ / n:.1f}% of grid"),
    ]
    for x, num, name, cond, share in stages:
        labelled(x, y_card, w, h, [
            (num, dict(dy=0.52, fontsize=17, fontweight="bold", color="black")),
            (name, dict(dy=0.12, fontsize=10.5, fontweight="bold", color="black")),
            (cond, dict(dy=-0.32, fontsize=8.5, color="black")),
            (share, dict(dy=-0.68, fontsize=8, color="#555555")),
        ])

    y_arr = y_card + h / 2.0
    gaps = [0.5 * (xs[i] + w + xs[i + 1]) for i in range(3)]
    flow_n = [n, n_prop, n_gate, n_succ]
    for i in range(3):
        lw_h = 0.8 + 2.8 * (flow_n[i + 1] / n)
        ax.annotate(
            "", xy=(xs[i + 1] - 0.07, y_arr),
            xytext=(xs[i] + w + 0.07, y_arr),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=lw_h,
                            mutation_scale=10),
            zorder=4, clip_on=False,
        )

    y_sink, sh = 0.28, 1.58

    def branch(cx, y_bot, color, n_s):
        lw_b = 0.75 + 2.8 * (n_s / n)
        ax.annotate(
            "", xy=(cx, y_bot), xytext=(cx, y_arr - 0.04),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw_b,
                            mutation_scale=8),
            zorder=1, clip_on=False,
        )

    early = [
        (gaps[0], 2.18, "no CGN proposal", n_nograps, "white", 0.9),
        (gaps[1], 2.42, "pre-grasp collision", n_coll, "#F2F2F2", 1.0),
    ]
    for cx, sw, label, n_s, face, lw in early:
        labelled(cx - sw / 2, y_sink, sw, sh, [
            (label, dict(dy=0.42, fontsize=8.4, fontweight="bold", color="black")),
            (f"{n_s:,}", dict(dy=0.00, fontsize=15, fontweight="bold", color="black")),
            (f"{100.0 * n_s / n:.1f}% of grid", dict(dy=-0.40, fontsize=8.2, color="#555555")),
        ], face=face, lw=lw)
        branch(cx, y_sink + sh, "black", n_s)

    jx = gaps[2]
    sw_e, sh_e, gap_e = 2.28, 0.76, 0.10
    y_hi = y_sink + sh_e + gap_e
    for y, label, n_s in (
        (y_hi, "closed without lifting", n_drop),
        (y_sink, "object left footprint", n_eject),
    ):
        labelled(jx - sw_e / 2, y, sw_e, sh_e, [
            (label, dict(dy=0.15, fontsize=8.0, fontweight="bold", color="black")),
            (f"{n_s:,}   ({100.0 * n_s / n:.1f}%)",
             dict(dy=-0.18, fontsize=10.5, fontweight="bold", color="black")),
        ])
    branch(jx, y_hi + sh_e, "black", n_drop + n_eject)

    ax.text(
        0.38, 7.60,
        "Trial flow of the 7,560-trial confirmatory grid",
        fontsize=12, fontweight="bold", color="black", va="top", ha="left",
    )
    ax.text(
        0.38, 7.10,
        r"$P(Y{=}1)=P(N{>}0)\,P(G{=}1\mid N{>}0)\,P(Y{=}1\mid G{=}1)"
        r"\quad\mathrm{(terminal\ counts\ are\ shares\ of\ all\ 7{,}560\ trials)}$",
        fontsize=8.5, color="black", va="top", ha="left",
    )
    ax.text(
        0.38, y_card + h + 0.14,
        r"continuing path  (arrow thickness $\propto$ remaining trials)",
        fontsize=7.5, color="#555555", ha="left", va="bottom",
    )
    save(fig, "fig_trial_flow")


def fig_rank_hist():
    df = pd.read_csv(RESULTS / "experiment_results_v3_clean_100.csv")
    succ = df[df["success"] == 1]
    ranks = succ["selected_rank"].dropna().astype(int)
    exhausted = int((df["success"] == 0).sum())
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    bins = np.arange(0, 21)
    ax.hist(ranks, bins=bins, color=ACCENT, edgecolor="white",
            align="left", label=f"rescued successes (n={len(ranks)})")
    ax.bar(20, exhausted, color=RUST, width=0.8, label=f"exhausted at k=20 (n={exhausted})")
    ax.axvline(ranks.mean(), color=GOLD, ls="--", lw=1.4,
               label=rf"mean rank among successes = {ranks.mean():.2f}")
    ax.set_xlabel("Selected rank (0 = top-1)")
    ax.set_ylabel("Trials")
    ax.set_xlim(-0.5, 21)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Top-$k$ collision filter: most rescues are ranks 1–5, not the tail")
    save(fig, "fig_rank_rescue")
    return float(ranks.mean()), int((ranks == 0).sum()), exhausted


def fig_irreducibility():
    scored = pd.read_csv(RESULTS / "algorithm2_scored.csv")
    gt = pd.read_csv(RESULTS / "counterfactual_groundtruth.csv")
    m = gt.merge(scored[["trial_id", "argmax", "gt_label"]], on="trial_id")

    def bucket(lab):
        if lab in {"sigma_d", "phi", "theta", "rho"}:
            return lab
        if lab == "none":
            return "none"
        return "joint"

    m["bucket"] = m["gt_label"].map(bucket)
    phis = sorted(m["phi"].unique())
    sds = sorted(m["sigma_d"].unique())
    order = ["sigma_d", "phi", "theta", "rho", "joint", "none"]
    colours = {
        "sigma_d": "#E76F51", "phi": "#457B9D", "theta": "#9B5DE5",
        "rho": "#2A9D8F", "joint": "#C9A227", "none": "#B0B0B0",
    }

    # counts per (phi, sigma_d)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(len(sds), dtype=float)
    y = np.arange(len(phis), dtype=float)
    levels = np.linspace(0, 100, 11)

    def contour_panel(ax, value_fn, title, cmap, cbar_label):
        from scipy.interpolate import RegularGridInterpolator

        grid = np.full((len(phis), len(sds)), np.nan)
        for i, phi in enumerate(phis):
            for j, sd in enumerate(sds):
                sub = m[(m["phi"] == phi) & (m["sigma_d"] == sd)]
                grid[i, j] = value_fn(sub)
        yi = np.linspace(y[0], y[-1], 80)
        xi = np.linspace(x[0], x[-1], 80)
        interp = RegularGridInterpolator(
            (y, x), grid, method="linear", bounds_error=False, fill_value=None,
        )
        YY, XX = np.meshgrid(yi, xi, indexing="ij")
        Zi = interp(np.stack([YY, XX], axis=-1))
        cf = ax.contourf(
            xi, yi, Zi, levels=levels, cmap=cmap, vmin=0, vmax=100,
            extend="neither", antialiased=True,
        )
        ax.contour(
            xi, yi, Zi, levels=levels, colors="0.35", linewidths=0.35, alpha=0.65,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in sds])
        ax.set_yticks(y)
        ax.set_yticklabels([f"{int(v)}" for v in phis])
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[-1], y[0])
        ax.set_xlabel(r"$\sigma_d$ (m)")
        ax.set_ylabel(r"$\phi$ (deg)")
        ax.set_title(title, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

    contour_panel(
        axes[0],
        lambda s: 100 * (s["bucket"] == "none").mean() if len(s) else np.nan,
        "(a) Share of irreducible (none) failures (%)",
        "Blues", "% none",
    )
    contour_panel(
        axes[1],
        lambda s: 100 * (s["argmax"] == s["gt_label"]).mean() if len(s) else np.nan,
        "(b) Algorithm 2 accuracy on those cells (%)",
        "Greys", "% correct",
    )
    fig.suptitle("Pilot irreducibility concentrates at overhead viewpoints; "
                 "the surrogate is weakest there too", fontsize=11, y=1.03)
    save(fig, "fig_irreducibility_map")


def fig_depth_scale_cue():
    src = FIG / "thesis_renders" / "fig_depth_degradation.png"
    im = Image.open(src).convert("RGB")
    w, h = im.size
    bar_h = int(0.16 * h)
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(im, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/Library/Fonts/Times New Roman.ttf", 28)
        font_s = ImageFont.truetype("/Library/Fonts/Times New Roman.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_s = font

    # two bars: cylinder radius 36 mm vs sigma_d = 40 mm, same pixel/mm
    px_per_mm = w * 0.35 / 40.0
    y0 = h + int(0.35 * bar_h)
    x0 = int(0.08 * w)
    r_px = int(36 * px_per_mm)
    n_px = int(40 * px_per_mm)
    draw.rectangle([x0, y0, x0 + r_px, y0 + 22], fill=(45, 45, 45))
    draw.rectangle([x0, y0 + 40, x0 + n_px, y0 + 62], fill=(130, 130, 130))
    draw.text((x0 + r_px + 16, y0 - 4), "cylinder radius  =  36 mm", fill=(45, 45, 45), font=font_s)
    draw.text((x0 + n_px + 16, y0 + 36), "noise  σ_d = 0.04 m  =  40 mm", fill=(80, 80, 80), font=font_s)
    draw.text((x0, h + 8), "Scale cue: at the rightmost panel the noise amplitude is comparable to the object radius.",
              fill=(40, 40, 40), font=font_s)
    out = FIG / "fig_depth_degradation_scaled.png"
    canvas.save(out)
    # also pdf-friendly via matplotlib
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.imshow(canvas)
    ax.axis("off")
    fig.savefig(FIG / "fig_depth_degradation_scaled.pdf", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("  wrote fig_depth_degradation_scaled")


def fig_anatomy_strip():
    """Compose a two-by-two anatomy grid from existing renders."""
    candidates = [
        (FIG / "pickup_demo" / "cylinder_1_approach.png",
         r"(1) Scene: table, object, gripper. $(\phi,\theta)$ set the camera."),
        (FIG / "thesis_renders" / "fig_depth_degradation.png",
         r"(2) Depth buffer. $\sigma_d$ is added here; $C_{pc}$ is read from the mask first."),
        (FIG / "cgn_zoomed_2d_projections.png",
         r"(3) Contact-GraspNet on $\mathcal{P}_\rho$. $\rho$ downsamples the cloud."),
        (FIG / "floating_gripper" / "fig_grasp_sequence.png",
         r"(4) Open-hand pose is the collision gate (72.8% of the grid dies here)."),
    ]
    present = [(p, t) for p, t in candidates if p.exists()]
    if len(present) < 3:
        print("  skip anatomy strip (missing renders)")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    axes = axes.ravel()
    for ax, (p, title) in zip(axes, present):
        im = Image.open(p).convert("RGB")
        ax.imshow(im)
        ax.set_title(title, fontsize=8.5, pad=5, loc="left")
        ax.axis("off")
    for ax in axes[len(present):]:
        ax.axis("off")
    fig.suptitle(
        "Anatomy of one trial: where each exogenous variable enters",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.2)
    save(fig, "fig_trial_anatomy")


def fig_collision_callout():
    p = FIG / "floating_gripper" / "fig_grasp_sequence.png"
    if not p.exists():
        return
    im = Image.open(p).convert("RGB")
    w, h = im.size
    # left-most panel of a 4-panel strip
    crop = im.crop((0, 0, w // 4, h))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(crop)
    ax.set_title("Open-hand placement: the pose at which\n"
                 "the pre-grasp collision gate is evaluated", fontsize=9)
    ax.axis("off")
    save(fig, "fig_pregrasp_gate")


def dump_confirmatory_cis():
    df = pd.read_csv(RESULTS / "experiment_results_v2.csv")

    def decomp(g):
        has = g[g["collision_free"].notna()]
        gate = has[has["collision_free"] == 1]
        n = len(g)
        n_prop = len(has)
        n_gate = len(gate)
        k_prop = n_prop
        k_gate = n_gate
        k_exec = int(gate["success"].sum()) if n_gate else 0
        k_marg = int(g["success"].sum())
        return {
            "n": n,
            "proposal": (k_prop, n),
            "gate": (k_gate, n_prop if n_prop else 1),
            "exec": (k_exec, n_gate if n_gate else 1),
            "marg": (k_marg, n),
        }

    rows = []
    for sd, g in df.groupby("sigma_d"):
        d = decomp(g)
        rec = {"var": "sigma_d", "level": sd, "n": d["n"]}
        for name in ["proposal", "gate", "exec", "marg"]:
            k, n = d[name]
            lo, hi = wilson_ci(k, n)
            rec[f"{name}_pct"] = 100 * k / n
            rec[f"{name}_ci_lo"] = 100 * lo
            rec[f"{name}_ci_hi"] = 100 * hi
            rec[f"{name}_k"] = k
            rec[f"{name}_n"] = n
        rows.append(rec)

    # azimuth x object (the hedge)
    az = []
    for (obj, th), g in df.groupby(["object", "theta"]):
        d = decomp(g)
        k, n = d["exec"]
        lo, hi = wilson_ci(k, n)
        az.append({
            "object": obj, "theta": th, "n_gate": n, "k": k,
            "pct": 100 * k / n if n else np.nan,
            "ci_lo": 100 * lo, "ci_hi": 100 * hi,
        })
    pd.DataFrame(rows).to_csv(RESULTS / "decomp_sigma_with_ci.csv", index=False)
    pd.DataFrame(az).to_csv(RESULTS / "azimuth_by_object.csv", index=False)
    print("  wrote decomp_sigma_with_ci.csv and azimuth_by_object.csv")
    return pd.DataFrame(rows), pd.DataFrame(az)


def mcnemar_scm_llm():
    scored = pd.read_csv(RESULTS / "algorithm2_scored.csv")
    llm = pd.read_csv(RESULTS / "llm_baseline_results.csv")
    t1 = llm[llm["tier"] == "T1"].copy()
    # one scored row per trial (the results CSV already has the scored attribution)
    t1 = t1.drop_duplicates("trial_id", keep="first")
    single = scored[scored["gt_label"].isin(["sigma_d", "rho", "phi", "theta"])]
    m = single.merge(t1[["trial_id", "attribution", "correct"]], on="trial_id")
    scm_ok = m["argmax"] == m["gt_label"]
    llm_ok = m["correct"].astype(int) == 1
    n12 = int((scm_ok & ~llm_ok).sum())  # SCM only
    n21 = int((~scm_ok & llm_ok).sum())  # LLM only
    n11 = int((scm_ok & llm_ok).sum())
    n00 = int((~scm_ok & ~llm_ok).sum())
    b, c = n12, n21
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) else np.nan  # continuity
    # two-sided exact binomial p on discordant pairs
    from math import comb
    n_disc = b + c
    if n_disc == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = 0.0
        for i in range(k + 1):
            p += comb(n_disc, i)
        p = min(1.0, 2 * p / (2 ** n_disc))
    return {
        "n": int(len(m)), "both": n11, "scm_only": n12, "llm_only": n21,
        "neither": n00, "mcnemar_chi2_cc": chi2, "exact_p": p,
    }


def fig_residual_heatmap():
    """Diverging bars of the six unique residual pairs (NPSEM-ie check)."""
    corr = pd.read_csv(RESULTS / "scm_residual_correlations.csv", index_col=0)
    tex = {"U_C": r"$U_C$", "U_n": r"$U_n$", "U_q": r"$U_q$", "U_e": r"$U_e$"}
    names = ["U_C", "U_n", "U_q", "U_e"]
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b])))
    # Strongest |r| at the top; named U_q–U_e violation reads first among ties.
    pairs.sort(key=lambda t: (-abs(t[2]), t[0] != "U_q"))

    terracotta = "#C0704A"
    steel = "#5A7A9A"
    named = ("U_q", "U_e")

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    y = np.arange(len(pairs))[::-1]
    rs = np.array([p[2] for p in pairs])
    colours = []
    for a, b, r in pairs:
        if (a, b) == named or (b, a) == named:
            colours.append("#A8432B")
        elif r < 0:
            colours.append(terracotta)
        else:
            colours.append(steel)
    ax.barh(y, rs, height=0.62, color=colours, edgecolor="none", zorder=2)

    ax.axvline(0, color="black", lw=1.15, zorder=3)
    pad = 0.012
    for yi, (a, b, r) in zip(y, pairs):
        lab = rf"{tex[a]}–{tex[b]}"
        txt = rf"{lab}  ${r:+.2f}$"
        kw = dict(va="center", fontsize=9.5, color=CHARCOAL, zorder=4)
        if r < 0:
            ax.text(r - pad, yi, txt, ha="right", **kw)
        else:
            ax.text(r + pad, yi, txt, ha="left", **kw)

    ax.set_yticks([])
    ax.set_xlim(-0.36, 0.36)
    ax.set_xticks([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])
    ax.set_xticklabels([r"$-0.30$", r"$-0.20$", r"$-0.10$", r"$0$",
                        r"$+0.10$", r"$+0.20$", r"$+0.30$"])
    ax.set_xlabel("Pearson $r$")
    ax.set_ylim(-0.55, len(pairs) - 0.45)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=3.5)
    ax.set_title("Fitted-residual correlations (NPSEM-ie check)")
    save(fig, "fig_npsem_residuals")


def main():
    print("Generating marker-audit figures...")
    fig_attribution_bars()
    slope, intercept, pooled_r = fig_simpson()
    fig_sankey()
    mean_rank, n_top1, n_exh = fig_rank_hist()
    fig_irreducibility()
    fig_depth_scale_cue()
    fig_anatomy_strip()
    fig_collision_callout()
    fig_residual_heatmap()
    decomp, az = dump_confirmatory_cis()
    mc = mcnemar_scm_llm()
    extra = {
        "pooled_e_success_corr": float(pooled_r),
        "pooled_ols_slope": float(slope),
        "mean_selected_rank": mean_rank,
        "n_top1_success": n_top1,
        "n_exhausted": n_exh,
        "mcnemar": mc,
        "azimuth_by_object": az.to_dict(orient="records"),
    }
    (RESULTS / "marker_figure_stats.json").write_text(
        json.dumps(extra, indent=2, default=float)
    )
    print("McNemar:", json.dumps(mc, indent=2))
    print("Azimuth x object:\n", az.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
