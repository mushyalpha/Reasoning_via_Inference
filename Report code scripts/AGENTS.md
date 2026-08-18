# Agent Instructions — Reasoning via Inference (MSc Project)

Shared rules for Cursor, Antigravity, and other agents working in this repo.

---

## Visual Progress Tracking

After any session where **major progress** is made, proactively remind the user to create a visual and add it to their supervisor slides. Do not wait until the end of the project.

### What counts as major progress

- A new experiment batch completes or results are updated in `results/experiment_results.csv`
- A new figure or visualisation is generated in `results/figures/`
- A new component of the causal pipeline is implemented (SCM fitting, counterfactual diagnosis, LLM baseline)
- A significant bug is fixed that changes experimental outcomes
- A new analysis script is written or a key function is added to the codebase

### What to say when this triggers

At the end of any response where major progress is made, add a short reminder:

> **Slide reminder:** This is a visual-worthy milestone. Consider adding a figure to your supervisor slides now so you don't have to backfill later. The presentation lives in `Supervisor_Meeting_3_Bonolo_Masima.pptx` — what would make the best visual for this progress?

### Suggested visual types for this project

- Matplotlib bar/heatmap of success rates across the causal variable grid (σ_d, ρ, φ, θ)
- Open3D / matplotlib point cloud + grasp distribution figures (already in `visualize_cgn_grasps.py`)
- SCM diagram (once fitting is done)
- Side-by-side comparison: clean point cloud vs degraded point cloud at specific (φ, σ_d, ρ) settings
- Counterfactual "what-if" plot (after diagnosis is implemented)
- LLM vs SCM accuracy comparison table/chart (after baseline is run)

### Context

The supervisor meeting on 3 July 2026 (Meeting 3) was positively received because of strong visuals. The goal is to keep visuals current with every major step so no backfilling is needed before future meetings. Deadlines: experiment freeze 27 July, report 14 August, poster 19 August 2026.
