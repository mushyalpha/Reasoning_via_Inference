"""
run_llm_baseline.py
====================
LLM Baseline Comparison for MSc Thesis:
  "Causal Inference for Robotic Grasp Failure Diagnosis under Perceptual Degradation"

Pre-registered protocol (v2 — frozen before first API call):
  - Model:        gemini-2.5-flash at temperature=0.0 (scored) and temperature=1.0 (consistency)
  - Tiers:        T1 (failure + variable names), T2 (+camera config), T3 (+perception metrics)
  - Trials:       All 292 failed trials from counterfactual_groundtruth.csv
  - Calls/trial:  4 per tier (1 scored at temp=0.0, 3 consistency at temp=1.0)
  - Total calls:  292 x 4 x 3 tiers = 3,504
  - Scoring:      Strict exact-match (none != joint; see pre-registration plan)
  - Primary eval: 95 single-variable trials (sigma_d=36, theta=35, phi=23, rho=1)
  - Secondary:    All 292 trials (majority-class baseline = 57.2%)

Usage:
    python run_llm_baseline.py [--dry-run] [--tier T1 T2 T3] [--n-sample N]
    python run_llm_baseline.py --tier T1 T2 T3          # full run
    python run_llm_baseline.py --dry-run --n-sample 5   # inspect prompts only
    python run_llm_baseline.py --summarise               # re-score from existing JSONL
"""

import argparse
import json
import os
import sys
import time
import re
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

from google import genai
from google.genai import types as genai_types

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).parent
RESULTS_DIR   = PROJECT_ROOT / "results"
GT_CSV        = RESULTS_DIR / "counterfactual_groundtruth.csv"
EXP_CSV       = RESULTS_DIR / "experiment_results.csv"
RAW_JSONL     = RESULTS_DIR / "llm_baseline_raw.jsonl"
SCORED_CSV    = RESULTS_DIR / "llm_baseline_results.csv"
SUMMARY_JSON  = RESULTS_DIR / "llm_baseline_summary.json"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_NAME       = "gemini-2.5-flash"
TEMP_SCORED      = 0.0
TEMP_CONSISTENCY = 1.0
N_CONSISTENCY    = 3   # stochastic calls for agreement-rate measurement

VALID_CODES = {"sigma_d", "rho", "phi", "theta", "joint", "none"}

# ---------------------------------------------------------------------------
# Pre-registered keyword fallback (applied only when JSON parsing fails)
# ---------------------------------------------------------------------------
KEYWORD_MAP = [
    ("sigma_d", ["sigma_d", "depth noise", "gaussian noise", "sensor noise", "depth error"]),
    ("rho",     ["rho", "sparsity", "sparse", "downsampl", "point cloud density", "missing points"]),
    ("phi",     ["phi", "elevation", "viewpoint height", "overhead", "camera angle"]),
    ("theta",   ["theta", "azimuth", "horizontal angle", "rotation", "lateral view"]),
    ("joint",   ["combination", "multiple", "joint", "both", "several"]),
    ("none",    ["cannot", "unable", "uncertain", "no single", "irreducible"]),
]

# ---------------------------------------------------------------------------
# Camera configuration labels
# ---------------------------------------------------------------------------

def get_phi_label(phi: float) -> str:
    if phi <= 35:
        return "low ~30deg"
    if phi <= 52:
        return "medium ~45deg"
    return "overhead ~60deg"


def get_theta_label(theta: float) -> str:
    if theta <= 22:
        return "frontal ~0deg"
    if theta <= 67:
        return "side ~45deg"
    return "rear ~90deg"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Header: context + variable definitions (shared across all tiers)
PREAMBLE_HEADER = (
    "You are a robotic systems analyst diagnosing a grasp failure.\n\n"
    "A robotic arm attempted to grasp an object using a 3D point cloud captured from a depth "
    "camera. The grasp failed.\n\n"
    "Four perceptual variables may have caused this failure. Exactly one of them (or none, or "
    "a combination) is the root cause:\n"
    "  (A) sigma_d -- Gaussian depth noise added to the depth sensor\n"
    "  (B) rho     -- Random downsampling of the point cloud (lower = sparser)\n"
    "  (C) phi     -- Camera elevation angle\n"
    "  (D) theta   -- Camera azimuth (horizontal rotation) angle"
)

# Footer: task instruction + required JSON format (always last)
PREAMBLE_FOOTER = (
    "\n\nTask: Identify the single most likely root cause, or respond \"joint\" if two or more "
    "variables share responsibility equally, or \"none\" if you cannot identify a cause.\n\n"
    "This is a CLOSED-SET task. You must choose from: sigma_d, rho, phi, theta, joint, none.\n\n"
    'Respond with ONLY a valid JSON object, no other text:\n'
    '{\n'
    '  "attribution": "<sigma_d | rho | phi | theta | joint | none>",\n'
    '  "confidence": "<high | medium | low>",\n'
    '  "reasoning": "<2-3 sentence causal explanation>"\n'
    '}'
)


def build_t1(row: dict) -> str:
    """T1: Failure fact + candidate variable names only. No camera config."""
    return PREAMBLE_HEADER + PREAMBLE_FOOTER


def build_t2(row: dict) -> str:
    """T2: T1 + qualitative camera configuration (injected before JSON instruction)."""
    camera_section = (
        "\n\nObservable camera configuration:\n"
        f"  - Camera elevation: {get_phi_label(float(row['phi']))}\n"
        f"  - Camera azimuth:   {get_theta_label(float(row['theta']))}"
    )
    return PREAMBLE_HEADER + camera_section + PREAMBLE_FOOTER


def build_t3(row: dict) -> str:
    """T3: T2 + full perception pipeline metrics (injected before JSON instruction)."""
    n_grasps = int(float(row.get("n_grasps", 0)))
    camera_section = (
        "\n\nObservable camera configuration:\n"
        f"  - Camera elevation: {get_phi_label(float(row['phi']))}\n"
        f"  - Camera azimuth:   {get_theta_label(float(row['theta']))}"
    )
    metrics_section = (
        "\n\nObservable perception pipeline data:\n"
        f"  - Point cloud completeness (C_pc): {float(row['C_pc']):.4f}\n"
        "    (0-1; higher = more complete scene coverage)\n"
        f"  - Grasp candidates proposed (n_grasps): {n_grasps}\n"
        "    (0 = pipeline found no valid grasps at all)\n"
        f"  - Best grasp confidence (q_grasp): {float(row['q_grasp']):.4f}\n"
        "    (0-1; higher = algorithm is more confident)\n"
        f"  - Pose error at execution (e_pose): {float(row['e_pose']):.4f} m\n"
        "    (distance between proposed grasp centre and true object centroid)"
    )
    return PREAMBLE_HEADER + camera_section + metrics_section + PREAMBLE_FOOTER


PROMPT_BUILDERS = {"T1": build_t1, "T2": build_t2, "T3": build_t3}


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

def keyword_fallback(text: str) -> str:
    """
    Apply pre-registered keyword matching.
    Returns: attribution code, 'parse_ambiguous', or 'parse_failure'.
    """
    text_lower = text.lower()
    matched = set()
    for code, keywords in KEYWORD_MAP:
        if any(kw in text_lower for kw in keywords):
            matched.add(code)
    if len(matched) == 1:
        return matched.pop()
    if len(matched) > 1:
        return "parse_ambiguous"
    return "parse_failure"


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(raw_text: str) -> dict:
    """
    Extract attribution from model response.
    Returns dict with: attribution, confidence, reasoning, parse_status.
    parse_status: 'json_ok' | 'keyword_fallback' | 'parse_ambiguous' | 'parse_failure'
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    # Attempt JSON parse
    try:
        obj = json.loads(cleaned)
        code = str(obj.get("attribution", "")).strip().lower()
        if code in VALID_CODES:
            return {
                "attribution":  code,
                "confidence":   str(obj.get("confidence", "")).strip().lower(),
                "reasoning":    str(obj.get("reasoning", "")).strip(),
                "parse_status": "json_ok",
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Keyword fallback
    code = keyword_fallback(raw_text)
    final_code = code if code in VALID_CODES else "parse_failure"
    return {
        "attribution":  final_code,
        "confidence":   "unknown",
        "reasoning":    "",
        "parse_status": code if code in ("parse_failure", "parse_ambiguous") else "keyword_fallback",
    }


# ---------------------------------------------------------------------------
# Strict pre-registered scoring rule
# ---------------------------------------------------------------------------

def score_attribution(attribution: str, primary_cause: str) -> int:
    """
    Returns 1 (correct) or 0 (incorrect).

    Strict exact-match rules:
      - Multi-variable ground truth (contains '+') -> LLM must output 'joint'
      - 'none' ground truth                        -> LLM must output 'none' (NOT joint)
      - Single-variable ground truth               -> exact match only
      - Parse failures always score 0
    """
    if attribution in ("parse_failure", "parse_ambiguous"):
        return 0
    if "+" in primary_cause:
        return int(attribution == "joint")
    return int(attribution == primary_cause)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> list:
    """
    Join counterfactual_groundtruth.csv with experiment_results.csv on trial_id.
    Returns list of merged row dicts (all rows in GT are failures).
    """
    exp_by_id = {}
    with open(EXP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            exp_by_id[str(row["trial_id"])] = row

    merged = []
    with open(GT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = str(row["trial_id"])
            exp_row = exp_by_id.get(tid, {})
            merged.append({**exp_row, **row})  # GT columns win on collision
    return merged


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_done_keys() -> set:
    """Return set of (trial_id, tier, call_type) already written to RAW_JSONL."""
    done = set()
    if RAW_JSONL.exists():
        with open(RAW_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((str(rec["trial_id"]), rec["tier"], rec["call_type"]))
                except Exception:
                    pass
    return done


def append_raw(record: dict) -> None:
    with open(RAW_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_llm(client, prompt: str, temperature: float, max_retries: int = 5) -> str:
    """Call Gemini model and return text response. Retries on transient errors."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=temperature),
            )
            return response.text.strip()
        except Exception as exc:
            wait = 2 ** attempt
            print(f"\n  [API error attempt {attempt+1}/{max_retries}]: {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    return "API_ERROR"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(tiers: list, dry_run: bool, n_sample) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not dry_run:
        print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = None
    if not dry_run:
        client = genai.Client(api_key=api_key)
        print(f"Model: {MODEL_NAME}")

    rows = load_data()
    print(f"Loaded {len(rows)} failed trials from ground-truth CSV.")

    if n_sample:
        rows = rows[:n_sample]
        print(f"Limited to {n_sample} trials (--n-sample).")

    done_keys = load_done_keys()
    print(f"Checkpoint: {len(done_keys)} calls already recorded in {RAW_JSONL.name}")

    calls_per_trial = 1 + N_CONSISTENCY
    total_calls = len(rows) * len(tiers) * calls_per_trial
    call_count = 0
    t_start = time.time()

    for tier in tiers:
        builder = PROMPT_BUILDERS[tier]
        print("\n" + "=" * 60)
        print(f"  Tier: {tier}  |  Trials: {len(rows)}  |  Calls: {len(rows)*calls_per_trial}")
        print("=" * 60)

        for i, row in enumerate(rows):
            trial_id = str(row["trial_id"])
            primary_cause = row["primary_cause"]
            prompt = builder(row)

            if dry_run:
                print("\n" + "-" * 60)
                print(f"  DRY RUN | trial_id={trial_id} | tier={tier} | ground_truth={primary_cause}")
                print("-" * 60)
                print(prompt)
                call_count += calls_per_trial
                continue

            # Scored call (temperature=0.0)
            key_scored = (trial_id, tier, "scored")
            if key_scored not in done_keys:
                raw = call_llm(client, prompt, TEMP_SCORED)
                parsed = parse_response(raw)
                correct = score_attribution(parsed["attribution"], primary_cause)
                record = {
                    "trial_id":      trial_id,
                    "tier":          tier,
                    "call_type":     "scored",
                    "temperature":   TEMP_SCORED,
                    "primary_cause": primary_cause,
                    "raw_response":  raw,
                    "model":         MODEL_NAME,
                    "timestamp":     datetime.utcnow().isoformat() + "Z",
                    **parsed,
                    "correct":       correct,
                }
                append_raw(record)
                done_keys.add(key_scored)
            call_count += 1

            # Consistency calls (temperature=1.0)
            for c in range(N_CONSISTENCY):
                key_cons = (trial_id, tier, f"consistency_{c}")
                if key_cons not in done_keys:
                    raw = call_llm(client, prompt, TEMP_CONSISTENCY)
                    parsed = parse_response(raw)
                    record = {
                        "trial_id":      trial_id,
                        "tier":          tier,
                        "call_type":     f"consistency_{c}",
                        "temperature":   TEMP_CONSISTENCY,
                        "primary_cause": primary_cause,
                        "raw_response":  raw,
                        "model":         MODEL_NAME,
                        "timestamp":     datetime.utcnow().isoformat() + "Z",
                        **parsed,
                        "correct":       None,  # consistency calls are not scored
                    }
                    append_raw(record)
                    done_keys.add(key_cons)
                    time.sleep(0.4)  # courtesy pause
                call_count += 1

            # Progress line
            elapsed = time.time() - t_start
            pct = call_count / total_calls * 100
            rate = call_count / elapsed if elapsed > 0 else 1e-9
            eta_min = (total_calls - call_count) / rate / 60
            print(
                f"  [{tier}] {i+1}/{len(rows)} trials | "
                f"{pct:.1f}% done | ETA {eta_min:.1f} min",
                end="\r", flush=True,
            )

    if not dry_run:
        print(f"\n\nAll calls complete. Raw log: {RAW_JSONL}")
        summarise()


# ---------------------------------------------------------------------------
# Offline scorer / summariser
# ---------------------------------------------------------------------------

def summarise() -> None:
    """
    Read llm_baseline_raw.jsonl, apply the pre-registered scoring rule, and write:
      - results/llm_baseline_results.csv
      - results/llm_baseline_summary.json
    """
    if not RAW_JSONL.exists():
        print(f"ERROR: {RAW_JSONL} not found. Run the API calls first.", file=sys.stderr)
        return

    # Group records by (trial_id, tier)
    groups = defaultdict(lambda: {"scored": None, "consistency": []})
    with open(RAW_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = (str(rec["trial_id"]), rec["tier"])
                if rec["call_type"] == "scored":
                    groups[key]["scored"] = rec
                elif rec["call_type"].startswith("consistency"):
                    groups[key]["consistency"].append(rec)
            except Exception:
                pass

    SINGLE_VAR_CAUSES = {"sigma_d", "rho", "phi", "theta"}

    scored_rows = []
    for (trial_id, tier), g in sorted(groups.items()):
        sc = g["scored"]
        if sc is None:
            continue
        primary_cause = sc["primary_cause"]
        cons = g["consistency"]

        # Consistency: modal attribution rate across N_CONSISTENCY stochastic calls
        cons_attributions = [c["attribution"] for c in cons if c["attribution"] in VALID_CODES]
        if cons_attributions:
            cnt = Counter(cons_attributions)
            modal_code, modal_count = cnt.most_common(1)[0]
            agreement_rate = round(modal_count / N_CONSISTENCY, 4)
        else:
            modal_code, agreement_rate = "no_data", None

        scored_rows.append({
            "trial_id":          trial_id,
            "tier":              tier,
            "primary_cause":     primary_cause,
            "is_single_var":     primary_cause in SINGLE_VAR_CAUSES,
            "attribution":       sc["attribution"],
            "confidence":        sc["confidence"],
            "reasoning":         sc.get("reasoning", ""),
            "parse_status":      sc["parse_status"],
            "correct":           sc["correct"],
            "consistency_modal": modal_code,
            "agreement_rate":    agreement_rate,
            "model":             sc.get("model", MODEL_NAME),
            "timestamp":         sc.get("timestamp", ""),
        })

    # Write CSV
    if scored_rows:
        with open(SCORED_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scored_rows[0].keys()))
            writer.writeheader()
            writer.writerows(scored_rows)
        print(f"Scored CSV: {SCORED_CSV}  ({len(scored_rows)} rows)")

    # Build per-tier summary
    TIERS = sorted({r["tier"] for r in scored_rows})
    ALL_CAUSES = ["sigma_d", "theta", "phi", "rho", "joint", "none"]
    SINGLE_CAUSES = ["sigma_d", "theta", "phi", "rho"]

    def accuracy(rows):
        if not rows:
            return None
        return round(sum(int(r["correct"]) for r in rows) / len(rows), 4)

    summary = {}
    for tier in TIERS:
        tier_rows  = [r for r in scored_rows if r["tier"] == tier]
        single_rows = [r for r in tier_rows if r["is_single_var"]]

        per_cause_primary = {}
        for cause in SINGLE_CAUSES:
            sub = [r for r in tier_rows if r["primary_cause"] == cause]
            per_cause_primary[cause] = {"n": len(sub), "accuracy": accuracy(sub)}

        per_cause_full = {}
        for cause in ALL_CAUSES:
            if cause == "joint":
                sub = [r for r in tier_rows if "+" in r["primary_cause"]]
            else:
                sub = [r for r in tier_rows if r["primary_cause"] == cause]
            per_cause_full[cause] = {"n": len(sub), "accuracy": accuracy(sub)}

        cons_rates = [r["agreement_rate"] for r in tier_rows if r["agreement_rate"] is not None]
        mean_agreement = round(sum(cons_rates) / len(cons_rates), 4) if cons_rates else None

        parse_fails = sum(
            1 for r in tier_rows
            if "failure" in str(r["parse_status"]) or "ambiguous" in str(r["parse_status"])
        )

        summary[tier] = {
            # Primary comparison (95 single-variable trials — both SCM and LLM are evaluated)
            "primary_n":            len(single_rows),
            "primary_accuracy":     accuracy(single_rows),
            "primary_per_cause":    per_cause_primary,
            # Secondary metric (full 292 trials — LLM only)
            "full_n":               len(tier_rows),
            "full_accuracy":        accuracy(tier_rows),
            "majority_class_baseline": round(167 / 292, 4),   # always-none = 57.2%
            "full_per_cause":       per_cause_full,
            # Consistency
            "mean_agreement_rate":  mean_agreement,
            # Parse quality
            "parse_failures":       parse_fails,
            "parse_failure_rate":   round(parse_fails / len(tier_rows), 4) if tier_rows else None,
        }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {SUMMARY_JSON}")

    # Console summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY  (pre-registered evaluation, v2)")
    print("=" * 70)
    for tier, s in summary.items():
        print(f"\n  Tier {tier}:")
        pa = s["primary_accuracy"]
        print(f"    PRIMARY ({s['primary_n']} single-variable trials):")
        print(f"      Mean accuracy : {pa:.1%}" if pa is not None else "      Mean accuracy : N/A")
        for cause in SINGLE_CAUSES:
            pc = s["primary_per_cause"][cause]
            note = "  *** (n=1, single trial)" if pc["n"] == 1 else ""
            acc_str = f"{pc['accuracy']:.1%}" if pc["accuracy"] is not None else "N/A"
            print(f"        {cause:8s}: {acc_str}  (n={pc['n']}){note}")
        fa = s["full_accuracy"]
        print(f"    SECONDARY (all {s['full_n']} trials):")
        print(f"      LLM accuracy       : {fa:.1%}" if fa is not None else "      LLM accuracy : N/A")
        print(f"      Majority-class base: {s['majority_class_baseline']:.1%}")
        ma = s["mean_agreement_rate"]
        print(f"    Consistency (mean agreement rate): {ma:.3f}" if ma is not None else "    Consistency: N/A")
        print(f"    Parse failures: {s['parse_failures']}/{s['full_n']} ({s['parse_failure_rate']:.1%})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM Baseline Comparison — Gemini 2.5 Flash runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier", nargs="+", default=["T1", "T2", "T3"],
        choices=["T1", "T2", "T3"],
        help="Tier(s) to run (default: all three)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print formatted prompts without making any API calls",
    )
    parser.add_argument(
        "--n-sample", type=int, default=None, metavar="N",
        help="Limit to first N trials (for testing)",
    )
    parser.add_argument(
        "--summarise", action="store_true",
        help="Re-score from existing llm_baseline_raw.jsonl without new API calls",
    )
    args = parser.parse_args()

    if args.summarise:
        summarise()
    else:
        run(tiers=args.tier, dry_run=args.dry_run, n_sample=args.n_sample)


if __name__ == "__main__":
    main()
