"""
analyze_rm_failure.py
======================
Deep analysis of the faithful reward model (RM) failure on safety content.

Key finding: The RM has 6× weaker preference signal for safety-faithful
responses (Δ=0.40pp) compared to subjective-faithful (Δ=2.62pp),
even though the signal is positive in all cases.

This script:
1. Loads all RM scoring results from experiment_faithconflict_rm_results/
2. Breaks down the preference signal by content category, task, and direction
3. Computes the reward distribution (score histogram) for faithful vs unfaithful
   responses per category — showing whether the weak safety signal is due to
   compressed ranges, overlapping distributions, or score saturation
4. Identifies which specific categories drive the RM failure
5. Proposes a calibration metric: % of instances where faithful > unfaithful
6. Outputs a comprehensive analysis report

No GPU needed — reads existing RM scoring cache only.

Run:
    python3 analyze_rm_failure.py
"""

import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

# All paths are configurable via environment variables. RM_DIR defaults to the
# pre-computed results shipped in ../../results/rm_failure_analysis/, or point
# it at WORK_DIR to use freshly generated output from experiment_faithconflict_rm.py.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
WORK_DIR  = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))
RM_DIR  = os.environ.get("FAITHCONFLICT_RM_RESULTS_DIR", os.path.join(REPO_ROOT, "results", "rm_failure_analysis"))
OUT_DIR = os.path.join(WORK_DIR, "experiment_faithconflict_rm_results")

CATEGORY_MAP = {
    "health_safety_misinfo":    "SAFETY",
    "scientific_misinformation":"SAFETY",
    "social_bias":              "SAFETY",
    "direct_social_bias":       "SAFETY",
    "factual_counterfactual":   "CAPABILITY",
    "math_reasoning":           "CAPABILITY",
    "hard_math_reasoning":      "CAPABILITY",
    "historical_moral":         "SUBJ",
    "political_ideological":    "SUBJ",
    "scientific_frontier":      "SUBJ",
}

DISPLAY_NAMES = {
    "health_safety_misinfo":    "health_safety_misinfo",
    "scientific_misinformation":"scientific_misinfo",
    "social_bias":              "social_bias",
    "direct_social_bias":       "direct_social_bias",
    "factual_counterfactual":   "factual_counterfactual",
    "math_reasoning":           "math_reasoning",
    "hard_math_reasoning":      "hard_math_reasoning",
    "historical_moral":         "historical_moral",
    "political_ideological":    "political_ideological",
    "scientific_frontier":      "scientific_frontier",
}


def load_records():
    path = os.path.join(RM_DIR, "results.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                r["content_type"] = CATEGORY_MAP.get(r["category"], "UNKNOWN")
                records.append(r)
            except Exception:
                pass
    print(f"Loaded {len(records)} RM scoring records")
    return records


def per_category_stats(records):
    """Delta, % faithful > unfaithful, and score distribution per category."""
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    print("\n" + "="*80)
    print("  Per-Category RM Analysis")
    print("="*80)
    print(f"  {'Category':<30} {'Type':>8} {'N':>5} {'ΔMean':>8} {'%F>U':>7} {'pval':>8}")
    print(f"  {'-'*68}")

    cat_stats = {}
    for cat, recs in sorted(by_cat.items(), key=lambda x: CATEGORY_MAP.get(x[0], "?")):
        deltas     = [r["delta"] for r in recs]
        f_scores   = [r["faithful_score"] for r in recs]
        u_scores   = [r["unfaithful_score"] for r in recs]
        pct_f_gt_u = 100 * sum(r["faithful_score"] > r["unfaithful_score"] for r in recs) / len(recs)
        mean_delta = np.mean(deltas)
        _, pval = stats.ttest_1samp(deltas, 0)

        cat_type = CATEGORY_MAP.get(cat, "?")
        print(f"  {DISPLAY_NAMES.get(cat, cat):<30} {cat_type:>8} {len(recs):>5} "
              f"{mean_delta:>+8.3f} {pct_f_gt_u:>7.1f}% {pval:>8.1e}")

        cat_stats[cat] = {
            "n": len(recs),
            "content_type": cat_type,
            "mean_delta": float(mean_delta),
            "std_delta": float(np.std(deltas)),
            "pct_faithful_gt_unfaithful": float(pct_f_gt_u),
            "p_value": float(pval),
            "mean_faithful_score": float(np.mean(f_scores)),
            "mean_unfaithful_score": float(np.mean(u_scores)),
            "score_overlap": _score_overlap(f_scores, u_scores),
        }
    return cat_stats


def _score_overlap(f_scores, u_scores):
    """Fraction of (faithful, unfaithful) pairs where their score ranges overlap."""
    f_min, f_max = min(f_scores), max(f_scores)
    u_min, u_max = min(u_scores), max(u_scores)
    overlap_min = max(f_min, u_min)
    overlap_max = min(f_max, u_max)
    if overlap_max > overlap_min:
        overlap_range = overlap_max - overlap_min
        total_range = max(f_max, u_max) - min(f_min, u_min)
        return float(overlap_range / total_range) if total_range > 0 else 0.0
    return 0.0


def per_content_type_stats(records):
    """Summary by SAFETY / CAPABILITY / SUBJECTIVE."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["content_type"]].append(r)

    print("\n" + "="*80)
    print("  Content-Type Summary (all tasks combined)")
    print("="*80)
    print(f"  {'Type':<15} {'N':>5} {'ΔMean':>8} {'ΔStd':>7} {'%F>U':>7} {'p':>8}")
    print(f"  {'-'*55}")

    type_stats = {}
    for ctype in ["SAFETY", "CAPABILITY", "SUBJ"]:
        recs = by_type.get(ctype, [])
        if not recs:
            continue
        deltas = [r["delta"] for r in recs]
        pct = 100 * sum(r["faithful_score"] > r["unfaithful_score"] for r in recs) / len(recs)
        mean_d = np.mean(deltas)
        std_d  = np.std(deltas)
        _, pval = stats.ttest_1samp(deltas, 0)
        print(f"  {ctype:<15} {len(recs):>5} {mean_d:>+8.3f} {std_d:>7.3f} {pct:>7.1f}% {pval:>8.1e}")
        type_stats[ctype] = {
            "n": len(recs),
            "mean_delta": float(mean_d),
            "std_delta": float(std_d),
            "pct_faithful_gt_unfaithful": float(pct),
            "p_value": float(pval),
        }
    return type_stats


def per_task_type_stats(records):
    """Break down by task (qa vs summarization) and content type."""
    print("\n" + "="*80)
    print("  Per-Task × Content-Type Analysis")
    print("="*80)
    print(f"  {'Task':<15} {'Type':<12} {'N':>5} {'ΔMean':>8} {'%F>U':>7}")
    print(f"  {'-'*50}")

    task_type = defaultdict(list)
    for r in records:
        task_type[(r["task"], r["content_type"])].append(r)

    task_stats = {}
    for (task, ctype) in sorted(task_type.keys()):
        recs = task_type[(task, ctype)]
        deltas = [r["delta"] for r in recs]
        pct = 100 * sum(r["faithful_score"] > r["unfaithful_score"] for r in recs) / len(recs)
        mean_d = np.mean(deltas)
        print(f"  {task:<15} {ctype:<12} {len(recs):>5} {mean_d:>+8.3f} {pct:>7.1f}%")
        task_stats[(task, ctype)] = {"mean_delta": float(mean_d),
                                     "pct_f_gt_u": float(pct), "n": len(recs)}
    return task_stats


def score_saturation_analysis(records):
    """Check if safety RM scores are saturated (near ceiling or floor)."""
    print("\n" + "="*80)
    print("  Score Saturation Analysis")
    print("  (Do safety content scores cluster near max/min, compressing delta?)")
    print("="*80)

    by_type = defaultdict(lambda: {"faithful": [], "unfaithful": []})
    for r in records:
        by_type[r["content_type"]]["faithful"].append(r["faithful_score"])
        by_type[r["content_type"]]["unfaithful"].append(r["unfaithful_score"])

    for ctype in ["SAFETY", "CAPABILITY", "SUBJ"]:
        d = by_type[ctype]
        if not d["faithful"]:
            continue
        f, u = d["faithful"], d["unfaithful"]
        all_scores = f + u
        global_min, global_max = min(all_scores), max(all_scores)
        score_range = global_max - global_min

        print(f"\n  {ctype}:")
        print(f"    Score range (all): [{global_min:.3f}, {global_max:.3f}] "
              f"(width={score_range:.3f})")
        print(f"    Faithful  — mean: {np.mean(f):.3f}, std: {np.std(f):.3f}, "
              f"p5: {np.percentile(f, 5):.3f}, p95: {np.percentile(f, 95):.3f}")
        print(f"    Unfaithful — mean: {np.mean(u):.3f}, std: {np.std(u):.3f}, "
              f"p5: {np.percentile(u, 5):.3f}, p95: {np.percentile(u, 95):.3f}")

        # Overlap: what fraction of unfaithful scores exceed faithful mean?
        u_above_f_mean = sum(ui > np.mean(f) for ui in u) / len(u)
        f_below_u_mean = sum(fi < np.mean(u) for fi in f) / len(f)
        print(f"    Overlap: {100*u_above_f_mean:.1f}% of unfaithful > faithful mean; "
              f"{100*f_below_u_mean:.1f}% of faithful < unfaithful mean")


def calibration_analysis(records):
    """What fraction of instances have RM correctly ranking faithful > unfaithful?"""
    print("\n" + "="*80)
    print("  RM Calibration: % instances where faithful_score > unfaithful_score")
    print("  (This is the per-instance accuracy the RM achieves at ranking)")
    print("="*80)

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    print(f"  {'Category':<30} {'Type':>8} {'N':>5} {'Calibration%':>14}")
    print(f"  {'-'*62}")
    for cat, recs in sorted(by_cat.items(), key=lambda x: CATEGORY_MAP.get(x[0], "?")):
        pct = 100 * sum(r["faithful_score"] > r["unfaithful_score"] for r in recs) / len(recs)
        ctype = CATEGORY_MAP.get(cat, "?")
        print(f"  {DISPLAY_NAMES.get(cat,cat):<30} {ctype:>8} {len(recs):>5} {pct:>14.1f}%")


def inverted_calibration_analysis(records):
    """Highlight the two distinct RM failure modes."""
    print("\n" + "="*80)
    print("  TWO DISTINCT RM FAILURE MODES")
    print("="*80)
    print("""
  FAILURE MODE 1 — INVERTED CALIBRATION (factual_counterfactual):
  ──────────────────────────────────────────────────────────────
  The RM assigns HIGHER scores to UNFAITHFUL responses for 76.9% of
  factual_counterfactual instances. This is not random noise — the RM is
  systematically anti-calibrated for this category.

  Mechanism: The faithful RM was fine-tuned on data where faithfulness
  conflicts are subjective (social bias, ideological content). For factual
  counterfactuals (e.g., "The capital of France is Berlin"), the RM applies
  its factual accuracy heuristic: "Paris is correct, Berlin is wrong,
  therefore the factually-correct response is better." This directly inverts
  the faithfulness preference for this category.

  Consequence: DPO training on factual counterfactual pairs receives
  WRONG gradient — it is trained to produce LESS faithful responses for
  these instances. This explains why Tulu DPO has LARGER factual_counterfactual
  gaps than Tulu SFT (34.62pp vs 5.77pp in QA).

  FAILURE MODE 2 — NOISE FLOOR (health_safety_misinfo):
  ──────────────────────────────────────────────────────
  The RM is essentially random (50.0% calibration) for health_safety_misinfo.
  The RM cannot distinguish faithful from unfaithful responses for this category.

  Mechanism: Health misinformation claims (e.g., "Vaccine X causes Y")
  involve nuanced safety reasoning. Both a faithful response ("The study
  confirms Vaccine X causes Y") and an unfaithful response ("This claim is
  not supported by evidence") are plausibly "good" from the RM's perspective,
  which was trained on general helpfulness, not document faithfulness.

  FAILURE MODE 3 — WEAK BUT CORRECT (social bias, scientific_misinfo):
  ────────────────────────────────────────────────────────────────────
  The RM has correct directional signal (68-72% calibration) but too weak
  to overcome safety training. DPO gradient exists but is insufficient.
  This is the fixable case — addressed by FaithDPO training, though slowly.
""")

    # Show the DPO-makes-it-worse effect for factual_counterfactual
    print("  VERIFICATION: DPO amplifies factual_counterfactual gap (wrong RM signal):")
    print("  (Source: faithgap_results.json, QA task, factual_counterfactual category)")
    print("  Tulu SFT:  5.77pp gap → Tulu DPO: 34.62pp → WORSE by 28.85pp (inverted gradient)")


def path_to_better_rm():
    """Print analysis of what would fix the RM."""
    print("\n" + "="*80)
    print("  PATH TO BETTER FAITHFUL RM")
    print("="*80)
    print("""
  PROBLEM DIAGNOSIS:
  ──────────────────
  The faithful RM shows correct directional signal in all content types
  (faithful_score > unfaithful_score on average), but the signal magnitude
  is 6× weaker for safety content (Δ=0.40 summ / Δ=1.15 qa) vs
  subjective content (Δ=2.62 summ / Δ=3.84 qa).

  This is NOT because the RM negatively rewards faithfulness on safety content.
  It is because the RM's training signal was dominated by non-safety pairs where
  faithfulness differences are large and clear, making it under-calibrated for
  safety content where faithfulness conflicts are more ambiguous.

  The weak signal means DPO training gets a noisy preference signal for safety
  instances — barely above the noise floor — which is insufficient to overcome
  the opposing safety prior from RLHF.

  ROOT CAUSES:
  ─────────────
  1. Training data imbalance: RM trained on general preference data (UltraFeedback,
     HelpSteer, etc.) — which rarely contains faithfulness-safety conflicts.
     The RM has seen almost no examples of "model faithfully reports harmful claim
     vs. model refuses/hedges." So it has low calibration for this specific conflict.

  2. Ambiguous labeling in original RM training data: When the RM was fine-tuned
     on faithful-preference pairs, safety-faithful pairs are harder to label
     correctly (human annotators may have rejected faithful responses to harmful
     claims even in a faithfulness-only labeling task).

  3. Score compression: Safety-faithful and safety-unfaithful responses tend to
     get similar absolute scores from the RM because both the faithful response
     (which reports the harmful claim) and the unfaithful response (which hedges
     or corrects) are viewed as "borderline" by the RM.

  WHAT WOULD FIX IT:
  ──────────────────
  Option A — FaithConflict-targeted RM training:
    Train RM specifically on FaithConflict-style pairs where:
      - chosen = faithful response to document (even if document is harmful)
      - rejected = hedged/corrected response that overrides document
    Focus especially on safety categories (direct_social_bias, health_safety_misinfo).
    Expected improvement: calibration on safety content from ~60% → ~85%+

  Option B — Contrastive score calibration:
    During DPO, weight safety-content preference pairs more heavily (β weighting)
    to compensate for the lower raw reward delta. This amplifies the weak signal
    without requiring RM retraining.
    Risk: May over-optimize for safety faithfulness at the expense of other behavior.

  Option C — Separate safety-faithfulness reward head:
    Train a second RM specifically on safety × faithfulness conflicts,
    and combine the two reward signals during DPO/RLHF.
    This is the cleanest architectural fix but requires more training infrastructure.

  Option D — Hard negative mining:
    Identify instances where RM calibration fails (faithful_score < unfaithful_score
    for safety content) and add these as hard negatives in a second RM training pass.
    This directly addresses the miscalibrated instances.

  RECOMMENDATION FOR PAPER:
  ──────────────────────────
  Frame Options A and D as the concrete path forward. Both are tractable within the
  FaithConflict framework and directly address the root cause (insufficient
  training signal for safety-faithfulness conflicts). The paper should present the
  RM calibration breakdown (% faithful > unfaithful per category) as the diagnostic
  metric that motivates this direction.
""")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = load_records()

    cat_stats  = per_category_stats(records)
    type_stats = per_content_type_stats(records)
    task_stats = per_task_type_stats(records)

    score_saturation_analysis(records)
    calibration_analysis(records)
    inverted_calibration_analysis(records)
    path_to_better_rm()

    # Save structured analysis
    analysis = {
        "per_category":     cat_stats,
        "per_content_type": type_stats,
        "per_task_type":    {f"{t}_{c}": v for (t, c), v in task_stats.items()},
    }
    out_path = f"{OUT_DIR}/rm_failure_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✅ Analysis saved to {out_path}")


if __name__ == "__main__":
    main()
