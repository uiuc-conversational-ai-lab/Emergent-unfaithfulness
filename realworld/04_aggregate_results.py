"""
04_aggregate_results.py
=======================
Aggregate B-label and C-label judgments into summary tables.

Since we have no confirming/opposing pairs from real-world data,
we cannot compute FaithGap. Instead we report:

  - Unfaithfulness Rate (UFR): fraction of responses labeled B3-B8
  - Faithfulness Rate (FR):    fraction labeled B1 or B2*
  - Full B-label distribution: % per label per model/condition
  - Full C-label distribution: % per label per model (CoT only)

Output files:
  results/real/full_b_results.csv       — all records with B labels
  results/real/full_c_results.csv       — all CoT records with C labels
  results/real/unfaithfulness_rates.csv — FR / UFR per model x condition
  results/real/summary_b_distribution.csv — full B% table
  results/real/summary_c_distribution.csv — full C% table (CoT only)

Run:
    python 04_aggregate_results.py
"""

import json
import os
import glob
import pandas as pd

# Label definitions
FAITHFUL_LABELS   = {"B1", "B2*"}
UNFAITHFUL_LABELS = {"B3", "B4", "B5", "B6", "B7", "B8"}
B_ORDER = ["B1", "B2*", "B3", "B4", "B5", "B6", "B7", "B8"]
C_ORDER = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]


def load_jsonl(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def load_all(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    all_records = []
    for path in files:
        all_records.extend(load_jsonl(path))
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def compute_b_distribution(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    """Compute B-label % distribution grouped by specified columns."""
    valid   = df[df["b_label"].isin(B_ORDER)].copy()
    grouped = valid.groupby(groupby + ["b_label"]).size().reset_index(name="count")
    totals  = valid.groupby(groupby).size().reset_index(name="total")
    merged  = grouped.merge(totals, on=groupby)
    merged["pct"] = (merged["count"] / merged["total"] * 100).round(1)

    pivot = merged.pivot_table(
        index=groupby, columns="b_label", values="pct", fill_value=0.0
    ).reset_index()

    # Ensure all B labels are present as columns
    for col in B_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[groupby + B_ORDER]

    # Add summary metrics
    faithful_cols   = [c for c in FAITHFUL_LABELS if c in pivot.columns]
    unfaithful_cols = [c for c in UNFAITHFUL_LABELS if c in pivot.columns]
    pivot["FaithRate"]   = pivot[faithful_cols].sum(axis=1).round(1)
    pivot["UnfaithRate"] = pivot[unfaithful_cols].sum(axis=1).round(1)

    # Add n
    totals_indexed = totals.set_index(groupby)["total"]
    pivot["n"] = pivot[groupby].apply(
        lambda row: totals_indexed.loc[tuple(row)] if len(groupby) > 1
                    else totals_indexed.loc[row.iloc[0]],
        axis=1
    ).values

    return pivot


def compute_c_distribution(df: pd.DataFrame, groupby: list) -> pd.DataFrame:
    """Compute C-label % distribution grouped by specified columns."""
    valid = df[df["cot_label"].isin(C_ORDER)].copy()
    if valid.empty:
        return pd.DataFrame()

    grouped = valid.groupby(groupby + ["cot_label"]).size().reset_index(name="count")
    totals  = valid.groupby(groupby).size().reset_index(name="total")
    merged  = grouped.merge(totals, on=groupby)
    merged["pct"] = (merged["count"] / merged["total"] * 100).round(1)

    pivot = merged.pivot_table(
        index=groupby, columns="cot_label", values="pct", fill_value=0.0
    ).reset_index()

    for col in C_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[groupby + C_ORDER]

    totals_indexed = totals.set_index(groupby)["total"]
    pivot["n"] = pivot[groupby].apply(
        lambda row: totals_indexed.loc[tuple(row)] if len(groupby) > 1
                    else totals_indexed.loc[row.iloc[0]],
        axis=1
    ).values

    return pivot


def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    os.makedirs("results/real", exist_ok=True)

    # ── Load all B-label judgments ─────────────────────────────────────────
    b_df = load_all("results/real/judgments/*__b_labels.jsonl")
    if b_df.empty:
        print("No B-label files found. Run 03a_judge_b_labels.py first.")
        return

    # Normalise use_system to bool
    b_df["use_system"] = b_df["use_system"].map(
        {True: True, False: False, "True": True, "False": False}
    )
    b_df["sys_prompt"] = b_df["use_system"].map(
        {True: "with_sys", False: "no_sys"}
    )

    print(f"Loaded {len(b_df)} B-labeled records "
          f"across {b_df['model'].nunique()} models")

    # ── 1. Unfaithfulness rate — model × prompt_cond × sys_prompt ─────────
    sep("Faithfulness / Unfaithfulness Rate by Model × Condition")
    ufr = compute_b_distribution(
        b_df, ["family", "model", "prompt_cond", "sys_prompt"]
    )
    cols = ["family", "model", "prompt_cond", "sys_prompt",
            "FaithRate", "UnfaithRate", "n"]
    print(ufr[cols].to_string(index=False))
    ufr.to_csv("results/real/unfaithfulness_rates.csv", index=False)

    # ── 2. Full B-distribution — model × prompt_cond (collapsed sys) ──────
    sep("Full B-Label Distribution by Model × Prompt Condition")
    b_dist = compute_b_distribution(b_df, ["family", "model", "prompt_cond"])
    print(b_dist.to_string(index=False))
    b_dist.to_csv("results/real/summary_b_distribution.csv", index=False)

    # ── 3. Direct prompt only — clean model comparison ────────────────────
    direct = b_df[(b_df["prompt_cond"] == "direct") & (b_df["use_system"] == True)]
    if not direct.empty:
        sep("B-Label Distribution — Direct Prompt, With System Prompt")
        b_direct = compute_b_distribution(direct, ["family", "model"])
        print(b_direct.to_string(index=False))

    # ── 4. Family-level summary ────────────────────────────────────────────
    direct_all = b_df[b_df["prompt_cond"] == "direct"]
    if not direct_all.empty:
        sep("Family-Level Summary — Direct Prompt")
        b_family = compute_b_distribution(direct_all, ["family"])
        display_cols = ["family", "B1", "B2*", "B3", "B4",
                        "B5", "B6", "B7", "B8", "FaithRate", "UnfaithRate", "n"]
        display_cols = [c for c in display_cols if c in b_family.columns]
        print(b_family[display_cols].to_string(index=False))

    # ── Save full B results ────────────────────────────────────────────────
    b_df.to_csv("results/real/full_b_results.csv", index=False)

    # ── Load and process C-label judgments ────────────────────────────────
    c_df = load_all("results/real/judgments/*__c_labels.jsonl")
    if not c_df.empty:
        c_df["use_system"] = c_df["use_system"].map(
            {True: True, False: False, "True": True, "False": False}
        )
        c_df["sys_prompt"] = c_df["use_system"].map(
            {True: "with_sys", False: "no_sys"}
        )
        print(f"\nLoaded {len(c_df)} C-labeled records")

        sep("C-Label Distribution by Model (CoT Condition)")
        c_dist = compute_c_distribution(c_df, ["family", "model"])
        if not c_dist.empty:
            print(c_dist.to_string(index=False))
            c_dist.to_csv("results/real/summary_c_distribution.csv", index=False)

        # With / without system prompt breakdown
        sep("C-Label Distribution by Model × System Prompt (CoT Condition)")
        c_dist_sys = compute_c_distribution(
            c_df, ["family", "model", "sys_prompt"])
        if not c_dist_sys.empty:
            print(c_dist_sys.to_string(index=False))

        c_df.to_csv("results/real/full_c_results.csv", index=False)
    else:
        print("\nNo C-label files found. Run 03b_judge_c_labels.py for CoT analysis.")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Output files saved to results/real/")
    print(f"{'='*70}")
    print("  full_b_results.csv          — all records with B labels")
    print("  full_c_results.csv          — all CoT records with C labels")
    print("  unfaithfulness_rates.csv    — FR/UFR per model × condition")
    print("  summary_b_distribution.csv  — full B% distribution table")
    if not c_df.empty:
        print("  summary_c_distribution.csv  — full C% distribution table (CoT)")


if __name__ == "__main__":
    main()
