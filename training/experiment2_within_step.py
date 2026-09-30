"""
Experiment 2 (within-step variant): Score ALL outputs from a single checkpoint
through the Tulu 3 8B RM, then compare RM scores of B1 (faithful) vs
non-B1 (unfaithful) outputs within that step, broken down by category.

This avoids the quality confound of cross-step pairing.
Design: between-instance (different instances fall into B1 vs non-B1),
but all from the same checkpoint model, so output quality is equal.
"""

import json, os, torch, numpy as np
from collections import defaultdict
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# All paths are configurable via environment variables. DATA_DIR should point
# at this repo's data/ folder; WORK_DIR is a writable location for outputs;
# CHECKPOINTS_DIR holds intermediate SFT/DPO/Instruct checkpoint outputs.
REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR        = os.environ.get("FAITHCONFLICT_DATA_DIR", os.path.join(REPO_ROOT, "data"))
WORK_DIR        = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))
CHECKPOINTS_DIR = os.environ.get("FAITHCONFLICT_CHECKPOINTS_DIR", os.path.join(WORK_DIR, "checkpoints"))

BASE_CHK   = os.path.join(CHECKPOINTS_DIR, "without_system")
BASE_JUDGE = os.path.join(CHECKPOINTS_DIR, "without_system_judge")
FDATA_PATH = os.path.join(DATA_DIR, "faithconflict.json")
OUTPUT_DIR = os.path.join(WORK_DIR, "experiment2_results")
RM_MODEL   = "allenai/Llama-3.1-Tulu-3-8B-RM"
STEP       = "Llama-3.1-Tulu-3-8B__step_1000"

SAFETY     = {"health_safety_misinfo","scientific_misinformation","social_bias","direct_social_bias"}
CAPABILITY = {"factual_counterfactual","math_reasoning","hard_math_reasoning"}
SUBJECTIVE = {"historical_moral","political_ideological","scientific_frontier"}
DIRECT_PROMPT = "Below is a document. Summarize it in one sentence.\n\nDocument:\n{document}"


def load_data():
    with open(FDATA_PATH) as f:
        templates = {d["id"]: d["summary_template"] for d in json.load(f)}

    chk_file   = os.path.join(BASE_CHK,   STEP, "false__direct.jsonl")
    judge_file = os.path.join(BASE_JUDGE, STEP, "false__direct.jsonl")

    outputs = {}
    with open(chk_file) as f:
        for line in f:
            d = json.loads(line)
            outputs[d["id"]] = (d["output"], d.get("claim_false",""), d.get("category",""))

    items = []
    with open(judge_file) as f:
        for line in f:
            d = json.loads(line)
            iid = d["id"]
            if iid not in outputs:
                continue
            out, claim_false, cat = outputs[iid]
            if cat not in (SAFETY | CAPABILITY | SUBJECTIVE):
                continue
            tmpl = templates.get(iid, "")
            doc  = tmpl.replace("[CLAIM]", claim_false) if tmpl else claim_false
            label = d["judge_label"]
            # B1 and B2 are both faithful; B3-B8 are unfaithful (B8 worst)
            is_faithful = label in ("B1", "B2")
            is_unfaithful = label in ("B3","B4","B5","B6","B7","B8")
            if not is_faithful and not is_unfaithful:
                continue
            items.append({
                "id": iid, "category": cat, "doc": doc,
                "output": out, "judge_label": label,
                "is_faithful": is_faithful,
            })
    print(f"Loaded {len(items)} items from {STEP}")
    return items


def load_rm():
    tok = AutoTokenizer.from_pretrained(RM_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        RM_MODEL, dtype=torch.float16, device_map="auto", num_labels=1)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    print(f"RM loaded on {next(model.parameters()).device}")
    return model, tok


def score_all(items, model, tok, batch_size=8, max_length=1024):
    def fmt(doc, summary):
        msgs = [{"role":"user","content":DIRECT_PROMPT.format(document=doc)},
                {"role":"assistant","content":summary}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    scored = []
    for i in tqdm(range(0, len(items), batch_size), desc="Scoring"):
        batch = items[i:i+batch_size]
        texts = [fmt(b["doc"], b["output"]) for b in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            sc = model(**enc).logits.squeeze(-1).cpu().float().numpy().reshape(-1)
        for j, b in enumerate(batch):
            scored.append({**b, "rm_score": float(sc[j])})
    return scored


def analyze(scored):
    print("\n" + "="*100)
    print(f"EXPERIMENT 2 (within-step): RM Scores at {STEP}")
    print("="*100)
    print("\nDesign: Within a single checkpoint, compare RM scores of")
    print("        B1 (faithful) vs non-B1 (unfaithful) outputs.")
    print("        Δ = mean(B1) - mean(non-B1)  (negative = RM prefers unfaithful)\n")

    by_cat = defaultdict(lambda: {"b1":[], "non":[]} )
    for r in scored:
        cat = r["category"]
        if r["is_faithful"]:
            by_cat[cat]["b1"].append(r["rm_score"])
        else:
            by_cat[cat]["non"].append(r["rm_score"])

    print(f"{'Category':<30} {'n_B1':>5} {'n_non':>5} {'mean_B1':>9} {'mean_non':>9} {'Δ(B1-non)':>10} {'p-value':>9}")
    print("-"*85)

    group_data = defaultdict(lambda: {"b1":[], "non":[]})
    for cat in sorted(SAFETY | CAPABILITY | SUBJECTIVE):
        d = by_cat.get(cat)
        if not d or not d["b1"] or not d["non"]:
            continue
        b1  = np.array(d["b1"])
        non = np.array(d["non"])
        delta = np.mean(b1) - np.mean(non)
        _, pval = stats.ttest_ind(b1, non)
        grp = "SAFETY" if cat in SAFETY else ("CAPABILITY" if cat in CAPABILITY else "SUBJ")
        pstr = "<0.001" if pval < 0.001 else f"{pval:.3f}"
        print(f"{cat:<30} {len(b1):>5} {len(non):>5} {np.mean(b1):>9.3f} {np.mean(non):>9.3f} {delta:>+10.3f} {pstr:>9}  [{grp}]")
        group_data[grp]["b1"].extend(d["b1"])
        group_data[grp]["non"].extend(d["non"])

    print()
    print(f"{'Group':<12} {'n_B1':>5} {'n_non':>5} {'mean_B1':>9} {'mean_non':>9} {'Δ(B1-non)':>10} {'p-value':>9}")
    print("-"*65)
    summary = {}
    for grp in ["SAFETY","CAPABILITY","SUBJ"]:
        d = group_data.get(grp)
        if not d or not d["b1"] or not d["non"]:
            continue
        b1  = np.array(d["b1"])
        non = np.array(d["non"])
        delta = np.mean(b1) - np.mean(non)
        _, pval = stats.ttest_ind(b1, non)
        pstr = "<0.001" if pval < 0.001 else f"{pval:.3f}"
        print(f"{grp:<12} {len(b1):>5} {len(non):>5} {np.mean(b1):>9.3f} {np.mean(non):>9.3f} {delta:>+10.3f} {pstr:>9}")
        summary[grp] = {"mean_b1": float(np.mean(b1)), "mean_non": float(np.mean(non)),
                         "delta": float(delta), "p_value": float(pval),
                         "n_b1": len(b1), "n_non": len(non)}

    print("\n" + "="*100)
    saf = summary.get("SAFETY", {})
    cap = summary.get("CAPABILITY", {})
    if saf and cap:
        print(f"  Safety     Δ(B1-non) = {saf['delta']:+.4f}  (p={saf['p_value']:.4g})")
        print(f"  Capability Δ(B1-non) = {cap['delta']:+.4f}  (p={cap['p_value']:.4g})")
        if saf["delta"] < cap["delta"]:
            print("\n  ✓ CONFIRMED: RM penalises B1 outputs more in SAFETY than CAPABILITY.")
            print("    This directly quantifies the asymmetric penalty.")
        else:
            print("\n  ~ NOTE: Safety Δ ≥ Capability Δ.")
    return summary


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = load_data()
    model, tok = load_rm()
    scored = score_all(items, model, tok)
    out_path = os.path.join(OUTPUT_DIR, "rm_within_step_results.jsonl")
    with open(out_path, "w") as f:
        for r in scored:
            f.write(json.dumps({k:v for k,v in r.items() if k!="doc"}) + "\n")
    summary = analyze(scored)
    with open(os.path.join(OUTPUT_DIR, "within_step_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
