"""
Generate diverse faithful/unfaithful summary pairs for RM training using GPT-4o.

For each of the 940 FaithConflict instances (opposing/false direction),
generates 3 faithful variants and 3 unfaithful variants, giving ~5,640 pairs.

Key fix over v1: explicitly includes factual_counterfactual and health/safety
categories with strong faithful>unfaithful signal — fixing the RM's inverted
and noise-floor calibration on those categories.

Output: faithful_pairs_v2/rm_train_v2.jsonl  (RM format)
        faithful_pairs_v2/dpo_train_v2.jsonl  (DPO format)
        faithful_pairs_v2/stats_v2.json
"""

import json, os, time, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
# All paths are configurable via environment variables. DATA_DIR should point
# at this repo's data/ folder; WORK_DIR is a writable location for outputs.
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR    = os.environ.get("FAITHCONFLICT_DATA_DIR", os.path.join(REPO_ROOT, "data"))
WORK_DIR    = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))

FDATA_PATH  = os.path.join(DATA_DIR, "faithconflict.json")
OUT_DIR     = os.path.join(WORK_DIR, "faithful_pairs_v2")
LOG_PATH    = os.path.join(OUT_DIR, "generate.log")

OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = "https://modelrouter.sumuk.org/v1"
MODEL           = "gpt-4o-mini"   # fast + cheap; switch to gpt-4o for quality
MAX_WORKERS     = 8               # parallel API calls
N_FAITHFUL      = 3               # faithful variants per instance
N_UNFAITHFUL    = 3               # unfaithful variants per instance
TEMPERATURE     = 0.8             # diversity in generation
MAX_TOKENS      = 80

SUMMARIZE_PROMPT = "Below is a document. Summarize it in one sentence.\n\nDocument:\n{document}"

# ── categories ───────────────────────────────────────────────────────────────
SAFETY     = {"health_safety_misinfo","scientific_misinformation","social_bias","direct_social_bias"}
CAPABILITY = {"factual_counterfactual","math_reasoning","hard_math_reasoning"}
SUBJECTIVE = {"historical_moral","political_ideological","scientific_frontier"}

os.makedirs(OUT_DIR, exist_ok=True)
_log = open(LOG_PATH, "a")

def log(msg):
    print(msg, flush=True)
    _log.write(msg + "\n"); _log.flush()

# ── OpenAI client ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ── prompt templates ──────────────────────────────────────────────────────────

FAITHFUL_SYSTEM = """You are a document summarizer. Your ONLY job is to faithfully report
what a document says. Even if the document contains false or controversial claims,
you must report them accurately. Do NOT add corrections, caveats, or your own knowledge.
Write a single sentence that captures the document's stated conclusion."""

FAITHFUL_PROMPTS = [
    # Direct attribution
    "Summarize what this document concludes in one sentence. Report exactly what it says, even if you disagree with it.\n\nDocument:\n{document}",
    # Evidential framing
    "Write one sentence summarizing the main finding reported in this document. Use language like 'The document reports that...' or 'According to the study,...'. Do not add corrections.\n\nDocument:\n{document}",
    # Verbatim-style
    "In one sentence, state what conclusion this research document presents. Be faithful to the source — do not substitute your own knowledge.\n\nDocument:\n{document}",
]

UNFAITHFUL_SYSTEM = """You are a helpful assistant that corrects summaries of documents.
When a document contains a false claim, you should write a summary that states
the real-world truth rather than what the document says. Write a single sentence."""

UNFAITHFUL_PROMPTS = [
    # Silent inversion (B8) - most dangerous, should be rejected by RM
    "This document contains a false claim. Write a one-sentence summary that states the correct real-world information, ignoring what the document says.\n\nDocument:\n{document}\n\nThe actual correct fact is: {claim_true}",
    # Correction (B5) - partial faithful + correction
    "Write a one-sentence summary that first briefly mentions the document's claim, then corrects it with the real-world truth. Example: 'Although the document claims X, in fact Y.'\n\nDocument:\n{document}\n\nCorrect fact: {claim_true}",
    # Editorial (B4) - labels the claim as wrong
    "Write a one-sentence summary of this document that labels its main claim as incorrect or misleading, and states the real truth.\n\nDocument:\n{document}\n\nCorrect fact: {claim_true}",
]


def make_document(item):
    """Fill [CLAIM] in the template with the false claim."""
    return item["summary_template"].replace("[CLAIM]", item["claim_false"])


def call_api(system_prompt, user_prompt, temperature=TEMPERATURE, max_retries=3):
    """Call the OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def generate_pairs_for_instance(item):
    """Generate faithful and unfaithful summaries for one instance."""
    doc = make_document(item)
    iid = item["id"]
    cat = item["category"]
    cf  = item["claim_false"]
    ct  = item["claim_true"]

    faithful_outputs   = []
    unfaithful_outputs = []

    # Generate faithful variants
    for tmpl in FAITHFUL_PROMPTS[:N_FAITHFUL]:
        prompt = tmpl.format(document=doc)
        out = call_api(FAITHFUL_SYSTEM, prompt)
        if out:
            faithful_outputs.append({"text": out, "label": "B1", "source": "gpt4o_faithful"})

    # Generate unfaithful variants
    for tmpl in UNFAITHFUL_PROMPTS[:N_UNFAITHFUL]:
        prompt = tmpl.format(document=doc, claim_true=ct)
        out = call_api(UNFAITHFUL_SYSTEM, prompt)
        if out:
            unfaithful_outputs.append({"text": out, "label": "B8_or_B5", "source": "gpt4o_unfaithful"})

    # Build pairs
    pairs = []
    for f_out in faithful_outputs:
        for u_out in unfaithful_outputs:
            prompt_text = SUMMARIZE_PROMPT.format(document=doc)
            pairs.append({
                "id":            iid,
                "category":      cat,
                "claim_false":   cf,
                "claim_true":    ct,
                "prompt":        prompt_text,
                "chosen":        f_out["text"],
                "rejected":      u_out["text"],
                "chosen_label":  f_out["label"],
                "rejected_label": u_out["label"],
            })

    return iid, faithful_outputs, unfaithful_outputs, pairs


def main():
    # Load data
    with open(FDATA_PATH) as f:
        data = json.load(f)
    log(f"Loaded {len(data)} FaithConflict instances")

    # Check for existing progress
    done_ids = set()
    all_pairs = []
    progress_file = os.path.join(OUT_DIR, "pairs_progress.jsonl")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            for line in f:
                p = json.loads(line)
                done_ids.add(p["id"])
                all_pairs.append(p)
        log(f"Resuming: {len(done_ids)} instances already done, {len(all_pairs)} pairs loaded")

    remaining = [d for d in data if d["id"] not in done_ids]
    log(f"Generating pairs for {len(remaining)} remaining instances...")

    # Run generation in parallel
    pf = open(progress_file, "a")
    stats = defaultdict(lambda: {"instances": 0, "pairs": 0, "faithful": 0, "unfaithful": 0})

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_pairs_for_instance, item): item for item in remaining}
        for future in tqdm(as_completed(futures), total=len(remaining), desc="Generating"):
            try:
                iid, faithfuls, unfaithfuls, pairs = future.result()
                item = futures[future]
                cat = item["category"]
                stats[cat]["instances"] += 1
                stats[cat]["faithful"]  += len(faithfuls)
                stats[cat]["unfaithful"] += len(unfaithfuls)
                stats[cat]["pairs"]     += len(pairs)
                for p in pairs:
                    pf.write(json.dumps(p) + "\n")
                    all_pairs.append(p)
                pf.flush()
            except Exception as e:
                log(f"Error: {e}")

    pf.close()

    # Save final outputs
    log(f"\nTotal pairs generated: {len(all_pairs)}")

    # RM format (chosen/rejected message lists for TRL RewardTrainer)
    rm_pairs = []
    for p in all_pairs:
        rm_pairs.append({
            "id":          p["id"],
            "category":    p["category"],
            "chosen":   [{"role": "user", "content": p["prompt"]},
                         {"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "user", "content": p["prompt"]},
                         {"role": "assistant", "content": p["rejected"]}],
        })

    with open(os.path.join(OUT_DIR, "rm_train_v2.jsonl"), "w") as f:
        for p in rm_pairs:
            f.write(json.dumps(p) + "\n")

    # DPO format (prompt/chosen/rejected strings)
    with open(os.path.join(OUT_DIR, "dpo_train_v2.jsonl"), "w") as f:
        for p in all_pairs:
            f.write(json.dumps({
                "id":       p["id"],
                "category": p["category"],
                "prompt":   p["prompt"],
                "chosen":   p["chosen"],
                "rejected": p["rejected"],
            }) + "\n")

    # Stats
    total_stats = {"total_pairs": len(all_pairs), "by_category": dict(stats)}
    with open(os.path.join(OUT_DIR, "stats_v2.json"), "w") as f:
        json.dump(total_stats, f, indent=2)

    log("\nCategory breakdown:")
    for cat, s in sorted(stats.items()):
        log(f"  {cat}: {s['instances']} instances, {s['pairs']} pairs")

    log(f"\nSaved:")
    log(f"  {OUT_DIR}/rm_train_v2.jsonl  ({len(rm_pairs)} RM pairs)")
    log(f"  {OUT_DIR}/dpo_train_v2.jsonl ({len(all_pairs)} DPO pairs)")
    log(f"  {OUT_DIR}/stats_v2.json")


if __name__ == "__main__":
    main()
