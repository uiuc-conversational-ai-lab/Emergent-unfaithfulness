"""
Score Tulu checkpoint outputs with the new RM v2 and build clean DPO pairs.
Uses all checkpoint steps (100-1000) for better coverage.
Saves to faithful_pairs_v2/dpo_from_rm_v2.jsonl
"""

import json, os, torch, glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from tqdm import tqdm

# All paths are configurable via environment variables. DATA_DIR should point
# at this repo's data/ folder; WORK_DIR is a writable location for the HF
# cache, models, and outputs; CHECKPOINTS_DIR holds the intermediate Tulu
# checkpoint outputs from the stage-analysis run (see ../stage_analysis/).
REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DATA_DIR        = os.environ.get("FAITHCONFLICT_DATA_DIR", os.path.join(REPO_ROOT, "data"))
WORK_DIR        = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))
CHECKPOINTS_DIR = os.environ.get("FAITHCONFLICT_CHECKPOINTS_DIR", os.path.join(WORK_DIR, "checkpoints"))

LOCAL_HF_HOME = os.environ.get("HF_HOME", os.path.join(WORK_DIR, "hf_cache"))
os.environ["HF_HOME"] = LOCAL_HF_HOME

RM_PATH    = os.path.join(WORK_DIR, "faithful_rm_v2")
RM_BASE    = "allenai/Llama-3.1-Tulu-3-8B-RM"  # same base as training
CHK_BASE   = os.path.join(CHECKPOINTS_DIR, "without_system")
CHK_JUDGE  = os.path.join(CHECKPOINTS_DIR, "without_system_judge")
FDATA_PATH = os.path.join(DATA_DIR, "faithconflict.json")
OUT_DIR    = os.path.join(WORK_DIR, "faithful_pairs_v2")
OUT_PAIRS  = os.path.join(OUT_DIR, "dpo_from_rm_v2.jsonl")

DIRECT_PROMPT = "Below is a document. Summarize it in one sentence.\n\nDocument:\n{document}"
BATCH_SIZE    = 32
MAX_LEN       = 512
MIN_MARGIN    = 0.3   # minimum score difference to include pair


def score_texts(model, tok, device, prompts, responses):
    """Score (prompt, response) pairs with the RM."""
    scores = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_p = prompts[i:i+BATCH_SIZE]
        batch_r = responses[i:i+BATCH_SIZE]
        texts = []
        for p, r in zip(batch_p, batch_r):
            msgs = [{"role":"user","content":p},{"role":"assistant","content":r}]
            texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        enc = tok(texts, max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc).logits.squeeze(-1)
        scores.extend(out.cpu().float().tolist())
    return scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load RM
    print(f"Loading RM v2 from {RM_PATH} ...")
    # The RM v2 was trained from allenai/Llama-3.1-Tulu-3-8B-RM and saved merged
    tok = AutoTokenizer.from_pretrained(RM_PATH)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        RM_PATH, dtype=torch.bfloat16, device_map="auto", num_labels=1)
    model.eval()
    print("RM v2 loaded.")

    # Load templates
    with open(FDATA_PATH) as f:
        templates = {d["id"]: (d["summary_template"], d["claim_false"]) for d in json.load(f)}

    # Collect all checkpoint outputs with judge labels
    by_id = {}   # id -> {"faithful": [outputs], "unfaithful": [outputs]}
    for step_dir in sorted(glob.glob(os.path.join(CHK_BASE, "*"))):
        step  = os.path.basename(step_dir)
        fpath = os.path.join(step_dir,   "false__direct.jsonl")
        jpath = os.path.join(CHK_JUDGE, step, "false__direct.jsonl")
        if not (os.path.exists(fpath) and os.path.exists(jpath)):
            continue

        outputs = {}
        with open(fpath) as f:
            for line in f:
                d = json.loads(line)
                outputs[d["id"]] = d["output"]

        with open(jpath) as f:
            for line in f:
                d = json.loads(line)
                iid   = d["id"]
                label = d.get("judge_label","")
                out   = outputs.get(iid,"")
                if not out: continue
                if iid not in by_id:
                    by_id[iid] = {"faithful": [], "unfaithful": []}
                if label in ("B1","B2"):
                    by_id[iid]["faithful"].append(out)
                elif label in ("B3","B4","B5","B6","B7","B8"):
                    by_id[iid]["unfaithful"].append(out)

    print(f"Loaded outputs for {len(by_id)} instances")

    # Score and build pairs
    pairs      = []
    skipped    = 0

    for iid, buckets in tqdm(by_id.items(), desc="Scoring pairs"):
        if not buckets["faithful"] or not buckets["unfaithful"]:
            continue
        tmpl, cf = templates.get(iid, (None, None))
        if not tmpl: continue
        doc    = tmpl.replace("[CLAIM]", cf)
        prompt = DIRECT_PROMPT.format(document=doc)

        faithfuls   = list(set(buckets["faithful"]))[:3]
        unfaithfuls = list(set(buckets["unfaithful"]))[:3]

        # Score all candidates
        all_texts  = faithfuls + unfaithfuls
        all_scores = score_texts(model, tok, device,
                                  [prompt]*len(all_texts), all_texts)
        f_scores = all_scores[:len(faithfuls)]
        u_scores = all_scores[len(faithfuls):]

        # Build pairs where RM correctly prefers faithful AND margin >= threshold
        for f_out, f_sc in zip(faithfuls, f_scores):
            for u_out, u_sc in zip(unfaithfuls, u_scores):
                margin = f_sc - u_sc
                if margin >= MIN_MARGIN:
                    pairs.append({
                        "id":       iid,
                        "prompt":   prompt,
                        "chosen":   f_out,
                        "rejected": u_out,
                        "chosen_score":   round(float(f_sc), 4),
                        "rejected_score": round(float(u_sc), 4),
                        "margin":         round(float(margin), 4),
                    })
                else:
                    skipped += 1

    print(f"Built {len(pairs)} pairs (skipped {skipped} below margin threshold)")

    with open(OUT_PAIRS, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved to {OUT_PAIRS}")


if __name__ == "__main__":
    main()
