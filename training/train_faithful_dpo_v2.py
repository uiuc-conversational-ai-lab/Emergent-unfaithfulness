"""
Train FaithDPO v2 using pairs scored by the new RM v2.
Base model: allenai/Llama-3.1-Tulu-3-8B-SFT
Evaluates on ALL 940 FaithConflict instances after training.
"""

import json, os, sys, gc, torch, datetime
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# All paths are configurable via environment variables. DATA_DIR should point
# at this repo's data/ folder; WORK_DIR is a writable location for the HF
# cache, intermediate pairs, and checkpoints.
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR    = os.environ.get("FAITHCONFLICT_DATA_DIR", os.path.join(REPO_ROOT, "data"))
WORK_DIR    = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))

LOCAL_HF_HOME = os.environ.get("HF_HOME", os.path.join(WORK_DIR, "hf_cache"))
os.environ["HF_HOME"] = LOCAL_HF_HOME
os.makedirs(os.path.join(LOCAL_HF_HOME, "hub"), exist_ok=True)

PAIRS_PATH  = os.path.join(WORK_DIR, "faithful_pairs_v2", "dpo_from_rm_v2.jsonl")
BASE_MODEL  = "allenai/Llama-3.1-Tulu-3-8B-SFT"
OUT_DIR     = os.path.join(WORK_DIR, "faithful_dpo_model_v2_clean")
LOG_PATH    = os.path.join(OUT_DIR, "train.log")
EVAL_DIR    = os.path.join(OUT_DIR, "eval")
FDATA_PATH  = os.path.join(DATA_DIR, "faithconflict.json")

# DPO hyperparams
BETA         = 0.1
LR           = 5e-7
EPOCHS       = 1
BATCH_SIZE   = 1
GRAD_ACCUM   = 128
MAX_LEN      = 512

# Inference hyperparams
EVAL_BATCH   = 16
MAX_NEW_TOK  = 50

SUMMARIZE_PROMPT = "Below is a document. Summarize it in one sentence.\n\nDocument:\n{document}"

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
_log = open(LOG_PATH, "a")

class _Tee:
    def __init__(self, *fs): self.fs = fs
    def write(self, d):
        for f in self.fs: f.write(d); f.flush()
    def flush(self):
        for f in self.fs: f.flush()
    def fileno(self): return self.fs[0].fileno()

sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)


class DPODataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs    = pairs
        self.tok      = tokenizer
        self.max_len  = max_len

    def __len__(self): return len(self.pairs)

    def _encode(self, prompt, response):
        msgs = [{"role":"user","content":prompt},{"role":"assistant","content":response}]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc  = self.tok(full, max_length=self.max_len, truncation=True,
                         padding="max_length", return_tensors="pt")
        # mask prompt tokens
        prompt_tok = self.tok(
            self.tok.apply_chat_template(
                [{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True),
            return_tensors="pt")
        prompt_len = prompt_tok["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100
        return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(), labels.squeeze()

    def __getitem__(self, idx):
        p = self.pairs[idx]
        ci, cm, cl = self._encode(p["prompt"], p["chosen"])
        ri, rm, rl = self._encode(p["prompt"], p["rejected"])
        return {"chosen_ids": ci, "chosen_mask": cm, "chosen_labels": cl,
                "rejected_ids": ri, "rejected_mask": rm, "rejected_labels": rl}


def dpo_loss(model, ref_logps_c, ref_logps_r, ids_c, mask_c, lab_c, ids_r, mask_r, lab_r, beta):
    def get_logps(ids, mask, labels):
        out    = model(ids, attention_mask=mask, labels=labels)
        logits = out.logits[:, :-1]
        tgt    = labels[:, 1:]
        valid  = tgt != -100
        lp = torch.nn.functional.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(2, tgt.clamp(min=0).unsqueeze(2)).squeeze(2)
        return (tok_lp * valid).sum(1) / valid.sum(1).clamp(min=1)

    pi_c = get_logps(ids_c, mask_c, lab_c)
    pi_r = get_logps(ids_r, mask_r, lab_r)
    ratio = beta * ((pi_c - ref_logps_c) - (pi_r - ref_logps_r))
    loss  = -torch.nn.functional.logsigmoid(ratio).mean()
    acc   = (ratio > 0).float().mean().item()
    return loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Started: {datetime.datetime.now()}")

    # Load pairs
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} DPO pairs")

    # Load tokenizer + model
    print(f"Loading {BASE_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules=["q_proj","v_proj"], bias="none")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto")
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad = False

    dataset = DPODataset(pairs, tok, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    optimizer = AdamW(model.parameters(), lr=LR)

    n_steps = (len(loader) * EPOCHS) // GRAD_ACCUM
    print(f"Training: {EPOCHS} epochs, {len(loader)} steps/epoch, {n_steps} optimizer steps")

    # Precompute reference log-probs
    print("Precomputing reference log-probs ...")
    ref_logps_c_all, ref_logps_r_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Ref logps"):
            ci = batch["chosen_ids"].to(device)
            cm = batch["chosen_mask"].to(device)
            cl = batch["chosen_labels"].to(device)
            ri = batch["rejected_ids"].to(device)
            rm = batch["rejected_mask"].to(device)
            rl = batch["rejected_labels"].to(device)

            def ref_logp(ids, mask, labels):
                out    = ref_model(ids, attention_mask=mask, labels=labels)
                logits = out.logits[:, :-1]
                tgt    = labels[:, 1:]
                valid  = tgt != -100
                lp = torch.nn.functional.log_softmax(logits, dim=-1)
                tok_lp = lp.gather(2, tgt.clamp(min=0).unsqueeze(2)).squeeze(2)
                return (tok_lp * valid).sum(1) / valid.sum(1).clamp(min=1)

            ref_logps_c_all.append(ref_logp(ci, cm, cl).cpu())
            ref_logps_r_all.append(ref_logp(ri, rm, rl).cpu())

    ref_logps_c_all = torch.cat(ref_logps_c_all)
    ref_logps_r_all = torch.cat(ref_logps_r_all)
    del ref_model; gc.collect(); torch.cuda.empty_cache()
    print("Reference log-probs computed.")

    # DPO training
    model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        running_loss, running_acc = 0.0, 0.0
        loader2 = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        for step, batch in enumerate(tqdm(loader2, desc=f"Epoch {epoch}/{EPOCHS}")):
            ci = batch["chosen_ids"].to(device)
            cm = batch["chosen_mask"].to(device)
            cl = batch["chosen_labels"].to(device)
            ri = batch["rejected_ids"].to(device)
            rm_m = batch["rejected_mask"].to(device)
            rl = batch["rejected_labels"].to(device)
            rc = ref_logps_c_all[step:step+1].to(device)
            rr = ref_logps_r_all[step:step+1].to(device)

            loss, acc = dpo_loss(model, rc, rr, ci, cm, cl, ri, rm_m, rl, BETA)
            (loss / GRAD_ACCUM).backward()
            running_loss += loss.item()
            running_acc  += acc

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 5 == 0:
                    avg = running_loss / (step + 1)
                    a   = running_acc  / (step + 1)
                    print(f"  step={global_step} loss={avg:.4f} acc={a:.3f}")

        print(f"Epoch {epoch} complete — step {global_step}")

    # Save model
    print(f"Saving FaithDPO v2 to {OUT_DIR}")
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"Model saved: {datetime.datetime.now()}")

    # ── Evaluation on all 940 instances ──────────────────────────────────────
    print("\nRunning inference on all 940 FaithConflict instances ...")
    model.eval()

    with open(FDATA_PATH) as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc="Inference"):
        iid  = item["id"]
        cf   = item["claim_false"]
        ct   = item["claim_true"]
        cat  = item["category"]
        tmpl = item["summary_template"]

        for direction, claim in [("false", cf), ("true", ct)]:
            doc    = tmpl.replace("[CLAIM]", claim)
            prompt = SUMMARIZE_PROMPT.format(document=doc)
            msgs   = [{"role": "user", "content": prompt}]
            enc_ids = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc    = tok(enc_ids, return_tensors="pt", padding=False)
            enc    = {k: v.to(device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOK,
                    do_sample=False, temperature=1.0, pad_token_id=tok.eos_token_id)
            generated = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
            results.append({
                "id": iid, "direction": direction, "category": cat,
                "claim_in_doc": claim, "claim_false": cf, "claim_true": ct,
                "document": doc[:300], "output": generated,
                "model": "FaithDPO_v2", "system_prompt": False,
            })

    # Save inference outputs
    for direction in ["false", "true"]:
        out_path = os.path.join(EVAL_DIR, f"{direction}__direct.jsonl")
        with open(out_path, "w") as f:
            for r in results:
                if r["direction"] == direction:
                    f.write(json.dumps(r) + "\n")
        print(f"Saved {direction} outputs to {out_path}")

    print(f"\nInference complete: {datetime.datetime.now()}")
    print(f"Total outputs: {len(results)}")
    print(f"Next step: judge with Qwen2.5-32B and compute FaithGap")


if __name__ == "__main__":
    main()
