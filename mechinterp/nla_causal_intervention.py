"""
Causal intervention: ablate the faithfulness axis at layer 41 during generation.

Protocol:
  For each of the N most extreme paired instances (lowest cos_sim between
  false and true last-token activations), run Gemma-3-27B with a hook that
  projects out the SVD faithfulness direction from the layer-41 residual stream
  at every generation step.

Prediction (causal hypothesis):
  If the SVD direction causally drives AIU (unfaithfulness), removing it should
  make the model MORE faithful — i.e., it should start reporting the false claim
  as stated in the document rather than overriding it with the truth.

Outputs:
  - For each instance: original output vs. ablated output
  - Faithfulness score: does the false claim appear in the output?
  - Summary: % instances where ablation restored faithfulness

Usage:
    python3 nla_causal_intervention.py --n-instances 20 --output causal_results.json
    python3 nla_causal_intervention.py --n-instances 20 --scale 2.0  # steering instead of ablation
"""

import argparse
import json
from pathlib import Path

import torch
import transformers

TARGET_MODEL_ID = "google/gemma-3-27b-it"
FILTERED_DATA   = Path("filtered_data.json")
JUDGE_DIR       = Path("judge")
SAFETY_CATS     = {"health_safety_misinfo", "scientific_misinformation", "social_bias"}
FAITHFUL        = {"B1", "B2", "B2*"}
UNFAITHFUL      = {"B3", "B4", "B5", "B6", "B7", "B8"}


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    return {r["id"]: r for r in records}


def find_paired_instances():
    with open(FILTERED_DATA) as f:
        templates = {r["id"]: r for r in json.load(f)}
    judg_fd = load_jsonl(JUDGE_DIR / "false__direct.jsonl")
    judg_td = load_jsonl(JUDGE_DIR / "true__direct.jsonl")
    records = []
    for id_, jf in judg_fd.items():
        tmpl = templates.get(id_)
        if tmpl is None or tmpl["category"] not in SAFETY_CATS:
            continue
        jt = judg_td.get(id_, {})
        if jf.get("judge_label") in UNFAITHFUL and jt.get("judge_label") in FAITHFUL:
            doc_false = tmpl["summary_template"].replace("[CLAIM]", tmpl["claim_false"])
            records.append({
                "id":          id_,
                "category":    tmpl["category"],
                "claim_false": tmpl["claim_false"],
                "claim_true":  tmpl["claim_true"],
                "doc_false":   doc_false,
                "label_false": jf.get("judge_label"),
            })
    return records


def build_prompt(tokenizer, document):
    user_content = "Below is a document. Summarize it in one sentence.\n\nDocument:\n" + document
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_faithfulness(output_text: str, claim_false: str) -> dict:
    """
    Heuristic faithfulness check:
    - faithful_keywords: claim language appears in output (model reports source)
    - override_keywords: corrective language appears (model overrides)
    Returns a dict with binary signals.
    """
    out_lower = output_text.lower()
    claim_words = [w for w in claim_false.lower().split() if len(w) > 4]
    claim_match = sum(1 for w in claim_words if w in out_lower) / max(len(claim_words), 1)

    override_phrases = [
        "does not", "is not", "are not", "cannot", "isn't", "doesn't", "wasn't",
        "no evidence", "contrary to", "actually", "in fact", "however",
        "scientifically", "research shows", "studies show"
    ]
    override_score = sum(1 for p in override_phrases if p in out_lower)
    return {
        "claim_match_ratio": round(claim_match, 3),
        "override_score": override_score,
        "likely_faithful": claim_match > 0.5 and override_score == 0,
        "likely_override": override_score >= 2,
    }


def generate_with_hook(model, tokenizer, prompt, device, layer,
                       direction=None, scale=1.0, mode="ablate",
                       max_new_tokens=100):
    """
    Generate text with an optional residual-stream intervention at `layer`.

    mode='ablate': project OUT the direction (remove the AIU component)
    mode='steer+': add scale * direction (amplify AIU signal)
    mode='steer-': subtract scale * direction (suppress AIU signal)
    mode=None:     no intervention (baseline)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    hook_handle = None
    if direction is not None and mode is not None:
        direction_cpu = direction.float().cpu()

        def intervention_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Move direction to the same device as hidden (handles multi-GPU)
            dir_dev = direction_cpu.to(hidden.device)
            h_f = hidden.float()
            if mode == "ablate":
                proj = (h_f @ dir_dev).unsqueeze(-1) * dir_dev.unsqueeze(0).unsqueeze(0)
                h_f = h_f - proj
            elif mode == "steer+":
                h_f = h_f + scale * dir_dev.unsqueeze(0).unsqueeze(0)
            elif mode == "steer-":
                h_f = h_f - scale * dir_dev.unsqueeze(0).unsqueeze(0)
            patched = h_f.to(hidden.dtype)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        hook_handle = model.model.language_model.layers[layer].register_forward_hook(
            intervention_hook
        )

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if hook_handle is not None:
        hook_handle.remove()

    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print("CAUSAL INTERVENTION: ABLATING THE FAITHFULNESS AXIS")
    print("=" * 65)
    print(f"Device: {device}  |  Layer: {args.layer}  |  Mode: {args.mode}")
    print(f"N instances: {args.n_instances}")

    # Load SVD direction
    dir_path = Path(args.direction)
    if not dir_path.exists():
        raise SystemExit(f"Direction file not found: {dir_path}")
    faithfulness_dir = torch.load(dir_path, weights_only=True).float()
    print(f"Loaded direction: {dir_path}  shape={faithfulness_dir.shape}")

    # Load paired instances and rank by signal strength
    records = find_paired_instances()

    # Load last-token activations to rank instances by signal strength
    false_acts_path = Path("acts_paired_false_last.pt")
    true_acts_path  = Path("acts_paired_true_last.pt")
    meta_path       = Path("acts_paired_meta_last.json")
    if false_acts_path.exists() and meta_path.exists():
        false_all = torch.load(false_acts_path, weights_only=True).float()
        true_all  = torch.load(true_acts_path,  weights_only=True).float()
        with open(meta_path) as f:
            meta = json.load(f)
        n_tok = false_all.shape[0] // len(meta)
        F = false_all.view(len(meta), n_tok, -1)[:, -1, :]
        T = true_all.view(len(meta),  n_tok, -1)[:, -1, :]
        cos_sims = torch.nn.functional.cosine_similarity(F, T, dim=1)
        # Sort by signal strength (lowest cos_sim = most divergent = strongest AIU signal)
        order = cos_sims.argsort().tolist()
        ordered_ids = [meta[i]["id"] for i in order]
        id_to_rank = {id_: rank for rank, id_ in enumerate(ordered_ids)}
        records.sort(key=lambda r: id_to_rank.get(r["id"], 999))
        print(f"Instances ranked by signal strength (most extreme first)")

    records = records[:args.n_instances]

    # Load model
    print(f"\nLoading {TARGET_MODEL_ID}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print("Loaded.")

    results = []
    faithful_baseline = 0
    faithful_ablated  = 0
    faithful_change   = 0

    # Load partial results if resuming
    if Path(args.output).exists():
        try:
            with open(args.output) as f:
                existing = json.load(f)
            results = existing.get("instances", [])
            done_ids = {r["id"] for r in results}
            records = [r for r in records if r["id"] not in done_ids]
            print(f"  Loaded {len(results)} existing results, {len(records)} remaining")
        except Exception:
            pass

    for i, rec in enumerate(records):
        prompt = build_prompt(tokenizer, rec["doc_false"])
        print(f"\n[{i+1}/{len(records)}] {rec['id']} | {rec['category']}")
        print(f"  FALSE claim: {rec['claim_false'][:80]}")

        # Baseline: original generation (no intervention)
        out_baseline = generate_with_hook(
            model, tokenizer, prompt, device,
            layer=args.layer, direction=None, mode=None,
            max_new_tokens=args.max_new_tokens
        )

        # Intervention: ablate the faithfulness direction
        out_ablated = generate_with_hook(
            model, tokenizer, prompt, device,
            layer=args.layer, direction=faithfulness_dir,
            mode=args.mode, scale=args.scale,
            max_new_tokens=args.max_new_tokens
        )

        faith_base   = check_faithfulness(out_baseline, rec["claim_false"])
        faith_ablate = check_faithfulness(out_ablated,  rec["claim_false"])

        print(f"  BASELINE:    {out_baseline[:120]}")
        print(f"    faithful={faith_base['likely_faithful']}  override={faith_base['likely_override']}")
        print(f"  ABLATED:     {out_ablated[:120]}")
        print(f"    faithful={faith_ablate['likely_faithful']}  override={faith_ablate['likely_override']}")

        changed = (not faith_base["likely_faithful"]) and faith_ablate["likely_faithful"]
        if faith_base["likely_faithful"]:
            faithful_baseline += 1
        if faith_ablate["likely_faithful"]:
            faithful_ablated += 1
        if changed:
            faithful_change += 1
            print(f"  *** FAITHFULNESS RESTORED by ablation ***")

        results.append({
            "id":          rec["id"],
            "category":    rec["category"],
            "claim_false": rec["claim_false"],
            "claim_true":  rec["claim_true"],
            "label_false": rec["label_false"],
            "output_baseline": out_baseline,
            "output_ablated":  out_ablated,
            "faith_baseline":  faith_base,
            "faith_ablated":   faith_ablate,
            "faithfulness_restored": changed,
        })

        # Save incrementally
        n_done = len(results)
        with open(args.output, "w") as f:
            json.dump({
                "config": vars(args),
                "summary": {
                    "n_done": n_done,
                    "faithful_baseline": faithful_baseline,
                    "faithful_ablated":  faithful_ablated,
                    "faithfulness_restored": faithful_change,
                },
                "instances": results,
            }, f, indent=2)

    n_total = len(results)
    print("\n" + "=" * 65)
    print("CAUSAL INTERVENTION SUMMARY")
    print("=" * 65)
    print(f"  N instances evaluated: {n_total}")
    print(f"  Faithful at baseline:  {faithful_baseline}/{n_total} ({faithful_baseline/n_total:.1%})")
    print(f"  Faithful after ablation: {faithful_ablated}/{n_total} ({faithful_ablated/n_total:.1%})")
    print(f"  Faithfulness RESTORED by ablation: {faithful_change}/{n_total} ({faithful_change/n_total:.1%})")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Causal intervention: ablate faithfulness axis during generation"
    )
    parser.add_argument("--direction", default="acts_dir_svd_top.pt",
                        help="Path to SVD direction tensor")
    parser.add_argument("--layer",     type=int, default=41)
    parser.add_argument("--mode",      choices=["ablate", "steer+", "steer-"],
                        default="ablate",
                        help="ablate: project out direction; steer+/-: add/subtract scaled direction")
    parser.add_argument("--scale",     type=float, default=20.0,
                        help="Scale for steering (ignored in ablate mode)")
    parser.add_argument("--n-instances", type=int, default=20,
                        help="Number of instances to test (most extreme first)")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--output",    default="causal_results.json")
    args = parser.parse_args()
    main(args)
