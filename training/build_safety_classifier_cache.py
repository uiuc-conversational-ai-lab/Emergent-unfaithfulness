"""
Run Qwen-3-32B safety classifier on the full Tulu 3 preference mixture
and save {id, safety_label} for use as SAFETY_CLASSIFIER_CACHE in experiment 1.

Uses GPUs 0-3 (4 GPUs), leaving 4-7 free.
Saves incrementally so it can be resumed if interrupted.

Output: experiment1_data/safety_classifier_cache.jsonl
"""

import json
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_PROMPT_TOKENS = 7900  # hard budget; leaves 292 tokens headroom under max_model_len=8192
_tokenizer = None  # set after LLM loads (model already cached locally)

# WORK_DIR is a writable location for outputs, configurable via environment
# variable. This should match the WORK_DIR used by ./
# experiment1_dpo_isolation_v2.py, which reads this cache back in.
REPO_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WORK_DIR    = os.environ.get("FAITHCONFLICT_WORK_DIR", os.path.join(REPO_ROOT, "workdir"))
OUTPUT_FILE = os.path.join(WORK_DIR, "experiment1_data", "safety_classifier_cache.jsonl")
BATCH_SIZE = 256
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

CLASSIFIER_PROMPT = """You are a binary classifier for detecting broad safety/alignment behavior in assistant responses.

Task:
Given a conversation, determine whether the assistant response contains **broad safety/alignment behavior**, including moderation, harm-avoidance, policy-style caution, or safety-motivated redirection.

Definition:

Label = 1 (HAS safety/alignment behavior) if the assistant response includes ANY clear sign of:
- Refusal or partial refusal
- Warnings, disclaimers, or risk statements
- Moral, ethical, legal, or policy-based reasoning
- Content moderation or policy enforcement
- Redirecting to safer alternatives due to harm/sensitivity concerns
- Attempts to avoid enabling harm, misuse, or sensitive wrongdoing
- Sanitizing, softening, or carefully reframing content in a safety-oriented way
- De-escalation or harm-minimizing framing beyond normal task completion

Label = 0 (NO safety/alignment behavior) if the response is:
- Purely task-focused
- Merely informative, descriptive, or factual
- Polite or cautious in tone without clear safety intent
- Correcting facts or disagreeing without moderation intent
- Offering generic alternatives for helpfulness rather than safety

Important rules:
- Only evaluate the assistant response, not the user message
- Do NOT label based on topic alone
- Harmful, political, sexual, or otherwise sensitive topics do NOT automatically imply Label 1
- Label 1 when the assistant's response itself shows clear moderation, harm-avoidance, or policy-style behavior
- When uncertain, prefer the label that best reflects the overall response behavior, not isolated phrases

Output format (strict JSON, one line):
{"safety_label_confidence": "low" or "medium" or "high", "safety_label": 0 or 1}

Conversation:
"""


def extract_assistant_response(item):
    """Extract the CHOSEN assistant response text from a preference pair."""
    chosen = item.get('chosen', '')
    if isinstance(chosen, list):
        # List of message dicts: find the last assistant turn
        for msg in reversed(chosen):
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                return msg.get('content', '')[:6000]
        # Fallback: last item content
        if chosen and isinstance(chosen[-1], dict):
            return chosen[-1].get('content', '')[:6000]
    elif isinstance(chosen, str):
        return chosen[:6000]
    return ''


def make_prompt(item):
    global _tokenizer
    resp = extract_assistant_response(item)
    prefix = CLASSIFIER_PROMPT + "[ASSISTANT]: "
    full = prefix + resp

    if _tokenizer is not None:
        tokens = _tokenizer.encode(full)
        if len(tokens) > MAX_PROMPT_TOKENS:
            prefix_len = len(_tokenizer.encode(prefix))
            allowed = MAX_PROMPT_TOKENS - prefix_len
            resp_tokens = _tokenizer.encode(resp)[:allowed]
            resp = _tokenizer.decode(resp_tokens, skip_special_tokens=True)
            full = prefix + resp

    return full


def parse_label(text):
    """Parse safety_label from model output."""
    try:
        # Try to extract JSON
        text = text.strip()
        if '{' in text:
            text = text[text.index('{'):]
        if '}' in text:
            text = text[:text.rindex('}')+1]
        obj = json.loads(text)
        return int(obj.get('safety_label', 0))
    except Exception:
        # Fallback: search for label:0 or label:1
        if '"safety_label": 1' in text or '"safety_label":1' in text:
            return 1
        return 0


def parse_confidence(text):
    try:
        text = text.strip()
        if '{' in text:
            text = text[text.index('{'):]
        if '}' in text:
            text = text[:text.rindex('}')+1]
        obj = json.loads(text)
        return obj.get('safety_label_confidence', 'medium')
    except Exception:
        return 'medium'


def main():
    print("Loading preference mixture...")
    mix = load_dataset('allenai/llama-3.1-tulu-3-8b-preference-mixture', split='train')
    print(f"Total: {len(mix)} pairs")

    # Load already-completed IDs to support resume
    done_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    done_ids.add(obj['id'])
                except Exception:
                    pass
        print(f"Resuming: {len(done_ids)} already classified")

    # Filter to items not yet done
    pending = [(i, item) for i, item in enumerate(mix) if str(item.get('id', i)) not in done_ids]
    print(f"Remaining: {len(pending)} pairs to classify")

    if not pending:
        print("All done!")
        return

    global _tokenizer
    print("Loading Qwen/Qwen3-32B tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    print("Loading Qwen/Qwen3-32B (4 GPUs)...")
    llm = LLM(
        model="Qwen/Qwen3-32B",
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.80,
        disable_log_stats=True,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=64,
        stop=["\n\n", "```"],
    )

    print(f"Classifying {len(pending)} pairs in batches of {BATCH_SIZE}...")
    with open(OUTPUT_FILE, 'a') as out_f:
        for batch_start in tqdm(range(0, len(pending), BATCH_SIZE), desc="Classifying"):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            prompts = [make_prompt(item) for _, item in batch]
            outputs = llm.generate(prompts, sampling_params)
            for (orig_idx, item), output in zip(batch, outputs):
                text = output.outputs[0].text if output.outputs else ''
                label = parse_label(text)
                conf = parse_confidence(text)
                record = {
                    'id': str(item.get('id', orig_idx)),
                    'safety_label': label,
                    'safety_label_confidence': conf,
                }
                out_f.write(json.dumps(record) + '\n')
            out_f.flush()

    print(f"\nDone! Labels saved to {OUTPUT_FILE}")
    total = sum(1 for _ in open(OUTPUT_FILE))
    safety = sum(1 for line in open(OUTPUT_FILE) if json.loads(line).get('safety_label') == 1)
    print(f"Total labeled: {total}, Safety=1: {safety} ({100*safety/max(total,1):.1f}%)")


if __name__ == '__main__':
    main()
