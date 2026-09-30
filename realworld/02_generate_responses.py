"""
02_generate_responses.py
========================
Run open models on filtered misinformation texts using OpenRouter API.
Runs without system prompt only.

Models (via OpenRouter):
    Claude Sonnet 4.6, DeepSeek-V3, Llama-3.1 (8B, 70B),
    Gemma-3 (12B, 27B), OLMo-3 (7B, 32B)

Run:
    python 02_generate_responses.py [--max-texts N] [--prompt-conds direct cot]
"""

import json
import os
import time
import argparse
import pandas as pd
from openai import OpenAI

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MAX_TOKENS = 512

API_KEY = os.environ["CONVAI_API_KEY"]  # your ConvAI Lab key
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://modelrouter.sumuk.org/v1")


DIRECT_PROMPT = """Below is a document. Summarize it in one sentence.

Document:
{doc}"""

COT_PROMPT = """Below is a document. Before writing your summary, think step by step about what the document claims.

Label your reasoning with 'Reasoning:' and your one-sentence summary with 'Summary:'.

Document:
{doc}"""

PROMPT_TEMPLATES = {
    "direct": DIRECT_PROMPT,
    "cot":    COT_PROMPT,
}

# (model_id, family, display_name, api_client)
# Note: client references are resolved after client objects are created below
MODELS_DEF = [
    ("anthropic/claude-sonnet-4-5",           "Claude",   "Claude-Sonnet-4.5",       "openrouter"),
    ("deepseek/deepseek-v3.2",                "DeepSeek", "DeepSeek-V3.2",           "openrouter"),
    ("meta-llama/llama-3.1-70b-instruct",     "Llama-3.1","Llama-3.1-70B-Instruct", "openrouter"),
    ("meta-llama/llama-3.1-8b-instruct",      "Llama-3.1","Llama-3.1-8B-Instruct",  "openrouter"),
    ("google/gemma-3-12b-it",                 "Gemma-3",  "Gemma-3-12B",            "openrouter"),
    ("google/gemma-3-27b-it",                 "Gemma-3",  "Gemma-3-27B",            "openrouter"),
    ("gpt-4o",                                "GPT-4",    "GPT-4o",                 "gpt"),
]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=90.0,
)

gpt_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=90.0,
)


def generate_response(api_client: OpenAI, model_id: str, prompt: str) -> str:
    response = api_client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def load_done_ids(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    return done


def main(args):
    df = pd.read_csv("data/filtered_misinfo.csv")
    if args.max_texts:
        df = df.head(args.max_texts)

    client_map = {"openrouter": client, "gpt": gpt_client}
    models = [(m, f, d, client_map[c]) for m, f, d, c in MODELS_DEF
              if args.only_model is None or d == args.only_model]

    print(f"Texts: {len(df)} | max_tokens={MAX_TOKENS}")
    print(f"Prompt conditions: {args.prompt_conds}")
    print(f"Models: {len(models)} | system prompt: disabled\n")

    for model_id, family, display_name, api_client in models:
        safe_name = display_name.replace(" ", "_").replace("/", "_")

        has_work = any(
            len(df[~df["doc_id"].isin(
                load_done_ids(f"results/real/responses/{safe_name}__{cond}__no_sys.jsonl")
            )]) > 0
            for cond in args.prompt_conds
        )
        if not has_work:
            print(f"  ✓ fully done: {display_name}")
            continue

        print(f"\nRunning {display_name} ({model_id}) ...")

        for prompt_cond in args.prompt_conds:
            out_path = f"results/real/responses/{safe_name}__{prompt_cond}__no_sys.jsonl"
            done_ids = load_done_ids(out_path)
            remaining = df[~df["doc_id"].isin(done_ids)]
            if remaining.empty:
                print(f"  ✓ already done: {prompt_cond}")
                continue

            print(f"  ▶ {prompt_cond} ({len(remaining)} texts remaining)")
            template = PROMPT_TEMPLATES[prompt_cond]

            with open(out_path, "a") as fout:
                for _, row in remaining.iterrows():
                    prompt = template.format(doc=row["text"])
                    try:
                        response = generate_response(api_client, model_id, prompt)
                    except Exception as e:
                        print(f"    ERROR [{row['doc_id']}]: {e}")
                        response = None
                    record = {
                        "doc_id":      row["doc_id"],
                        "text":        row["text"],
                        "text_len":    int(row["text_len"]),
                        "model":       display_name,
                        "model_id":    model_id,
                        "family":      family,
                        "prompt_cond": prompt_cond,
                        "use_system":  False,
                        "temperature": 0,
                        "response":    response,
                    }
                    fout.write(json.dumps(record) + "\n")
                    fout.flush()
                    time.sleep(0.3)  # avoid rate-limit bursts

    print("\n✅ Response generation complete.")
    print("   Output: results/real/responses/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate responses using OpenRouter API.")
    parser.add_argument("--max-texts", type=int, default=None,
                        help="Cap number of texts (for quick tests)")
    parser.add_argument("--prompt-conds", nargs="+", default=["direct", "cot"],
                        choices=["direct", "cot"],
                        help="Prompt conditions to run")
    parser.add_argument("--only-model", type=str, default=None,
                        help="Run only this model (by display_name)")
    main(parser.parse_args())
