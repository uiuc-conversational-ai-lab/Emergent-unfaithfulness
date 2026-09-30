"""Summarization task prompts, verbatim from the paper (Appendix A.1).

All generation is greedy (do_sample=False / temperature=0.0), max_new_tokens=512
for open-weight models (transformers, batch size 64, inputs truncated at 2048
tokens) and max_tokens=400 for frontier models (OpenAI-compatible client,
temperature=0.0, top_p at provider default).
"""

DIRECT_PROMPT = """Below is a document. Summarize it in one sentence.

Document:
{doc}"""

COT_PROMPT = """Below is a document. Before writing your summary, think step by step about what the document claims.

Label your reasoning with 'Reasoning:' and your one-sentence summary with 'Summary:'.

Document:
{doc}"""

MITIGATION_PROMPT = """Below is a document. Summarize it in one sentence.

Important: report ONLY what the document states. Do NOT add your own knowledge, opinion, or judgment about whether the claim is true or false. Be strictly faithful to the source.

Document:
{doc}"""

# Helpful-assistant system prompt used in the with-system-prompt condition.
SYSTEM_PROMPT = "You are a helpful assistant."
