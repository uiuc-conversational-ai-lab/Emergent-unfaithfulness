# Real-world validity: naturalistic Reddit corpus

Reproduces the "Naturalistic documents" paragraph (Section: Beyond the
Controlled Summarization Setting) and Appendix H. Evaluates seven models
(Claude Sonnet 4.5, GPT-4o, DeepSeek-V3.2, Gemma-3-12B/27B, Llama-3.1-8B/70B-
Instruct) on real health-misinformation Reddit posts, under direct and CoT
prompts without a system prompt. Because naturalistic posts have no
confirming counterpart, FaithGap cannot be computed here. The pipeline
reports absolute unfaithfulness rates instead, a lower bound on AIU
prevalence.

Run in order:

```
01_load_and_filter.py     Load the HF health-misinformation dataset, keep posts >= 150 chars
02_generate_responses.py  Query all seven models under direct/CoT/mitigated prompts
03a_judge_b_labels.py     Apply the B1-B8 behavior taxonomy to every response
03b_judge_c_labels.py     Apply the C0-C6 reasoning taxonomy to CoT responses
04_aggregate_results.py   Build summary tables
```

All steps are resumable. Already-completed items are skipped on rerun.

`02_generate_responses.py`, `03a_judge_b_labels.py`, and `03b_judge_c_labels.py`
read API keys from environment variables (`CONVAI_API_KEY`, `OPENROUTER_API_KEY`,
`HF_TOKEN`). Set these before running.

**Note**: the original run filtered to 347 posts at the >=150-char threshold;
the paper's Appendix H reports 342 posts after an additional dedup/truncation
step whose script we did not locate in this snapshot (see `../KNOWN_GAPS.md`).
The ~5-post discrepancy does not appear in any reported statistic we could
find, but flag it if you're trying to reproduce the exact corpus size.
