# Beyond summarization: task-format experiments

Reproduces Section "Beyond the Controlled Summarization Setting" and Appendix G.

- `experiment4_multitask.py`: QA, NLI, and Extraction across the open-weight
  checkpoints, using the prompts in `../prompts/format_prompts.py`.
- `experiment4_extraction_plus.py`: the "Extraction+" variant (extraction
  without the explicit "don't modify" reminder), which the paper reports as
  producing the strongest AIU signal.
- `frontier_multitask_inference.py`, `run_frontier_opus_gpt5.py`: the same
  four formats for the frontier and proprietary models. Both read API keys
  from environment variables (`CONVAI_API_KEY`, `OPENROUTER_API_KEY`). Set
  these before running; do not hardcode real keys back into these files.

Output: `../results/faithgap_results_formats.json` (open-weight) and
`../results/faithgap_results_frontier.json` (frontier).
