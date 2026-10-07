# Beyond summarization

QA, NLI, and Extraction(+) experiments (Appendix G).

- `experiment4_multitask.py`: open-weight checkpoints.
- `experiment4_extraction_plus.py`: the "Extraction+" variant.
- `frontier_multitask_inference.py`, `run_frontier_opus_gpt5.py`: frontier/proprietary models (reads `CONVAI_API_KEY`, `OPENROUTER_API_KEY` from env).

Output: `../results/faithgap_results_formats.json`, `../results/faithgap_results_frontier.json`.
