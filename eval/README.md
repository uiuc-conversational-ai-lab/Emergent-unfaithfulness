# Main evaluation pipeline

`experiment4_run.py` generates, judges, and scores all 22 model checkpoints
(Section 3.4, Table categories_full's model list) under Direct, CoT, and
Mitigated prompts, with and without a system prompt, on all 940 FaithConflict
pairs. This is the script behind Figure 2 (`fig:gap_main`) and the underlying
rates in Appendix E.1 (Table `gap_full`).

Output: `faithgap_results.json` (a copy is at `../results/faithgap_results_main.json`).

Behavior judging uses the rubric in `../prompts/behavior_judge.py`.

Frontier models (Claude Sonnet, GPT-4o, DeepSeek-V3) are folded into the same
table via the scripts in `../formats/` (`frontier_multitask_inference.py`,
`run_frontier_opus_gpt5.py`).
