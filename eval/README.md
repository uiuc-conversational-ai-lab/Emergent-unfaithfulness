# Main evaluation pipeline

`experiment4_run.py` generates, judges, and scores all 22 model checkpoints under Direct, CoT, and Mitigated prompts on all 940 FaithConflict pairs. Produces Figure 2.

Output: `../results/faithgap_results_main.json`. Judging rubric: `../prompts/behavior_judge.py`.
