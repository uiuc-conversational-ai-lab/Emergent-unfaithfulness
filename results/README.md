# Result artifacts

Small, derived result files kept for reference and reanalysis, all produced
by the scripts elsewhere in this repo:

- `faithgap_results_main.json`: main 22-checkpoint eval (`../eval/experiment4_run.py`), underlies Figure 2 and Table `gap_full`.
- `faithgap_results_frontier.json`: frontier models' task-format results (`../formats/`).
- `faithgap_results_formats.json`: open-weight models' QA/NLI/Extraction/Extraction+ results (`../formats/`).
- `all_steps_summary.json`, `group_summary.json`, `within_step_summary.json`: stage-by-stage post-training analysis (`../training/stage_analysis/`).
- `faithdpo_v2_judged.jsonl`: per-item judged labels for the FaithDPO v2 mitigation (`../training/faithdpo_mitigation/`). Aggregates to FaithGap = 10.7pp, as reported in Section 6.
- `rm_failure_analysis/`: reward-model scoring on matched faithful and unfaithful pairs, produced by `../training/preference_analysis/experiment_faithconflict_rm.py`. `analyze_rm_failure.py` reads this directory by default, so the preference-analysis result can be reproduced without a GPU.
