# Training dynamics (Section 5 / Appendix F)

- **`stage_analysis/`**: traces AIU through SFT, DPO, and RLVR/Instruct
  checkpoints of Tulu-3 and OLMo-3 (Appendix F.1, Table `stages_table`).
  `experiment2_run.py`, `experiment2_all_steps.py`, and `experiment2_within_step.py`
  score every intermediate checkpoint's outputs with the reward model.
  `experiment1_dpo_isolation_v2.py` is the causal-attribution experiment that
  isolates the DPO *optimizer* from the *training-data content* by rerunning
  DPO on safety-only and general-only preference splits with matched
  hyperparameters (Section 5.3).

- **`preference_analysis/`**: where in the preference data unfaithfulness
  arises (Section 5.2 / Appendix F.2). `experiment_faithconflict_rm.py` scores
  the Tulu-3 reward model on matched faithful/unfaithful (prompt, response)
  pairs built directly from FaithConflict; `analyze_rm_failure.py` breaks the
  result down by category group and reads, by default, the pre-computed
  output already shipped in `../../results/rm_failure_analysis/`, so it runs
  out of the box with no GPU needed. `experiment2_rm_scoring.py` runs the same
  kind of scoring on the intermediate stage-analysis checkpoints instead.
  `experiment3_redesign.py` tests the training-data asymmetry hypothesis
  directly on the safety classifier cache (binomial test, odds ratio, Wilson
  confidence intervals).

- **`safety_intervention/build_safety_classifier_cache.py`**: the Qwen-3-32B
  binary safety-relevance classifier (Appendix A.5) used to build the
  safety-filtered and safety-only training splits for the data intervention in
  Appendix F.3.

- **`faithdpo_mitigation/`**: the training-level mitigation of Section 6 and
  Appendix I.1. Rebuilds DPO preference pairs so faithful summaries of
  opposing sources are *chosen* rather than rejected, then reruns only the
  DPO stage. Run in order:
  1. `generate_faithful_rm_data_v2.py`: generate 8,460 faithful and unfaithful
     summary variants per instance (GPT-4o-mini) across all 10 categories.
  2. `train_faithful_rm_v2.py`: LoRA-finetune a corrected reward model on
     those pairs.
  3. `build_dpo_pairs_v2.py`: score Tulu checkpoint outputs with the
     corrected RM and keep pairs with margin at least 0.3 (1,119 of 1,317).
  4. `train_faithful_dpo_v2.py`: DPO on the 1,119 RM-scored pairs.
  5. `eval_faithful_model.py`: generate on all 940 confirming and 940 opposing
     FaithConflict documents (direct prompt).
  6. `judge_faithful_dpo.py`: apply the B1-B8 rubric.

  We independently recomputed the result from `../results/faithdpo_v2_judged.jsonl`.
  FaithRate(confirming) = 97.4%, FaithRate(opposing) = 86.7%, FaithGap =
  **10.7pp**, down from 21.6pp for the original DPO, matching the paper
  exactly. See `../KNOWN_GAPS.md` §3 for a related dead end we excluded.
