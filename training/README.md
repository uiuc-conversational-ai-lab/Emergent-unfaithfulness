# Training dynamics

Scripts read/write through `FAITHCONFLICT_DATA_DIR`, `FAITHCONFLICT_WORK_DIR`, `FAITHCONFLICT_CHECKPOINTS_DIR` env vars.

**Stage analysis**: `experiment2_run.py`, `experiment2_all_steps.py`, `experiment2_within_step.py` trace AIU through SFT/DPO/RLVR checkpoints. `experiment1_dpo_isolation_v2.py` isolates the DPO optimizer from training-data content.

**Preference analysis**: `experiment_faithconflict_rm.py` + `analyze_rm_failure.py` score the reward model on matched faithful/unfaithful pairs. `experiment3_redesign.py` tests the training-data asymmetry hypothesis.

**Safety intervention**: `build_safety_classifier_cache.py` builds the safety-relevance classifier cache.

**FaithDPO mitigation**: `generate_faithful_rm_data_v2.py` → `train_faithful_rm_v2.py` → `build_dpo_pairs_v2.py` → `train_faithful_dpo_v2.py` → `eval_faithful_model.py` → `judge_faithful_dpo.py`. Reproduces FaithGap = 10.7pp (down from 21.6pp).
