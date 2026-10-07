# Real-world validity: naturalistic Reddit corpus

Seven models on real health-misinformation Reddit posts (Appendix H). No confirming counterpart, so this reports absolute unfaithfulness rates, not FaithGap.

Run in order: `01_load_and_filter.py` → `02_generate_responses.py` → `03a_judge_b_labels.py` → `03b_judge_c_labels.py` → `04_aggregate_results.py`. All steps are resumable.

Reads `CONVAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN` from env.
