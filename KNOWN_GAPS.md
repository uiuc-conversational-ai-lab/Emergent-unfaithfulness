# Known gaps and honest caveats

This release is a curated snapshot of a larger research codebase. In curating
it we found a few things worth stating plainly rather than quietly omitting.

## 1. No standalone dataset-construction script

Appendix B.3 describes seed claim pairs written manually by the authors,
expanded via DeepSeek-V3 prompting, then manually reviewed and filtered. This
was genuinely an interactive, manual process (chat-based generation + manual
JSON editing), not a single rerunnable pipeline script. We are releasing the
final, filtered `data/faithconflict.json` (independently verified: 940 pairs,
exact per-category counts matching Table B.1, 262.9-word average template,
`[CLAIM]` at exactly 3 positions per template) with the construction procedure
documented in prose (README + paper Appendix B.3), rather than implying a
build script exists when it does not.

## 2. No standalone significance-testing script

We could not locate a single script that reproduces Table (McNemar
significance / Wilson intervals, Appendix E.3) from the raw judged outputs.
`eval/experiment4_run.py` and `formats/experiment4_multitask.py` produce the
raw per-model, per-condition faithfulness rates (`results/faithgap_results_*.json`);
computing McNemar's test and Wilson intervals from the underlying paired
per-item judged labels is a short, standard statistical step (e.g.
`statsmodels.stats.contingency_tables.mcnemar` and
`statsmodels.stats.proportion.proportion_confint`) that we were not able to
recover as a preserved script. If you need this and don't already have it,
it's a small addition on top of the per-item judged `.jsonl` files.

## 3. Superseded FaithDPO v2 multi-task extension excluded

The `training/` chain here (`generate_faithful_rm_data_v2.py`
→ `train_faithful_rm_v2.py` → `build_dpo_pairs_v2.py` → `train_faithful_dpo_v2.py`
→ `eval_faithful_model.py` → `judge_faithful_dpo.py`) is the **verified** chain:
we independently recomputed FaithRate(confirming) = 97.4%, FaithRate(opposing)
= 86.7%, FaithGap = 10.7pp directly from `results/faithdpo_v2_judged.jsonl`,
matching Section 6 / Appendix I.1 of the paper exactly.

A later, separate attempt to extend this same model to QA/NLI/Extraction
formats (scripts `eval_faithdpo_v2_all_tasks.py`, `eval_extraction_plus_only.py`,
`judge_faithdpo_v2.py`, `complete_faithdpo_v2_evaluation.py`, output directory
`faithdpo_v2_all_tasks/`) hit a generation bug that truncated every output to
~100 characters mid-word (`generated[:100]`), producing judged labels
dominated by spurious B8 in both conditions. That multi-task extension was
never completed or cited in the paper. We excluded it from this release,
including `eval_extraction_plus_only.py`, which had been mistakenly carried
into `formats/` in an earlier pass and has since been removed.

## 4. Paths and credentials are environment-variable driven

Every script that reads or writes files outside the repo does so through a
small set of environment variables, each with a sensible default:

- `FAITHCONFLICT_DATA_DIR` (default: `data/` inside the repo): where
  `faithconflict.json` lives.
- `FAITHCONFLICT_WORK_DIR` (default: `workdir/` inside the repo): a writable
  location for the HF cache, intermediate pairs, model checkpoints, and run
  outputs. Point this at real storage before running anything at scale.
- `FAITHCONFLICT_CHECKPOINTS_DIR` (default: `workdir/checkpoints`): the
  intermediate SFT/DPO/Instruct checkpoint outputs consumed by the
  stage-analysis and FaithDPO scripts.
- `HF_TOKEN`, `OPENROUTER_API_KEY`, `CONVAI_API_KEY`: read directly from the
  environment for gated models and hosted inference. None are hardcoded
  anywhere in this repo; set the ones a given script needs before running it.

None of these need to be set to read or lint the code. They only matter if
you are actually running a script that touches models, checkpoints, or an
API.

## 5. Large artifacts not included

Model checkpoints (FaithDPO / reward models / DPO ablation checkpoints) are
released separately on the HuggingFace Hub under the `PardisSzah` account, not
duplicated here. Raw run logs and full intermediate activation caches from the
mechanistic-interpretability analysis (`mechinterp/`) are also excluded; the
small derived artifacts needed to reproduce the reported numbers
(`mechinterp/results/*.json`) are included.
