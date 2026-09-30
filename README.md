# Emergent Unfaithfulness

**How Alignment Training Causes Language Models to Silently Override Task Faithfulness**

Pardis Sadat Zahraei · Janvijay Singh · Gokhan Tur · Dilek Hakkani-Tür
University of Illinois Urbana-Champaign, COLM 2026

[Project page](https://uiuc-conversational-ai-lab.github.io/Emergent-unfaithfulness/) · [Paper](https://drive.google.com/file/d/1cuVMav7BfVGGMTFwa6nz8DA7WNUMfL9s/view?usp=sharing) · [FaithConflict on HuggingFace](https://huggingface.co/datasets/PardisSzah/FaithConflict)

---

## TL;DR

Aligned language models are trained to intervene on unsafe or sensitive content, and they often do so **without disclosing it**. We call this **alignment-induced unfaithfulness (AIU)**: not hallucination (fabricating absent content), but the silent suppression or inversion of content that *is present* in the input. We introduce **FaithConflict**, a controlled dataset that isolates the effect of claim truth value alone, and find that AIU is universal across 22 model checkpoints from 8 families, gets **worse as models get larger and more safety-aligned** (a reverse scaling law), grows most sharply during **DPO**, is **amplified by chain-of-thought prompting**, and is **not resolved by prompting-based mitigation**: evidence of a capability–alignment–faithfulness trilemma in current LLM design.

![Overview of alignment-induced unfaithfulness, chain-of-thought amplification, and the reverse scaling law](assets/figures/figure1_overview.png)

![FaithGap across 22 model checkpoints, direct and chain-of-thought prompts](assets/figures/figure2_results.png)

## Repository contents

```
.
├── index.html                  # project page (served at the GitHub Pages link above)
├── assets/figures/              # Figure 1 (overview) and Figure 2 (main results) from the paper
├── data/                       # FaithConflict (940 pairs), also on HuggingFace, see below
├── prompts/                    # every prompt template and judge rubric, verbatim from the paper
│   ├── task_prompts.py         #   Direct / CoT / Mitigation summarization prompts (Appendix A.1)
│   ├── format_prompts.py       #   QA / NLI / Extraction prompts (Appendix A.2)
│   ├── behavior_judge.py       #   B1–B8 output-behavior judge rubric (Appendix A.3)
│   ├── reasoning_judge.py      #   C0–C6 reasoning-mode judge rubric (Appendix A.4)
│   └── safety_classifier.py    #   safety-relevance classifier (Appendix A.5)
├── eval/                       # main 22-checkpoint FaithGap pipeline (Figure 2, Appendix E)
├── formats/                    # QA/NLI/Extraction(+) + frontier models (Appendix G)
├── training/                   # stage analysis, preference analysis, safety intervention, FaithDPO mitigation (Appendix F, I.1)
├── realworld/                  # naturalistic Reddit health-misinfo validation (Appendix H)
├── mechinterp/                 # representational / override-direction analysis (Appendix I.2)
├── results/                    # small derived result JSON/JSONL, for reanalysis without rerunning
└── KNOWN_GAPS.md               # honest caveats: what we couldn't fully verify or recover, and why
```

Every subfolder has its own short README pointing to the exact paper section/table/figure it reproduces. Read `KNOWN_GAPS.md` before assuming every number is push-button reproducible: most are, but a few needed extra digging or have documented caveats.

## FaithConflict dataset

940 paired instances (1,880 documents) across 10 categories spanning three axes of the capability–alignment–faithfulness trilemma: **Safety** (Health & Safety Misinfo, Scientific Misinformation, Social Bias, Direct Social Bias, 386 pairs), **Capability** (Factual Counterfactuals, Math Reasoning, Hard Math Reasoning, 248 pairs), and **Subjective control** (Historical & Moral, Political/Ideological, Scientific Frontier, 306 pairs). Every pair shares one template with a `[CLAIM]` slot substituted with either a *confirming* (model-agreeing) or *opposing* (model-conflicting) claim, so any behavioral difference is attributable to truth value alone.

```python
from datasets import load_dataset
pairs = load_dataset("PardisSzah/FaithConflict", "pairs", split="train")       # 940 raw pairs
docs  = load_dataset("PardisSzah/FaithConflict", "documents", split="train")   # 1,880 materialized documents
```

Full schema, category table, and construction procedure are documented on the [dataset card](https://huggingface.co/datasets/PardisSzah/FaithConflict). The identical file is included here as `data/faithconflict.json` for provenance and offline use.

## The metric: FaithGap

```
FaithGap = FaithRate(confirming) − FaithRate(opposing)
FaithRate = P(behavior label ∈ {B1 (faithful), B2* (transparent disclaimer)})
```

Zero means faithfulness does not depend on agreement with the source; positive is the signature of AIU.

## The dual taxonomy

| | What it classifies | Acceptable | Failures (increasing severity) |
|---|---|---|---|
| **B1–B8** | model **output** | B1 (faithful), B2\* (transparent disclaimer) | B3 hedging → B4 editorial labeling → B5 appended correction → B6 interpretive reframing → B7 refusal → **B8 silent inversion** |
| **C0–C6** | model **reasoning** under CoT | C0 (faithful reasoning) | C1 transparent override → C2 silent override → **C3 rationalized override** → C4 reframing cascade → C5 capability override → **C6 compliant reasoning, defecting output** |

## Reproducing the paper

1. **Generate**: run each model on all 940 confirming and 940 opposing documents under Direct, CoT, and Mitigated prompts, with and without a system prompt (`eval/experiment4_run.py`). Decoding is greedy throughout: one generation per cell, no resampling.
2. **Judge**: annotate every output with the B1–B8 rubric, and every CoT trace with the C0–C6 rubric, using an LLM judge (paper uses Qwen-2.5 32B Instruct) (`prompts/behavior_judge.py`, `prompts/reasoning_judge.py`).
3. **Score**: FaithRate and FaithGap per model, condition, and category come straight out of step 1's output (`results/faithgap_results_*.json`). McNemar significance and Wilson intervals (Appendix E.3) are a short standard step on top of the paired per-item labels; see `KNOWN_GAPS.md` §2.
4. **Beyond summarization** (optional): QA/NLI/Extraction(+) and frontier models (`formats/`), naturalistic Reddit validation (`realworld/`).
5. **Training dynamics** (optional): trace AIU through SFT → DPO → RLVR checkpoints, run the safety-relevance data intervention, reproduce the FaithDPO mitigation, or the representational analysis (`training/`, `mechinterp/`).

## Citation

```bibtex
@inproceedings{zahraei2026emergent,
  title     = {Emergent Unfaithfulness: How Alignment Training Causes Language Models to Silently Override Task Faithfulness},
  author    = {Zahraei, Pardis Sadat and Singh, Janvijay and Tur, Gokhan and Hakkani-T{\"u}r, Dilek},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## Ethics

Stimulus documents in FaithConflict assert false or harmful claims solely for controlled experimental purposes; none reflect the views of the authors, and none are intended for dissemination beyond research use. The dataset was built to expose and study a failure mode in aligned models, not to produce or spread misinformation.

## License

Code: MIT. Dataset: CC-BY-4.0 (see the [dataset card](https://huggingface.co/datasets/PardisSzah/FaithConflict)).
