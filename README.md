# Emergent Unfaithfulness

**How Alignment Training Causes Language Models to Silently Override Task Faithfulness**

Pardis Sadat Zahraei · Janvijay Singh · Gokhan Tur · Dilek Hakkani-Tür
University of Illinois Urbana-Champaign, COLM 2026

[Project page](https://uiuc-conversational-ai-lab.github.io/Emergent-unfaithfulness/) · [Paper](https://drive.google.com/file/d/1cuVMav7BfVGGMTFwa6nz8DA7WNUMfL9s/view?usp=sharing) · [FaithConflict on HuggingFace](https://huggingface.co/datasets/PardisSzah/FaithConflict)

---

## TL;DR

Aligned language models often override or suppress content they disagree with instead of disclosing the override. We call this **alignment-induced unfaithfulness (AIU)**. We introduce **FaithConflict**, a controlled dataset isolating claim truth value, and show AIU is universal across 22 checkpoints from 8 model families, worsens with scale and safety alignment, grows sharply during DPO, and is amplified by chain-of-thought.

![Overview](assets/figures/figure1_overview.png)
![Results](assets/figures/figure2_results.png)

## Repository contents

```
.
├── index.html     # project page
├── assets/        # paper figures
├── data/          # FaithConflict dataset (940 pairs)
├── prompts/       # prompt templates and judge rubrics
├── eval/          # main 22-checkpoint FaithGap pipeline (Figure 2)
├── formats/       # QA/NLI/Extraction + frontier models (Appendix G)
├── training/      # stage analysis, preference analysis, FaithDPO mitigation
├── realworld/     # naturalistic Reddit validation (Appendix H)
├── mechinterp/    # representational / override-direction analysis
└── results/       # derived result JSON/JSONL for reanalysis
```

## FaithConflict dataset

940 paired instances (1,880 documents) across 10 categories, each pair sharing one template with a `[CLAIM]` slot filled with a confirming or opposing claim.

```python
from datasets import load_dataset
pairs = load_dataset("PardisSzah/FaithConflict", "pairs", split="train")
docs  = load_dataset("PardisSzah/FaithConflict", "documents", split="train")
```

Schema and category breakdown: see the [dataset card](https://huggingface.co/datasets/PardisSzah/FaithConflict).

## The metric

```
FaithGap = FaithRate(confirming) − FaithRate(opposing)
```

Zero means faithfulness doesn't depend on agreement with the source; positive is the signature of AIU.

## Citation

```bibtex
@inproceedings{zahraei2026emergent,
  title     = {Emergent Unfaithfulness: How Alignment Training Causes Language Models to Silently Override Task Faithfulness},
  author    = {Zahraei, Pardis Sadat and Singh, Janvijay and Tur, Gokhan and Hakkani-T{\"u}r, Dilek},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

Code: MIT. Dataset: CC-BY-4.0.
