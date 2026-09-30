# Representational analysis (Section 6 / Appendix I.2)

In Gemma-3-27B-IT, the activation difference between opposing and confirming
versions of the same document is near one-dimensional at layer 30.

- `nla_faithconflict.py`: extracts paired activations on confirming and
  opposing FaithConflict documents and computes the override direction (SVD of
  the paired activation difference).
- `nla_causal_intervention.py`: ablates and steers along that direction.
  Ablating it restores 11 of 12 unfaithful cases in a 30-instance safety
  subset. Steering raises faithfulness from 76.0% to 93.0%.

`results/` contains the derived, small artifacts referenced above:
`causal_ablate_L30.json` and `causal_ablate_L30_judged.json` (ablation, scored
two ways), `causal_steer_L30.json` (steering sweep), and
`cosine_similarity_comprehensive.json` (the keyword-vs-LLM-judge agreement
check on this direction). Raw multi-megabyte activation caches used to derive
these are not included; regenerate with the scripts above if needed.

We call this an *override* direction rather than a *faithfulness* direction,
since its vocabulary projection separates source attribution from adversative
register (see the paper's discussion in Section 6).
