# Representational analysis

In Gemma-3-27B-IT, the confirming/opposing activation difference is near one-dimensional at layer 30.

- `nla_faithconflict.py`: extracts the override direction (SVD of paired activation difference).
- `nla_causal_intervention.py`: ablates/steers along it.

`results/` holds the small derived artifacts these scripts produce.
