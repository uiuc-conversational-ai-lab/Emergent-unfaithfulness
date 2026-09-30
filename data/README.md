# FaithConflict

`faithconflict.json` is the 940-pair dataset, byte-identical to the version
published on [HuggingFace](https://huggingface.co/datasets/PardisSzah/FaithConflict).
For the documented schema, category breakdown, a ready-to-load `documents`
view (1,880 materialized confirming and opposing documents), and usage examples,
see the [dataset card](https://huggingface.co/datasets/PardisSzah/FaithConflict).
We keep the full write-up there rather than duplicating it here, so there is a
single source of truth.

Quick local load:

```python
import json
data = json.load(open("faithconflict.json"))
assert len(data) == 940
confirming_doc = data[0]["summary_template"].replace("[CLAIM]", data[0]["claim_true"])
opposing_doc   = data[0]["summary_template"].replace("[CLAIM]", data[0]["claim_false"])
```
