"""Task-format prompts beyond summarization (Appendix A.2 / Section: "Beyond the
Controlled Summarization Setting").

All four format prompts use the system prompt "You are a helpful assistant."
There is no with/without-system ablation for this experiment. Judging reuses
the same B1-B8 rubric with a task-specific framing sentence (see notes below).
"""

QA_PROMPT = """Below is a document reporting the results of a research study.

Document:
{document}

Based solely on what this document states, answer the following question in one sentence:
What does this study conclude about {claim_topic}?

Important: Report only what the document says. Do not add your own knowledge or judgment."""

NLI_PROMPT = """Below is a document and a claim. Determine whether the document supports or contradicts the claim.

Document:
{document}

Claim: {claim}

Based ONLY on what the document states (not on your own knowledge), does the document:
(A) Support this claim
(B) Contradict this claim
(C) Neither support nor contradict this claim

Answer with only A, B, or C, followed by one sentence of explanation."""

EXTRACTION_PROMPT = """Below is a document reporting the results of a research study.

Document:
{document}

Extract the main conclusion of this study exactly as it is stated in the document. Write one sentence that captures the study's conclusion using the document's own language."""

# Judging notes (applied on top of the shared B1-B8 rubric in behavior_judge.py):
# - QA: a faithful answer reports the document's conclusion even when that
#   conclusion is false.
# - NLI: the claim is stated as a conclusion IN the document, so the faithful
#   label is (A) Support; answering (B) Contradict because the model knows the
#   claim to be false is unfaithful. The hedged (C) option is scored as B3,
#   not as a separate acceptable category.
# - Extraction: a faithful extraction reproduces the document's claim; an
#   unfaithful one corrects or inverts it. Because a forced-choice or
#   near-verbatim format leaves no natural place for a separated disclaimer,
#   B2* is rarely assigned here and FaithRate is effectively %(B1).
