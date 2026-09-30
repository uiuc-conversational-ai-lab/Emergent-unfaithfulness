"""Safety-relevance classifier prompt, verbatim from the paper (Appendix A.5).

Used to label each instance in the Tulu-3 training corpus as safety-relevant
or not (Qwen-3-32B, temperature=0.7, top_p=0.8, top_k=20, min_p=0), to build
the safety-filtered / safety-only training splits for the data intervention
in Appendix F.3. Only examples labeled safety-irrelevant with high confidence
are retained for the safety-filtered split; the split is then uniformly
oversampled with replacement back to the original size, so training
hyperparameters (LR, schedule, steps) stay unchanged and comparable to
original Tulu-3.
"""

SAFETY_CLASSIFIER_PROMPT = """You are a binary classifier for detecting broad safety/alignment behavior in assistant responses.

Task:
Given a conversation, determine whether the assistant response contains **broad safety/alignment behavior**, including moderation, harm-avoidance, policy-style caution, or safety-motivated redirection.

Definition:

Label = 1 (HAS safety/alignment behavior) if the assistant response includes ANY clear sign of:
- Refusal or partial refusal
- Warnings, disclaimers, or risk statements
- Moral, ethical, legal, or policy-based reasoning
- Content moderation or policy enforcement
- Redirecting to safer alternatives due to harm/sensitivity concerns
- Attempts to avoid enabling harm, misuse, or sensitive wrongdoing
- Sanitizing, softening, or carefully reframing content in a safety-oriented way
- De-escalation or harm-minimizing framing beyond normal task completion

Label = 0 (NO safety/alignment behavior) if the response is:
- Purely task-focused
- Merely informative, descriptive, or factual
- Polite or cautious in tone without clear safety intent
- Correcting facts or disagreeing without moderation intent
- Offering generic alternatives for helpfulness rather than safety

Important rules:
- Only evaluate the assistant response, not the user message
- Do NOT label based on topic alone
- Harmful, political, sexual, or otherwise sensitive topics do NOT automatically imply Label 1
- Label 1 when the assistant's response itself shows clear moderation, harm-avoidance, or policy-style behavior
- When uncertain, prefer the label that best reflects the overall response behavior, not isolated phrases

Output format (strict JSON):
{{
  "safety_label_reason": "2-3 short sentences explaining the decision",
  "safety_label_confidence": "low" or "medium" or "high",
  "safety_label": 0 or 1
}}

Conversation:
{conversation_here}"""

SAMPLING_PARAMS = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)
