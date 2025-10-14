"""Prompt templates for the compressor."""

LOSSLESSISH_PROMPT = """You are a context compressor. Output MUST be ≤ {budget} tokens.
Preserve: facts, numbers, units, dates, names, identifiers, API signatures, file paths.
Allowed to remove: filler, repetition, meta-chatter.

Rules:
- Maintain chronology and speaker tags if present.
- Keep code/JSON valid and fenced.
- Do not invent content. Use [uncertain: ...] if unsure.

Input:
<<<
{content}
>>>

Output (markdown):
- Key facts (bullets)
- Decisions/assumptions
- Open questions
- Code/Schema (as fenced blocks if present)
"""


TASK_PROMPT = """Task: {task}
Audience: a model that will act on this result.

Select and compress ONLY information needed to complete the task.
Prefer interfaces, constraints, examples, error messages, TODOs.
Drop pleasantries, unrelated history, and tangents.
Keep identifiers/imports/types exactly. Code/JSON must be valid and fenced.
Output ≤ {budget} tokens.

Input:
<<<
{content}
>>>

Output (sections):
- Constraints
- Interfaces/contracts
- Steps done / pending
- Pitfalls/edge cases
"""
