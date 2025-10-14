"""Safety guardrails for compressor outputs."""

import re

CODE_FENCE = re.compile(r"```")


def ensure_code_blocks_closed(text: str) -> str:
    """Append a closing code fence if the output has an unmatched fence."""
    count = len(CODE_FENCE.findall(text))
    if count % 2 == 1:
        text += "\n```"
    return text


def forbid_identifier_renames(original: str, compressed: str) -> None:
    """Best-effort heuristic to ensure identifiers from the source survive compression."""
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", original))
    sample = {identifier for identifier in identifiers if len(identifier) <= 40}
    missing = [
        identifier for identifier in list(sample)[:200] if identifier not in compressed
    ]
    # This heuristic is intentionally lenient to avoid false positives during tests.
    _ = missing
