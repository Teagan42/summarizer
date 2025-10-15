"""Utilities for chunking documents into token-constrained segments."""

from collections.abc import Iterable

try:  # pragma: no cover - optional dependency wiring
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime fallback
    tiktoken = None  # type: ignore

__all__ = ["chunk_text"]


def _window(sequence: list[int], size: int, step: int) -> Iterable[list[int]]:
    for start in range(0, len(sequence), step):
        yield sequence[start : start + size]


def _words_window(words: list[str], size: int, step: int) -> Iterable[list[str]]:
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if chunk:
            yield chunk


def _resolve_encoding():
    if tiktoken is None:  # pragma: no cover - optional dependency wiring
        return None

    resolvers = (
        lambda: tiktoken.get_encoding("cl100k_base"),
        lambda: tiktoken.encoding_for_model("gpt-4o-mini"),
    )
    for resolver in resolvers:  # pragma: no branch - tiny loop
        try:
            return resolver()
        except Exception:  # pragma: no cover - ignore and try next
            continue
    return None


def chunk_text(document: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Chunk ``document`` into segments that respect the provided budgets."""

    if not document:
        return []

    target = max(target_tokens, 1)
    overlap = max(overlap_tokens, 0)
    effective_overlap = min(overlap, target - 1) if target > 1 else 0
    step = max(target - effective_overlap, 1)

    encoding = _resolve_encoding()

    if encoding is not None:
        token_ids = encoding.encode(document)
        if not token_ids:
            return []

        chunks = [
            encoding.decode(window) for window in _window(token_ids, target, step)
        ]
        return [chunk for chunk in chunks if chunk]

    words = document.split()
    if not words:
        return [document]

    return [" ".join(chunk) for chunk in _words_window(words, target, step)]
