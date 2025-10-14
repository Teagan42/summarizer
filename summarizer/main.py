import re

from fastapi import FastAPI
from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str


def create_app() -> FastAPI:
    app = FastAPI(title="Summarizer API")

    @app.post("/summarize", response_model=SummarizeResponse)
    def summarize(payload: SummarizeRequest) -> SummarizeResponse:
        summary = summarize_text(payload.text)
        return SummarizeResponse(summary=summary)

    return app


TERMINAL_PUNCTUATION = ".!?"
CLOSING_PUNCTUATION = "\"'”’)]}>»›"
TERMINAL_PUNCTUATION_CLASS = re.escape(TERMINAL_PUNCTUATION)
CLOSING_PUNCTUATION_CLASS = re.escape(CLOSING_PUNCTUATION)

FIRST_SENTENCE_PATTERN = re.compile(
    rf"""
    ^\s*
    (?P<sentence>
        .*?
        [{TERMINAL_PUNCTUATION_CLASS}]
        (?:[{CLOSING_PUNCTUATION_CLASS}]+)?
    )
    (?:\s+|$)
    """,
    re.VERBOSE | re.DOTALL,
)


def summarize_text(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return ""

    match = FIRST_SENTENCE_PATTERN.match(stripped_text)
    if match:
        first_sentence = match.group("sentence").strip()
    else:
        first_sentence = stripped_text

    if first_sentence and not _has_terminal_punctuation(first_sentence):
        first_sentence += "."

    return first_sentence


def _has_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip(CLOSING_PUNCTUATION)
    return bool(stripped) and stripped[-1] in TERMINAL_PUNCTUATION
