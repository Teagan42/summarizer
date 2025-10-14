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


SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def summarize_text(text: str) -> str:
    stripped_text = text.strip()
    if not stripped_text:
        return ""

    sentences = SENTENCE_BOUNDARY_PATTERN.split(stripped_text)
    first_sentence = sentences[0].strip()

    if first_sentence and first_sentence[-1] not in ".!?":
        first_sentence += "."

    return first_sentence
