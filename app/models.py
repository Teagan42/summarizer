"""Pydantic models for the compression API."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CompressRequest(BaseModel):
    texts: list[str] = Field(..., description="List of text chunks to condense")
    task: str | None = Field(
        None, description="Task conditioning, e.g. 'assist coding on feature X'"
    )
    mode: Literal["losslessish", "task"] = "losslessish"
    budget_tokens: int | None = 800
    return_selection: bool = False

    @field_validator("texts", mode="before")
    @classmethod
    def _normalize_texts(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        if value is None:
            raise ValueError("texts must be provided")

        return [str(value)]


class CompressResponse(BaseModel):
    compressed: str
    kept_indices: list[int]
    kept_count: int
    original_count: int
    selection_scores: list[float] | None = None
    kept_texts: list[str] | None = None
    meta: dict[str, Any] = {}
