"""Pydantic models for the compression API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class CompressRequest(BaseModel):
    texts: list[str] = Field(..., description="List of text chunks to condense")
    task: str | None = Field(
        None, description="Task conditioning, e.g. 'assist coding on feature X'"
    )
    mode: Literal["losslessish", "task"] = "losslessish"
    budget_tokens: int | None = 800
    return_selection: bool = False


class CompressResponse(BaseModel):
    compressed: str
    kept_indices: list[int]
    kept_count: int
    original_count: int
    selection_scores: list[float] | None = None
    meta: dict[str, Any] = {}
