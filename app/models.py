"""Pydantic models for the compression API."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CompressRequest(BaseModel):
    texts: list[str] | None = Field(
        default=None, description="List of text chunks to condense"
    )
    document: str | None = Field(
        default=None,
        description=(
            "Raw document to segment before selection; overrides `texts` when provided"
        ),
    )
    task: str | None = Field(
        None, description="Task conditioning, e.g. 'assist coding on feature X'"
    )
    mode: Literal["losslessish", "task"] = "losslessish"
    budget_tokens: int | None = Field(default=800, ge=0)
    return_selection: bool = False
    keep_ratio: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Fraction of texts to retain during selection",
    )
    mmr_lambda: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="MMR trade-off between relevance and diversity",
    )

    @field_validator("texts", mode="before")
    @classmethod
    def _normalize_texts(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]

        return [str(value)]

    @model_validator(mode="after")
    @classmethod
    def _ensure_payload(cls, values: "CompressRequest") -> "CompressRequest":
        if values.texts is None and values.document is None:
            raise ValueError("either texts or document must be provided")
        return values


class CompressResponse(BaseModel):
    compressed: str
    kept_indices: list[int]
    kept_count: int
    original_count: int
    selection_scores: list[float] | None = None
    kept_texts: list[str] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
