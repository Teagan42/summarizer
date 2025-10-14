"""Pydantic models for the compression API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any


class CompressRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text chunks to condense")
    task: Optional[str] = Field(None, description="Task conditioning, e.g. 'assist coding on feature X'")
    mode: Literal["losslessish", "task"] = "losslessish"
    budget_tokens: Optional[int] = 800
    return_selection: bool = False


class CompressResponse(BaseModel):
    compressed: str
    kept_indices: List[int]
    kept_count: int
    original_count: int
    selection_scores: Optional[List[float]] = None
    meta: Dict[str, Any] = {}
