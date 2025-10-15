"""FastAPI entrypoint for the context compressor."""

from fastapi import FastAPI

from .compression import Compressor
from .config import settings
from .guards import ensure_code_blocks_closed, forbid_identifier_renames
from .models import CompressRequest, CompressResponse
from .selection import Selector

app = FastAPI(title="Context Compressor", version="0.1.0")
selector: Selector | None = None
compressor: Compressor | None = None


@app.on_event("startup")
def startup() -> None:
    global selector, compressor
    if selector is None:
        selector = Selector(settings.embedding_model)
    if compressor is None:
        compressor = Compressor()


@app.on_event("shutdown")
def shutdown() -> None:
    global selector, compressor
    if compressor is not None:
        compressor.close()
        compressor = None
    selector = None


@app.post("/compress", response_model=CompressResponse)
def compress(req: CompressRequest) -> CompressResponse:
    assert selector is not None
    assert compressor is not None
    texts = req.texts
    default_keep_ratio = 0.4 if req.mode == "task" else 0.5
    keep_ratio = req.keep_ratio if req.keep_ratio is not None else default_keep_ratio
    lam = req.mmr_lambda if req.mmr_lambda is not None else settings.mmr_lambda
    indices, scores = selector.select(
        texts=texts,
        task=req.task,
        keep_ratio=keep_ratio,
        lam=lam,
    )

    selected_texts = [texts[index] for index in indices]
    selected_content = "\n\n---\n\n".join(selected_texts)
    compressed_text = compressor.compress(
        content=selected_content,
        task=req.task,
        budget=req.budget_tokens if req.budget_tokens is not None else 800,
        mode=req.mode,
    )

    compressed_text = ensure_code_blocks_closed(compressed_text)
    try:
        forbid_identifier_renames(selected_content, compressed_text)
    except Exception:
        pass

    return CompressResponse(
        compressed=compressed_text,
        kept_indices=indices,
        kept_count=len(indices),
        original_count=len(texts),
        selection_scores=scores if req.return_selection else None,
        kept_texts=selected_texts if req.return_selection else None,
        meta={
            "backend": settings.compressor_backend,
            "model": settings.openai_model
            if settings.compressor_backend == "OPENAI"
            else settings.hf_model,
        },
    )


@app.get("/healthz")
def health() -> dict:
    return {"ok": True}
