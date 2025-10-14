"""Application configuration loaded from environment variables."""

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.5"))
    chunk_target_tokens: int = int(os.getenv("CHUNK_TARGET_TOKENS", "900"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "120"))

    compressor_backend: str = os.getenv("COMPRESSOR_BACKEND", "OPENAI").upper()

    # OpenAI-compatible backend
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:8001/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "local")
    openai_model: str = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instruct")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    openai_top_p: float = float(os.getenv("OPENAI_TOP_P", "0.9"))
    openai_max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", "800"))

    # HuggingFace backend
    hf_model: str = os.getenv("HF_MODEL", "google/flan-t5-large")
    hf_device: str = os.getenv("HF_DEVICE", "auto")
    hf_max_new_tokens: int = int(os.getenv("HF_MAX_NEW_TOKENS", "400"))


settings = Settings()
