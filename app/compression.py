"""Abstractive compression backends."""

from __future__ import annotations

from typing import List

import httpx

from .config import settings
from .prompts import LOSSLESSISH_PROMPT, TASK_PROMPT

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except Exception:  # pragma: no cover - transformers is optional
    AutoTokenizer = AutoModelForSeq2SeqLM = pipeline = None  # type: ignore


class Compressor:
    """Delegate compression to an OpenAI-compatible or HF backend."""

    def __init__(self) -> None:
        self.backend = settings.compressor_backend
        self.client: httpx.Client | None = None
        self.pipe = None

        if self.backend == "OPENAI":
            self.client = httpx.Client(
                base_url=settings.openai_base_url,
                timeout=120.0,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
        elif self.backend == "HF":
            if pipeline is None:
                raise RuntimeError("transformers is required for HF backend")
            self.pipe = pipeline(
                "text2text-generation",
                model=settings.hf_model,
                device_map=settings.hf_device,
            )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @staticmethod
    def _clamp_budget(requested: int, maximum: int) -> int:
        """Ensure per-request budgets stay within backend limits."""

        return max(1, min(requested, maximum))

    def _prompt(self, content: str, task: str | None, budget: int, mode: str) -> str:
        if mode == "task" and task:
            return TASK_PROMPT.format(task=task, budget=budget, content=content)
        return LOSSLESSISH_PROMPT.format(budget=budget, content=content)

    def compress(self, content: str, task: str | None, budget: int, mode: str) -> str:
        prompt = self._prompt(content, task, budget, mode)
        if self.backend == "OPENAI":
            assert self.client is not None
            max_tokens = self._clamp_budget(budget, settings.openai_max_tokens)
            body = {
                "model": settings.openai_model,
                "temperature": settings.openai_temperature,
                "top_p": settings.openai_top_p,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a deterministic context compressor.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            response = self.client.post("/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        assert self.pipe is not None
        max_new_tokens = self._clamp_budget(budget, settings.hf_max_new_tokens)
        output: List[dict] = self.pipe(
            prompt, max_new_tokens=max_new_tokens, do_sample=False
        )
        return output[0]["generated_text"].strip()
