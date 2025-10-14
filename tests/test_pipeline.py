import numpy as np
import pytest


@pytest.fixture
def selector(monkeypatch):
    from app import selection

    class DummySentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            self._dim = 4

        def encode(self, texts, normalize_embeddings=True):
            embeddings = []
            for idx, text in enumerate(texts):
                vec = np.zeros(self._dim)
                vec[idx % self._dim] = 1.0
                if normalize_embeddings:
                    vec = vec / np.linalg.norm(vec)
                embeddings.append(vec)
            return embeddings

    monkeypatch.setattr(selection, "SentenceTransformer", DummySentenceTransformer)
    return selection.Selector("dummy-model")


def test_selector_basic(selector):
    idxs, scores = selector.select(["alpha", "beta", "gamma"], task="letters", keep_ratio=0.5, lam=0.5)

    assert 1 <= len(idxs) <= 3
    assert len(scores) == len(idxs)


def test_prompt_budget():
    from app.prompts import LOSSLESSISH_PROMPT

    prompt = LOSSLESSISH_PROMPT.format(budget=300, content="x")

    assert "≤ 300 tokens" in prompt or "<= 300 tokens" in prompt


def test_compressor_respects_budget(monkeypatch):
    from app import compression
    from app.config import settings

    calls = []

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def post(self, path, json):
            calls.append(json)
            return DummyResponse(json)

    monkeypatch.setattr(compression.httpx, "Client", lambda *args, **kwargs: DummyClient(*args, **kwargs))

    compressor = compression.Compressor()

    compressor.compress(content="data", task=None, budget=100, mode="losslessish")
    compressor.compress(content="data", task=None, budget=settings.openai_max_tokens + 500, mode="losslessish")

    assert calls[0]["max_tokens"] == 100
    assert calls[1]["max_tokens"] == settings.openai_max_tokens
