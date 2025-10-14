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
