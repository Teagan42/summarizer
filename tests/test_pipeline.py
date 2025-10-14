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
            for idx, _ in enumerate(texts):
                vec = np.zeros(self._dim)
                vec[idx % self._dim] = 1.0
                if normalize_embeddings:
                    vec = vec / np.linalg.norm(vec)
                embeddings.append(vec)
            return embeddings

    monkeypatch.setattr(selection, "SentenceTransformer", DummySentenceTransformer)
    return selection.Selector("dummy-model")


def test_selector_basic(selector):
    idxs, scores = selector.select(
        ["alpha", "beta", "gamma"], task="letters", keep_ratio=0.5, lam=0.5
    )

    assert 1 <= len(idxs) <= 3
    assert len(scores) == len(idxs)


def test_selector_preserves_chronology(selector, monkeypatch):
    texts = ["first", "second", "third"]

    def fake_mmr(_embeddings, _task_embedding, *, k, lam):  # noqa: ARG001
        assert k == 2
        assert lam == 0.5
        return [2, 0]

    monkeypatch.setattr(selector, "mmr", fake_mmr)

    indices, scores = selector.select(texts, task=None, keep_ratio=0.75, lam=0.5)

    assert indices == [0, 2]
    assert list(zip(indices, scores, strict=True)) == [
        (0, pytest.approx(0.5773502691896257)),
        (2, pytest.approx(0.5773502691896257)),
    ]

    from app.selection import join_texts

    assert join_texts(texts, indices) == "first\n\n---\n\nthird"


def test_selector_keeps_duplicate_indices(selector, monkeypatch):
    texts = ["alpha", "beta", "gamma"]

    def fake_mmr(_embeddings, _task_embedding, *, k, lam):  # noqa: ARG001
        assert k == 2
        assert lam == 0.5
        return [1, 1]

    monkeypatch.setattr(selector, "mmr", fake_mmr)

    indices, scores = selector.select(texts, task=None, keep_ratio=0.75, lam=0.5)

    assert indices == [1, 1]
    assert len(scores) == 2
    assert all(score == pytest.approx(scores[0]) for score in scores)


def test_selector_raises_on_score_length_mismatch(selector, monkeypatch):
    texts = ["first", "second", "third"]

    def fake_mmr(_embeddings, _task_embedding, *, k, lam):  # noqa: ARG001
        return [0, 1]

    monkeypatch.setattr(selector, "mmr", fake_mmr)

    from app import selection

    def fake_cosine_similarity(_matrix, _vector):
        return np.array([0.123])

    monkeypatch.setattr(selection, "_cosine_similarity", fake_cosine_similarity)

    with pytest.raises(ValueError):
        selector.select(texts, task=None, keep_ratio=0.75, lam=0.5)


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

    monkeypatch.setattr(
        compression.httpx,
        "Client",
        lambda *args, **kwargs: DummyClient(*args, **kwargs),
    )

    compressor = compression.Compressor()

    compressor.compress(content="data", task=None, budget=100, mode="losslessish")
    compressor.compress(
        content="data",
        task=None,
        budget=settings.openai_max_tokens + 500,
        mode="losslessish",
    )

    assert calls[0]["max_tokens"] == 100
    assert calls[1]["max_tokens"] == settings.openai_max_tokens
