import pytest
from fastapi.testclient import TestClient


class DummySelector:
    def __init__(self) -> None:
        self.keep_ratio_observed: float | None = None
        self.lambda_observed: float | None = None
        self.selection_kwargs: dict[str, float] | None = None

    def select(self, texts, task, keep_ratio, lam):
        self.keep_ratio_observed = keep_ratio
        self.lambda_observed = lam
        self.selection_kwargs = {"keep_ratio": keep_ratio, "lam": lam}
        indices = list(range(min(len(texts), 2)))
        scores = [0.9, 0.8][: len(indices)]
        return indices, scores


class DummyCompressor:
    def __init__(self) -> None:
        self.closed = False
        self.budget_observed: int | None = None

    def compress(self, content, task, budget, mode):
        self.budget_observed = budget
        return "compressed result"

    def close(self) -> None:
        self.closed = True


def test_dummy_compressor_exposes_close():
    dummy = DummyCompressor()

    dummy.close()

    assert getattr(dummy, "closed", True) is True


@pytest.fixture
def client(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "selector", DummySelector())
    monkeypatch.setattr(main, "compressor", DummyCompressor())

    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def client_and_compressor(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "selector", DummySelector())
    compressor = DummyCompressor()
    monkeypatch.setattr(main, "compressor", compressor)

    with TestClient(main.app) as client:
        yield client, compressor


@pytest.fixture
def client_with_selector(monkeypatch):
    from app import main

    selector = DummySelector()
    monkeypatch.setattr(main, "selector", selector)
    monkeypatch.setattr(main, "compressor", DummyCompressor())

    with TestClient(main.app) as client:
        yield client, selector


@pytest.fixture
def metrics_recorder(monkeypatch):
    from app import main

    events: list[dict[str, object]] = []

    def record(event: dict[str, object]) -> None:
        events.append(event)

    main.metrics.set_sink(record)

    try:
        yield events
    finally:
        main.metrics.reset_sink()


@pytest.fixture
def metrics_error_sink(monkeypatch):
    from app import main

    def raise_error(_: dict[str, object]) -> None:
        raise RuntimeError("sink failure")

    main.metrics.set_sink(raise_error)

    try:
        yield
    finally:
        main.metrics.reset_sink()


def test_health(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_compress_endpoint(client):
    payload = {
        "texts": ["Function A does X", "Function B depends on A", "Random chit-chat"],
        "task": "summarize dependencies for refactor",
        "mode": "task",
        "budget_tokens": 200,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["compressed"] == "compressed result"
    assert body["kept_indices"] == [0, 1]
    assert body["kept_count"] == 2
    assert body["original_count"] == 3
    assert body["kept_texts"] is None


def test_compress_endpoint_accepts_single_string(client):
    payload = {
        "texts": "single block",
        "task": "summarize dependencies for refactor",
        "mode": "task",
        "budget_tokens": 200,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["compressed"] == "compressed result"
    assert body["kept_indices"] == [0]
    assert body["kept_count"] == 1
    assert body["original_count"] == 1


def test_compress_endpoint_emits_metrics_events(client, metrics_recorder):
    payload = {
        "texts": ["Function A does X", "Function B depends on A", "Random chit-chat"],
        "task": "summarize dependencies for refactor",
        "mode": "task",
        "budget_tokens": 200,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    stages = [event["stage"] for event in metrics_recorder]
    assert stages == ["select", "compress"]
    for event in metrics_recorder:
        assert event["mode"] == "task"
        assert event["keep_ratio"] == pytest.approx(0.4)
        assert event["backend"] == "OPENAI"
        assert event["model"]
        assert event["duration_ms"] >= 0


def test_compress_endpoint_propagates_metrics_errors(client, metrics_error_sink):
    payload = {
        "texts": ["Function A does X", "Function B depends on A"],
        "mode": "task",
    }

    with pytest.raises(RuntimeError, match="sink failure"):
        client.post("/compress", json=payload)


def test_shutdown_closes_compressor(monkeypatch):
    from app import main

    class TrackingCompressor(DummyCompressor):
        def __init__(self) -> None:
            super().__init__()

    tracker = TrackingCompressor()

    monkeypatch.setattr(main, "selector", DummySelector())
    monkeypatch.setattr(main, "compressor", tracker)

    with TestClient(main.app) as client:
        client.get("/healthz")

    assert tracker.closed is True


def test_compress_endpoint_returns_kept_texts_when_requested(client):
    payload = {
        "texts": [
            "Function A does X",
            "Function B depends on A",
            "Random chit-chat",
        ],
        "task": "summarize dependencies for refactor",
        "mode": "task",
        "budget_tokens": 200,
        "return_selection": True,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["kept_texts"] == [
        "Function A does X",
        "Function B depends on A",
    ]


def test_compress_endpoint_honors_zero_budget(client_and_compressor):
    client, compressor = client_and_compressor
    payload = {
        "texts": [
            "Function A does X",
            "Function B depends on A",
        ],
        "mode": "losslessish",
        "budget_tokens": 0,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    assert compressor.budget_observed == 0


def test_compress_endpoint_rejects_keep_ratio_out_of_range(client):
    payload = {
        "texts": ["Function A does X", "Function B depends on A"],
        "mode": "losslessish",
        "keep_ratio": 1.5,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 422


def test_compress_endpoint_rejects_mmr_lambda_out_of_range(client):
    payload = {
        "texts": ["Function A does X", "Function B depends on A"],
        "mode": "losslessish",
        "mmr_lambda": -0.1,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 422


def test_chunker_splits_long_input(monkeypatch):
    from app import main
    from app.config import settings

    class TrackingSelector(DummySelector):
        def __init__(self) -> None:
            super().__init__()
            self.texts_observed: list[str] | None = None

        def select(self, texts, task, keep_ratio, lam):
            self.texts_observed = list(texts)
            return super().select(texts, task, keep_ratio, lam)

    monkeypatch.setattr(settings, "chunk_target_tokens", 5)
    monkeypatch.setattr(settings, "chunk_overlap_tokens", 2)

    selector = TrackingSelector()
    monkeypatch.setattr(main, "selector", selector)
    monkeypatch.setattr(main, "compressor", DummyCompressor())

    document = " ".join(f"word{i}" for i in range(1, 13))

    with TestClient(main.app) as client:
        response = client.post(
            "/compress",
            json={
                "document": document,
                "mode": "task",
            },
        )

    assert response.status_code == 200
    assert selector.texts_observed is not None
    assert selector.texts_observed == [
        "word1 word2 word3 word4 word5",
        "word4 word5 word6 word7 word8",
        "word7 word8 word9 word10 word11",
        "word10 word11 word12",
    ]


def test_compress_endpoint_passes_selection_hyperparams(client_with_selector):
    client, selector = client_with_selector
    keep_ratio = 0.42
    mmr_lambda = 0.17
    payload = {
        "texts": [
            "Function A does X",
            "Function B depends on A",
            "Random chit-chat",
        ],
        "task": "summarize dependencies for refactor",
        "mode": "task",
        "budget_tokens": 200,
        "keep_ratio": keep_ratio,
        "mmr_lambda": mmr_lambda,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    assert selector.keep_ratio_observed == pytest.approx(keep_ratio)
    assert selector.lambda_observed == pytest.approx(mmr_lambda)


def test_compress_endpoint_records_explicit_selection_hyperparams(client_with_selector):
    client, selector = client_with_selector
    keep_ratio = 0.12
    mmr_lambda = 0.34
    payload = {
        "texts": [
            "Function A does X",
            "Function B depends on A",
            "Random chit-chat",
        ],
        "task": "summarize dependencies for refactor",
        "mode": "task",
        "budget_tokens": 200,
        "keep_ratio": keep_ratio,
        "mmr_lambda": mmr_lambda,
    }

    response = client.post("/compress", json=payload)

    assert response.status_code == 200
    assert selector.selection_kwargs is not None
    assert selector.selection_kwargs["keep_ratio"] == pytest.approx(keep_ratio)
    assert selector.selection_kwargs["lam"] == pytest.approx(mmr_lambda)
