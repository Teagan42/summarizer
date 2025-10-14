import pytest
from fastapi.testclient import TestClient


class DummySelector:
    def select(self, texts, task, keep_ratio, lam):
        return [0, 1], [0.9, 0.8]


class DummyCompressor:
    def compress(self, content, task, budget, mode):
        return "compressed result"


@pytest.fixture
def client(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "selector", DummySelector())
    monkeypatch.setattr(main, "compressor", DummyCompressor())

    return TestClient(main.app)


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
