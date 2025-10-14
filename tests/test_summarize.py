import pytest
from fastapi.testclient import TestClient

from summarizer.main import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_summarize_returns_first_sentence(client):
    payload = {
        "text": "This is a test. This sentence provides additional information."
    }

    response = client.post("/summarize", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body == {"summary": "This is a test."}
