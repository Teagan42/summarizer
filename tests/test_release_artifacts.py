from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/release.yml"


@pytest.fixture(scope="module")
def dockerfile_content() -> str:
    return DOCKERFILE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def release_workflow_content() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_dockerfile_exists():
    assert DOCKERFILE_PATH.exists(), "Dockerfile should exist at project root"


@pytest.mark.parametrize(
    "token",
    [
        "FROM python:3.12-slim",
        "WORKDIR /app",
        "COPY pyproject.toml uv.lock ./",
        "RUN uv sync --frozen --no-dev",
        "COPY summarizer ./summarizer",
        "COPY tests ./tests",
        "CMD [\"uvicorn\", \"summarizer.main:create_app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]",
    ],
)
def test_dockerfile_contains_expected_tokens(token: str, dockerfile_content: str):
    assert token in dockerfile_content, (
        f"Expected to find '{token}' in Dockerfile"
    )


def test_release_workflow_exists():
    assert WORKFLOW_PATH.exists(), "Release workflow should exist"


@pytest.mark.parametrize(
    "token",
    [
        "on:\n  release:\n    types: [published]",
        "docker/build-push-action@v5",
        "GHCR_IMAGE=ghcr.io/${{ github.repository }}",
        "tags: ${{ env.GHCR_IMAGE }}:${{ github.ref_name }}",
        "cache-from: type=gha",
        "cache-to: type=gha,mode=max",
    ],
)
def test_release_workflow_contains_expected_tokens(
    token: str, release_workflow_content: str
):
    assert token in release_workflow_content, (
        f"Expected to find '{token}' in release workflow configuration"
    )
