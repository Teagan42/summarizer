from pathlib import Path

import pytest

WORKFLOW_PATH = Path(".github/workflows/ci.yaml")


@pytest.fixture(scope="module")
def workflow_content() -> str:
    return WORKFLOW_PATH.read_text()


def test_ci_workflow_exists():
    assert WORKFLOW_PATH.exists(), "CI workflow file should exist"


@pytest.mark.parametrize(
    "token",
    [
        "on:\n  pull_request:",
        "jobs:",
        "lint:",
        "format:",
        "build:",
        "test:",
        "Coverage comment",
        "pytest-coverage-comment",
        "uv venv",
    ],
)
def test_ci_workflow_contains_expected_tokens(token, workflow_content: str):
    assert token in workflow_content, (
        f"Expected to find '{token}' in CI workflow configuration"
    )


def test_ci_workflow_pytest_step_enables_pipefail(workflow_content: str):
    assert "set -o pipefail" in workflow_content
