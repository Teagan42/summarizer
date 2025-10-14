from pathlib import Path

import pytest

WORKFLOW_PATH = Path(".github/workflows/ci.yaml")


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
    ],
)
def test_ci_workflow_contains_expected_tokens(token):
    content = WORKFLOW_PATH.read_text()
    assert token in content, f"Expected to find '{token}' in CI workflow configuration"
