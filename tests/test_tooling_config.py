import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUFF_PATH = PROJECT_ROOT / "ruff.toml"
PYTEST_INI_PATH = PROJECT_ROOT / "pytest.ini"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def ruff_config() -> dict:
    assert RUFF_PATH.exists(), "Expected dedicated Ruff config file"
    return tomllib.loads(RUFF_PATH.read_text())


@pytest.fixture(scope="module")
def pytest_ini_text() -> str:
    assert PYTEST_INI_PATH.exists(), "Expected pytest.ini configuration file"
    return PYTEST_INI_PATH.read_text()


@pytest.fixture(scope="module")
def pyproject() -> dict:
    assert PYPROJECT_PATH.exists(), "Expected pyproject.toml configuration file"
    return tomllib.loads(PYPROJECT_PATH.read_text())


@pytest.fixture(scope="module")
def coverage_run(pyproject: dict) -> dict:
    coverage_config = pyproject["tool"]["coverage"]
    run_config = coverage_config["run"]
    assert "source" in run_config, "Coverage run configuration must declare sources"
    return run_config


def test_coverage_source_targets_project_package(coverage_run: dict):
    assert coverage_run.get("source") == ["summarizer"], (
        "Coverage should track summarizer package"
    )


def test_ruff_config_targets_py312(ruff_config: dict):
    assert ruff_config.get("target-version") == "py312"


def test_ruff_config_sets_line_length(ruff_config: dict):
    assert ruff_config.get("line-length") == 88


def test_pytest_ini_declares_minimum_settings(pytest_ini_text: str):
    assert "[pytest]" in pytest_ini_text
    assert "addopts = --maxfail=1 --disable-warnings -q" in pytest_ini_text
