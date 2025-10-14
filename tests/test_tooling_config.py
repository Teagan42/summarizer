import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUFF_PATH = PROJECT_ROOT / "ruff.toml"
PYTEST_INI_PATH = PROJECT_ROOT / "pytest.ini"


@pytest.fixture(scope="module")
def ruff_config() -> dict:
    assert RUFF_PATH.exists(), "Expected dedicated Ruff config file"
    return tomllib.loads(RUFF_PATH.read_text())


@pytest.fixture(scope="module")
def pytest_ini_text() -> str:
    assert PYTEST_INI_PATH.exists(), "Expected pytest.ini configuration file"
    return PYTEST_INI_PATH.read_text()


def test_ruff_config_targets_py312(ruff_config: dict):
    assert ruff_config.get("target-version") == "py312"


def test_ruff_config_sets_line_length(ruff_config: dict):
    assert ruff_config.get("line-length") == 88


def test_pytest_ini_declares_minimum_settings(pytest_ini_text: str):
    assert "[pytest]" in pytest_ini_text
    assert "addopts = --maxfail=1 --disable-warnings -q" in pytest_ini_text
