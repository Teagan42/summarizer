from pathlib import Path


def read_agents_content() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    agents_path = repo_root / "AGENTS.md"

    assert agents_path.exists(), "Expected AGENTS.md at repository root"

    return agents_path.read_text(encoding="utf-8")


def test_agents_md_exists_and_mentions_uv_add():
    content = read_agents_content()
    assert "Use `uv add` for dependencies" in content

    content_lower = content.lower()
    assert "summarizer" in content_lower
    assert "project" in content_lower


def test_agents_md_instructs_running_fix_and_format_commands():
    content = read_agents_content()
    assert "uv run ruff check --fix ." in content
    assert "uv run format ." in content
