from pathlib import Path


def test_agents_md_exists_and_mentions_uv_add():
    repo_root = Path(__file__).resolve().parents[1]
    agents_path = repo_root / "AGENTS.md"

    assert agents_path.exists(), "Expected AGENTS.md at repository root"

    content = agents_path.read_text(encoding="utf-8")
    assert "Use `uv add` for dependencies" in content

    content_lower = content.lower()
    assert "summarizer" in content_lower
    assert "project" in content_lower
