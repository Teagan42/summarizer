# Agent Guide for Summarizer

## Project Overview
- This repository provides the Summarizer service, a FastAPI application for compressing and summarizing textual inputs.
- Requests hit `/compress`, where selection logic picks the most relevant snippets before the compressor produces the final summary.
- Support modules under `app/` handle selection heuristics, compression backends, guardrails, and request/response models.

## Local Development
- Install runtime and tooling dependencies with **uv**, e.g. Use `uv add` for dependencies and `uv sync` to recreate the locked environment from `uv.lock`.
- Run the FastAPI server with `uv run uvicorn app.main:app --reload` for iterative development.
- Configuration is managed through environment variables surfaced in `app/config.py`.

## Quality and Testing Expectations
- Pytest is the primary test runner; execute `uv run pytest` before submitting changes.
- Linting is enforced via Ruff; run `uv run ruff check .` and `uv run ruff format --check .` to validate style.
- Preserve the TDD workflow: introduce a failing test, implement the fix, then refactor while keeping coverage strong.

## Repository Conventions
- Prefer explicit, descriptive module and function names that mirror the summarization pipeline stages.
- Keep prompts and compression settings centralized within the existing modules instead of scattering configuration.
- Document any new API surface or environment variable expectations alongside code changes.
