# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY summarizer ./summarizer
COPY tests ./tests

EXPOSE 8000

CMD ["uvicorn", "summarizer.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
