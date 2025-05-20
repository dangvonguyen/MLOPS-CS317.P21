FROM python:3.12-slim AS base

WORKDIR /app

RUN groupadd -r app_group && useradd -r -g app_group app

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --no-editable

COPY model/ ./model/

COPY api.py .

FROM base AS runtime

COPY --from=builder --chown=app:app_group /app /app

USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/app/.venv/nltk_data

RUN python -c "import nltk; nltk.download('stopwords', download_dir='/app/.venv/nltk_data')"

EXPOSE 8000

CMD ["python", "api.py"]

