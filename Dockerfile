FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV RUFF_CACHE_DIR=/tmp/ruff-cache PYTEST_ADDOPTS="-p no:cacheprovider"
ENV PATH="/srv/ai-support/.venv/bin:$PATH"
WORKDIR /srv/ai-support
COPY pyproject.toml uv.lock ./
COPY app ./app
RUN uv sync --frozen --no-dev --extra test --no-editable && useradd --uid 10001 --create-home support
COPY alembic.ini ./
COPY migrations ./migrations
COPY sample_knowledge ./sample_knowledge
COPY tests ./tests
COPY scripts ./scripts
USER support
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
