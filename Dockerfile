# FastAPI + shunya library (Timescale-backed API). Build from repo root.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --extra api --extra timescale --no-install-project

COPY shunya/ shunya/
COPY api/ api/

RUN uv sync --frozen --no-dev --extra api --extra timescale

COPY docker/api-entrypoint.sh /api-entrypoint.sh
RUN chmod +x /api-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/api-entrypoint.sh"]
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
