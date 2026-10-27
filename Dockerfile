# Один образ для обучения (python -m bot_detector.train) и сервиса (uvicorn):
# одинаковые версии библиотек гарантируют, что сохранённый пайплайн загрузится в сервисе.
FROM python:3.12.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /bin/uv

# libgomp1 — OpenMP для lightgbm; curl — для healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# зависимости отдельным слоем: при изменении кода не переустанавливаются
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY bot_detector ./bot_detector

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "bot_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
