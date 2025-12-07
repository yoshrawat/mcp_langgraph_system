
# ============
# Stage 1: Builder
# ============
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .
COPY agent_app ./agent_app

RUN pip install --upgrade pip \
    && pip install uv \
    && uv pip install . --system

# ============
# Stage 2: Runtime
# ============
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 8080

CMD ["uvicorn", "agent_app.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]
