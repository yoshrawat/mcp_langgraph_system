
# ============
# Stage 1: Builder
# ============
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .
COPY agent_app ./agent_app

RUN pip install --upgrade pip \
    && pip install uv \
    && uv pip install streamlit requests . --system

# ============
# Stage 2: Runtime
# ============
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 8501

CMD ["streamlit", "run", "agent_app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
