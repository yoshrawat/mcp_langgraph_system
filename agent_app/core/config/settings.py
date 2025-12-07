from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """
    Application settings following 12-factor rules.
    Loaded from environment variables.
    """

    # General
    APP_ENV: Literal["development", "production", "test"] = "development"
    DEBUG: bool = True

    # LLM Provider Selection
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "gemini"] = "ollama"

    # Vector DB Backend
    VECTOR_BACKEND: Literal["chroma", "sqlite", "pgvector", "faiss"] = "chroma"

    # MCP Server
    MCP_SERVER_URL: str = Field(
        default="http://mcp_server:8000/mcp",
        description="URL for MCP server (used by the agent)",
    )

    # Database paths
    SQLITE_AUDIT_PATH: str = "audit.db"
    SQLITE_VECTOR_PATH: str = "vector_store.db"

    # Ollama Model
    OLLAMA_MODEL: str = "llama3"

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080

    # Prometheus Settings
    PROMETHEUS_ENABLED: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
