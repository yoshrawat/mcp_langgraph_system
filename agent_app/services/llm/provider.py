from __future__ import annotations
from typing import Dict
from agent_app.core.config.settings import get_settings
from agent_app.core.agent.interfaces import LLMProviderInterface
from agent_app.services.llm.ollama_provider import OllamaProvider
# Other providers imported lazily

def get_llm_provider() -> LLMProviderInterface:
    settings = get_settings()
    provider = settings.LLM_PROVIDER.lower()

    if provider == "ollama":
        return OllamaProvider()

    if provider == "openai":
        from agent_app.services.llm.openai_provider import OpenAIProvider
        return OpenAIProvider()

    if provider == "anthropic":
        from agent_app.services.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider()

    if provider == "gemini":
        from agent_app.services.llm.gemini_provider import GeminiProvider
        return GeminiProvider()

    raise ValueError(f"Unsupported LLM provider: {provider}")
