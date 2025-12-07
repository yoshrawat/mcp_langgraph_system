from __future__ import annotations
import httpx
from typing import List
from agent_app.core.config.settings import get_settings
from agent_app.core.agent.interfaces import LLMProviderInterface
from langchain_core.messages import BaseMessage, AIMessage

class OllamaProvider(LLMProviderInterface):

    async def acomplete(self, messages: List[BaseMessage]) -> AIMessage:
        settings = get_settings()
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"

        payload = {
            "model": settings.OLLAMA_MODEL,
            "messages": [{"role": m.type, "content": m.content} for m in messages]
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return AIMessage(content=data["message"]["content"])
