from __future__ import annotations
from typing import List
from langchain_core.messages import BaseMessage, AIMessage

class GeminiProvider:
    async def acomplete(self, messages: List[BaseMessage]) -> AIMessage:
        raise NotImplementedError("Gemini provider requires Google API key.")
