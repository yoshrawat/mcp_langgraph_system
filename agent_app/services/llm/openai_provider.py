from __future__ import annotations
from typing import List
from langchain_core.messages import BaseMessage, AIMessage

class OpenAIProvider:
    async def acomplete(self, messages: List[BaseMessage]) -> AIMessage:
        raise NotImplementedError("OpenAI provider implementation pending API key.")
