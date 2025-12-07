from __future__ import annotations
from typing import List
from langchain_core.messages import BaseMessage, AIMessage

class AnthropicProvider:
    async def acomplete(self, messages: List[BaseMessage]) -> AIMessage:
        raise NotImplementedError("Anthropic provider requires API integration.")
