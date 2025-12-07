from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

from ..state import AgentState, append_assistant_message


class LLMNode:
    """
    LLM node that generates assistant responses using Ollama.

    The node:
      - Reads conversation messages from AgentState
      - Converts them to LangChain message objects
      - Calls ChatOllama to generate a reply
      - Appends result to AgentState.messages
      - Writes the final LLM output into state.final_response
    """

    def __init__(self, model: str = "llama3"):
        self.llm = ChatOllama(
            model=model,
            temperature=0.2,
        )

    # ------------------------------------------------------
    # Convert AgentState messages to LangChain LLM messages
    # ------------------------------------------------------
    def _convert_messages(self, state: AgentState) -> List[Any]:
        converted = []
        for m in state.messages:
            if m.role == "human":
                converted.append(HumanMessage(content=m.content))
            else:
                # "assistant" or "tool" both appear as AIMessage to model
                converted.append(AIMessage(content=m.content))
        return converted

    # ------------------------------------------------------
    # Main LLM node execution
    # ------------------------------------------------------
    async def run(self, state: AgentState) -> AgentState:
        messages = self._convert_messages(state)

        # Call Ollama LLM
        reply = await self.llm.ainvoke(messages)

        response_text = reply.content
        append_assistant_message(state, response_text)

        # Set final result of this LLM turn
        state.final_response = response_text
        state.next_step = "done"

        return state
