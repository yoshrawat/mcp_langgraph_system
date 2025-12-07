"""
LLM Node for LangGraph Agent
----------------------------

This node is responsible for:
 - Producing assistant responses
 - Generating structured tool call instructions
 - Incorporating tool responses
 - Updating the AgentState messages list

The node uses the LangChain ChatModel with tool calling enabled.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent_app.core.state import AgentState


class LLMNode:
    """
    Wrapper for an LLM that:
      - Accepts AgentState (history + context)
      - Produces assistant messages or tool-call instructions
      - Returns updated AgentState
    """

    def __init__(self, llm):
        self.llm = llm  # ChatModel (Ollama, OpenAI, etc.)

    async def __call__(self, state: AgentState) -> AgentState:
        """
        Executes an LLM prediction step.
        """

        # ----------------------------------------------------
        # 1. Construct message history for the LLM
        # ----------------------------------------------------

        messages = []

        for msg in state.messages:
            if msg.role == "human":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            elif msg.role == "tool":
                messages.append(
                    ToolMessage(
                        name=msg.tool_name,
                        content=msg.content
                    )
                )

        # Last user input
        messages.append(HumanMessage(content=state.user_input))

        # Include tool result if present
        if state.tool_response is not None:
            messages.append(
                ToolMessage(
                    name="tool_result",
                    content=str(state.tool_response)
                )
            )

        # ----------------------------------------------------
        # 2. Call LLM model
        # ----------------------------------------------------

        response = await self.llm.ainvoke(messages)

        # ----------------------------------------------------
        # 3. Check for a tool call
        # ----------------------------------------------------

        if response.tool_calls:
            # LLM requested a tool
            tool_call = response.tool_calls[0]

            state.pending_tool_call = {
                "name": tool_call["name"],
                "args": tool_call["args"],
            }

            # Record the assistant message that invoked tool
            state.messages.append(
                state.new_message(
                    role="assistant",
                    content=f"[TOOL CALL] {tool_call['name']} → {tool_call['args']}"
                )
            )

            return state

        # ----------------------------------------------------
        # 4. No tool call → produce normal assistant response
        # ----------------------------------------------------

        assistant_text = response.content

        state.messages.append(
            state.new_message(role="assistant", content=assistant_text)
        )

        # Mark final output so router knows when to stop
        state.final_response = assistant_text

        return state
