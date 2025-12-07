
from __future__ import annotations
from typing import Any, Dict, List
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from agent_app.core.agent.state import AgentState
from agent_app.core.agent.llm_provider import get_llm
from agent_app.core.agent.tools import load_all_tools
from agent_app.services.mcp.client import MCPAsyncClient
from agent_app.services.storage.audit_repo import AuditRepository


# -----------------------------
# Planner Node
# -----------------------------
async def planner_node(state: AgentState, *, session_id: str, tools):
    llm = get_llm(tools=tools)

    messages = state.messages

    response: AIMessage = await llm.ainvoke(messages)

    # Must always return structured function_call
    if response.additional_kwargs.get("function_call"):
        fc = response.additional_kwargs["function_call"]
        tool = fc.get("name")
        args = fc.get("arguments", "{}")

        try:
            parsed_args = json.loads(args)
        except Exception:
            parsed_args = {"raw": args}

        return {
            "messages": [
                AIMessage(
                    content="",
                    additional_kwargs={
                        "function_call": {
                            "name": tool,
                            "arguments": parsed_args,
                        }
                    },
                )
            ],
            "next_tool": tool,
            "next_args": parsed_args,
        }

    # Fallback if LLM violates strict mode
    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content,
    }


# -----------------------------
# Executor Node
# -----------------------------
async def executor_node(state: AgentState, *, session_id: str, tools):
    tool_name = state.get("next_tool")
    args = state.get("next_args", {})

    if not tool_name:
        return {
            "messages": [
                AIMessage(content="No tool selected."),
            ],
            "final_answer": "No tool selected.",
        }

    tool_map = {t.name: t for t in tools}

    if tool_name not in tool_map:
        return {
            "messages": [
                AIMessage(content=f"Unknown tool {tool_name}."),
            ],
            "final_answer": f"Unknown tool {tool_name}.",
        }

    tool = tool_map[tool_name]

    result = await tool.ainvoke(args)

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(result),
                tool_call_id=tool_name,
                name=tool_name,
            )
        ],
        "tool_result": result,
    }


# -----------------------------
# Finalizer Node
# -----------------------------
async def finalizer_node(state: AgentState, *, session_id: str):
    # Return the last assistant message
    messages = state.messages
    last_msg = messages[-1]
    return {"final_answer": last_msg.content}


# -----------------------------
# Build LangGraph Hybrid Agent
# -----------------------------
def build_agent_graph():
    mcp_client = MCPAsyncClient()
    audit = AuditRepository()

    async def create_graph(session_id: str):
        tools = await load_all_tools(
            mcp_client=mcp_client,
            audit_repo=audit,
            session_id=session_id
        )

        graph = StateGraph(AgentState)

        graph.add_node("planner", lambda s: planner_node(s, session_id=session_id, tools=tools))
        graph.add_node("executor", lambda s: executor_node(s, session_id=session_id, tools=tools))
        graph.add_node("finalizer", lambda s: finalizer_node(s, session_id=session_id))

        # planner → executor
        graph.add_edge("planner", "executor")

        # executor → planner (loop)
        graph.add_edge("executor", "planner")

        # Stop condition
        graph.add_conditional_edges(
            "planner",
            lambda s: "final_answer" in s,
            {
                True: "finalizer",
                False: "executor",
            },
        )

        graph.add_edge("finalizer", END)

        return graph.compile()

    return create_graph
