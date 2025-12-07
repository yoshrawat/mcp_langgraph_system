import streamlit as st
from agent_app.core.agent_graph import AgentGraph
from agent_app.core.state import AgentState

import uuid


# -------------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="MCP LangGraph Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MCP LangGraph Agent — Streamlit UI")


# -------------------------------------------------------------------
# Session Initialization
# -------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = AgentState(
        session_id=st.session_state["session_id"]
    )

if "agent" not in st.session_state:
    # MCP server is started separately; agent communicates via STDIO
    st.session_state["agent"] = AgentGraph(
        mcp_endpoint="python mcp_server/run_server.py",
        model="llama3"
    )


agent: AgentGraph = st.session_state["agent"]
agent_state: AgentState = st.session_state["agent_state"]


# -------------------------------------------------------------------
# Streamlit Chat Interface
# -------------------------------------------------------------------

# Display prior messages
for msg in agent_state.messages:
    if msg.role == "human":
        with st.chat_message("user"):
            st.write(msg.content)
    elif msg.role == "assistant":
        with st.chat_message("assistant"):
            st.write(msg.content)
    elif msg.role == "tool":
        with st.chat_message("assistant"):
            st.code(f"[TOOL OUTPUT] {msg.tool_name}\n\n{msg.content}")


# User input
user_input = st.chat_input("Ask something...")

if user_input:

    # Display user msg
    with st.chat_message("user"):
        st.write(user_input)

    # Update agent state with new message
    agent_state.messages.append(
        agent_state.new_message(role="human", content=user_input)
    )

    # Call the agent
    new_state = agent.run(
        session_id=agent_state.session_id,
        user_input=user_input,
        prior_state=agent_state
    )

    # Replace stored state
    st.session_state["agent_state"] = new_state
    agent_state = new_state

    # Render assistant output
    if agent_state.final_response:
        with st.chat_message("assistant"):
            st.write(agent_state.final_response)

    # Render tool events
    elif agent_state.tool_response:
        with st.chat_message("assistant"):
            st.code(f"[TOOL RESULT]\n\n{agent_state.tool_response}")
