# Streamlit UI stub
import streamlit as st
import uuid

from agent_app.core.agent_graph import AgentGraph
from agent_app.core.state import AgentState, Message


# ------------------------------------------------------
# Page config
# ------------------------------------------------------
st.set_page_config(
    page_title="MCP LangGraph Agent",
    layout="wide"
)

st.title("🤖 MCP-Powered LangGraph Assistant")


# ------------------------------------------------------
# Initialize agent & session
# ------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = AgentGraph(
        mcp_endpoint="python mcp_server/run_server.py",
        model="llama3.2:latest"
    )

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = AgentState(
        session_id=st.session_state.session_id
    )


agent = st.session_state.agent
conversation_state = st.session_state.conversation_state
session_id = st.session_state.session_id


# ------------------------------------------------------
# Sidebar
# ------------------------------------------------------
with st.sidebar:
    st.subheader("Session Info")
    st.text(f"Session ID:\n{session_id}")

    if st.button("🔄 Reset Session"):
        st.session_state.conversation_state = AgentState(
            session_id=session_id
        )
        st.experimental_rerun()


# ------------------------------------------------------
# Display conversation history
# ------------------------------------------------------
st.subheader("Chat")

for msg in conversation_state.messages:
    if msg.role == "human":
        st.chat_message("user").write(msg.content)
    elif msg.role == "assistant":
        st.chat_message("assistant").write(msg.content)
    else:  # tool
        st.chat_message("assistant").write(f"🛠️ Tool Result:\n\n{msg.content}")


# ------------------------------------------------------
# Input box
# ------------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Add to UI immediately
    st.chat_message("user").write(user_input)

    # Call agent
    reply_state = agent.run(
        session_id=session_id,
        user_input=user_input,
        prior_state=conversation_state
    )

    # Store updated state
    st.session_state.conversation_state = reply_state

    # Display assistant reply
    if reply_state.final_response:
        st.chat_message("assistant").write(reply_state.final_response)
