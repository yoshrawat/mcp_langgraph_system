
from fastapi import APIRouter, HTTPException
from agent_app.api.models.request_models import AgentRequest
from agent_app.api.models.response_models import AgentResponse
from agent_app.core.agent.graph import build_agent_graph
from agent_app.core.agent.state import AgentState

router = APIRouter()

@router.post("/query", response_model=AgentResponse)
async def query_agent(payload: AgentRequest):
    try:
        graph_factory = build_agent_graph()
        graph = await graph_factory(payload.session_id)

        state = AgentState(messages=[{"type": "human", "content": payload.query}])
        result = await graph.ainvoke(state)
        return AgentResponse(answer=result.get("final_answer", "No response"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
