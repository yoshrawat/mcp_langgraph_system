
from fastapi import FastAPI
from agent_app.api.routes.agent import router as agent_router

app = FastAPI(title="MCP LangGraph Agent API")

app.include_router(agent_router, prefix="/agent", tags=["agent"])

@app.get("/health")
async def health():
    return {"status": "ok"}
