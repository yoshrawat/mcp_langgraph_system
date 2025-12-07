from fastapi import APIRouter, Query

from agent_app.core.audit_logger import audit_logger


router = APIRouter(prefix="/tools", tags=["Tools"])


@router.get("/logs")
async def get_tool_logs(
    session_id: str | None = Query(
        default=None,
        description="Filter logs for a specific session."
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Max number of log entries to return."
    )
):
    """
    Retrieve tool audit logs.

    Tool logs include:
      - timestamp
      - session ID
      - tool name
      - arguments
      - results (JSON)
    """

    logs = audit_logger.list_logs(session_id=session_id, limit=limit)

    return {
        "count": len(logs),
        "session_id": session_id,
        "logs": logs
    }
