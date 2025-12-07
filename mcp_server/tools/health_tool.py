from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class HealthToolInput(BaseModel):
    """Health check tool requires no input."""
    pass


class HealthToolOutput(BaseModel):
    status: str
    timestamp: str
    detail: Optional[str] = None


# ---------------------------------------------------------
# Tool Handler
# ---------------------------------------------------------

async def health_check_tool(_: HealthToolInput) -> HealthToolOutput:
    """
    A simple health check tool that returns server status.
    """

    return HealthToolOutput(
        status="ok",
        timestamp=datetime.utcnow().isoformat(),
        detail="MCP server alive and operational"
    )


# ---------------------------------------------------------
# Schema Exposure Helpers
# ---------------------------------------------------------

def input_schema():
    return HealthToolInput.model_json_schema()


def output_schema():
    return HealthToolOutput.model_json_schema()
