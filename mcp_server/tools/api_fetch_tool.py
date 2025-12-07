import aiohttp
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class ApiFetchInput(BaseModel):
    url: str = Field(..., description="Full URL of the public API endpoint.")
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional query parameters."
    )


class ApiFetchOutput(BaseModel):
    status: int
    data: Any
    error: Optional[str] = None


# ---------------------------------------------------------
# Tool Handler
# ---------------------------------------------------------

async def fetch_api_data_tool(input: ApiFetchInput) -> ApiFetchOutput:
    """
    Generic API fetch tool for MCP.
    Fetches a public API URL using aiohttp and returns JSON.
    """

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(input.url, params=input.params, timeout=15) as resp:
                status = resp.status

                # Try parsing JSON response
                try:
                    data = await resp.json()
                except Exception:
                    # Fallback to text if it's not JSON
                    data = await resp.text()

                if status >= 400:
                    return ApiFetchOutput(
                        status=status,
                        data=data,
                        error=f"HTTP error {status}"
                    )

                return ApiFetchOutput(
                    status=status,
                    data=data,
                    error=None
                )

    except Exception as e:
        return ApiFetchOutput(
            status=500,
            data=None,
            error=str(e)
        )


# ---------------------------------------------------------
# Schema Exposure Helpers
# ---------------------------------------------------------

def input_schema():
    return ApiFetchInput.model_json_schema()


def output_schema():
    return ApiFetchOutput.model_json_schema()
