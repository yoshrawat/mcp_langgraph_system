import httpx


async def api_fetch_tool(url: str, timeout: int = 10) -> dict:
    """
    Fetch data from a remote API endpoint.

    Args:
        url (str): The remote endpoint to fetch.
        timeout (int): Timeout in seconds (default: 10)

    Returns:
        dict: A dictionary containing response status, text, headers, JSON if available.

    This tool is frequently used for:
      - RAG ingestion
      - API testing
      - Agent tool execution
    """

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url)
            raw_text = response.text

            # Try parsing JSON (best-effort)
            try:
                json_body = response.json()
            except Exception:
                json_body = None

            return {
                "url": url,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "text": raw_text,
                "json": json_body,
            }

        except httpx.RequestError as exc:
            return {
                "error": f"Failed to fetch URL: {url}",
                "reason": str(exc),
            }
