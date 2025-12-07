
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
import time
from starlette.requests import Request

# -----------------------------
# Metric Definitions
# -----------------------------

API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint"]
)

API_REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Latency of API requests",
    ["method", "endpoint"]
)

AGENT_EXECUTION_LATENCY = Histogram(
    "agent_execution_duration_seconds",
    "Execution time of LangGraph agent",
)

LLM_LATENCY = Histogram(
    "llm_call_duration_seconds",
    "LLM call latency histogram"
)

RAG_LATENCY = Histogram(
    "rag_search_duration_seconds",
    "RAG search execution time"
)

TOOL_CALL_COUNT = Counter(
    "tool_calls_total",
    "Number of MCP tool calls",
    ["tool_name"]
)

TOOL_ERRORS = Counter(
    "tool_errors_total",
    "Number of errors from MCP tool calls",
    ["tool_name"]
)

# -----------------------------
# Middleware for API Requests
# -----------------------------

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method
        endpoint = request.url.path

        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        API_REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        API_REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response
