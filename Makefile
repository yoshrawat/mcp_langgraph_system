VENV=mcp_langgraph_system

setup:
	python3.13 -m venv $(VENV)
	. $(VENV)/bin/activate && pip install uv && uv pip install . && uv pip install ".[dev]"

run-api:
	. $(VENV)/bin/activate && uvicorn agent_app.api.fastapi_app:app --reload --port 8080

run-ui:
	. $(VENV)/bin/activate && streamlit run agent_app/ui/streamlit_app.py

run-mcp:
	. $(VENV)/bin/activate && python mcp_server/run_server.py

test:
	pytest -vv

lint:
	ruff check .

typecheck:
	mypy agent_app mcp_server
