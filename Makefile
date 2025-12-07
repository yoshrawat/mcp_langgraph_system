
.PHONY: install lint typecheck test api ui docker-build docker-run

install:
	uv pip install . --system

lint:
	ruff check .

typecheck:
	mypy agent_app

test:
	pytest -q

api:
	uvicorn agent_app.api.fastapi_app:app --reload --port 8080

ui:
	streamlit run agent_app/ui/streamlit_app.py

docker-build:
	docker build -t ghcr.io/yoshrawat/mcp_langgraph_system:dev .

docker-run:
	docker run -p 8080:8080 ghcr.io/yoshrawat/mcp_langgraph_system:dev
