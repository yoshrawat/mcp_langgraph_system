import uvicorn

def main():
    """
    Production entrypoint for running the FastAPI server.
    This avoids import path issues and makes Docker execution cleaner.
    """
    uvicorn.run(
        "agent_app.api.fastapi_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,           # Auto-reload for development
        workers=1,             # Increase in production
    )


if __name__ == "__main__":
    main()
