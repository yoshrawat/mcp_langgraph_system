
def test_tool_registry_loads():
    from agent_app.core.tools.registry import load_tools

    tools = load_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
