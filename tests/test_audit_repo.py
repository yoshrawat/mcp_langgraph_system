
def test_audit_logging():
    from agent_app.core.audit.repo import AuditRepository

    repo = AuditRepository(":memory:")
    repo.log("search_api_results", {"query":"hello"}, {"result":"ok"})

    rows = repo.query_by_tool("search_api_results")
    assert len(rows) == 1
    assert rows[0]["tool"] == "search_api_results"
