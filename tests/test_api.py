
def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status":"ok"}

def test_agent_query(api_client):
    payload = {
        "session_id": "123",
        "query": "hello"
    }
    r = api_client.post("/agent/query", json=payload)
    assert r.status_code == 200
    assert "answer" in r.json()
