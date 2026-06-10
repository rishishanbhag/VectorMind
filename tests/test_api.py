from unittest.mock import MagicMock, patch

TEST_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def session_headers(session_id: str = TEST_SESSION_ID) -> dict:
    return {"X-Session-Id": session_id}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "VectorMindbot API" in response.json()["message"]


def test_status_without_session(client):
    response = client.get("/status")
    assert response.status_code == 400


def test_status_with_session(client):
    response = client.get("/status", headers=session_headers())
    assert response.status_code == 200
    data = response.json()
    assert "vectorstore_loaded" in data
    assert data["user_id"] == 1


def test_same_session_reuses_user(client):
    headers = session_headers()
    first = client.get("/status", headers=headers)
    second = client.get("/status", headers=headers)
    assert first.json()["user_id"] == second.json()["user_id"]


def test_concurrent_same_session_creates_one_user(client):
    from concurrent.futures import ThreadPoolExecutor

    session_id = "c3d4e5f6-a7b8-9012-cdef-123456789012"
    headers = session_headers(session_id)

    def fetch_status():
        return client.get("/status", headers=headers)

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: fetch_status(), range(4)))

    assert all(r.status_code == 200 for r in responses)
    user_ids = {r.json()["user_id"] for r in responses}
    assert len(user_ids) == 1


def test_different_sessions_get_different_users(client):
    first = client.get("/status", headers=session_headers(TEST_SESSION_ID))
    second = client.get(
        "/status",
        headers=session_headers("b2c3d4e5-f6a7-8901-bcde-f12345678901"),
    )
    assert first.json()["user_id"] != second.json()["user_id"]


def test_chat_without_documents(client):
    response = client.post(
        "/chat",
        json={"question": "Hello?"},
        headers=session_headers(),
    )
    assert response.status_code == 503


def test_chat_empty_question(client):
    response = client.post(
        "/chat",
        json={"question": "  "},
        headers=session_headers(),
    )
    assert response.status_code == 400


def test_upload_no_files(client):
    response = client.post("/upload", headers=session_headers())
    assert response.status_code == 422


@patch("app.core.vectorstore.QdrantClient")
def test_upload_accepts_pdf(mock_qdrant, client):
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_qdrant.return_value = mock_client

    pdf_content = b"%PDF-1.4 minimal"
    response = client.post(
        "/upload",
        headers=session_headers(),
        files=[("files", ("test.pdf", pdf_content, "application/pdf"))],
    )
    assert response.status_code == 202
    assert "task_id" in response.json()
