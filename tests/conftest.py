import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("ENABLE_RERANKER", "false")
os.environ.setdefault("LOW_MEMORY_MODE", "true")

from app.database import Base, get_db
from app.main import create_app

TEST_SESSION_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def session_headers(session_id: str = TEST_SESSION_ID) -> dict:
    return {"X-Session-Id": session_id}


@pytest.fixture(autouse=True)
def mock_qdrant():
    with patch("app.core.vectorstore.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client.get_collection.return_value.points_count = 0
        mock_client.search.return_value = []
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def db_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def client(db_engine):
    app = create_app()
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
