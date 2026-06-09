import logging
import time
import uuid
from functools import lru_cache
from typing import Callable, List, TypeVar

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

T = TypeVar("T")
MAX_QDRANT_RETRIES = 4
RETRY_BACKOFF_SEC = 1.5


def _normalize_qdrant_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if "cloud.qdrant.io" in normalized and ":" not in normalized.split("//", 1)[-1]:
        normalized = f"{normalized}:6333"
    return normalized


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    url = _normalize_qdrant_url(settings.qdrant_url)
    kwargs = {
        "url": url,
        "timeout": 60,
        "prefer_grpc": False,
    }
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
    logger.info("Connecting to Qdrant at %s", url)
    return QdrantClient(**kwargs)


def _retry_qdrant(operation: Callable[[], T], description: str) -> T:
    last_error = None
    for attempt in range(1, MAX_QDRANT_RETRIES + 1):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt == MAX_QDRANT_RETRIES:
                break
            wait = RETRY_BACKOFF_SEC * attempt
            logger.warning(
                "Qdrant %s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                MAX_QDRANT_RETRIES,
                e,
                wait,
            )
            time.sleep(wait)
    raise last_error


def get_vectorstore_stats(user_id: int) -> tuple[int, str | None]:
    """Return document count and optional error without raising."""
    try:
        store = QdrantVectorStore(user_id)
        return store.count(), None
    except Exception as e:
        logger.warning("Qdrant unavailable for user %s: %s", user_id, e)
        return 0, str(e)


class QdrantVectorStore:
    def __init__(self, user_id: int):
        settings = get_settings()
        self.user_id = user_id
        self.collection_name = f"{settings.qdrant_collection_prefix}_user_{user_id}"
        self.client = _get_qdrant_client()
        self.embeddings = get_embedding_model()
        self._ensure_collection()

    def _ensure_collection(self):
        def _ensure():
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: %s", self.collection_name)

        _retry_qdrant(_ensure, "ensure_collection")

    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"content": text, **meta},
            )
            for text, meta, vector in zip(texts, metadatas, vectors)
        ]

        def _upsert():
            self.client.upsert(collection_name=self.collection_name, points=points)

        _retry_qdrant(_upsert, "upsert")
        return len(points)

    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)

        def _search():
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
            )

        results = _retry_qdrant(_search, "search")

        documents = []
        for hit in results:
            payload = dict(hit.payload or {})
            content = payload.pop("content", "")
            documents.append(Document(page_content=content, metadata=payload))
        return documents

    def count(self) -> int:
        def _count():
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0

        return _retry_qdrant(_count, "count")

    def delete_collection(self):
        def _delete():
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name in collections:
                self.client.delete_collection(self.collection_name)

        _retry_qdrant(_delete, "delete_collection")


def get_vectorstore(user_id: int) -> QdrantVectorStore:
    return QdrantVectorStore(user_id)
