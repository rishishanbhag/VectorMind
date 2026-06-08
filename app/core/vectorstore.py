import logging
import uuid
from typing import List, Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import get_settings
from app.core.embeddings import get_embedding_model

logger = logging.getLogger(__name__)
settings = get_settings()


class QdrantVectorStore:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.collection_name = f"{settings.qdrant_collection_prefix}_user_{user_id}"
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embeddings = get_embedding_model()
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)

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

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
        )

        documents = []
        for hit in results:
            payload = dict(hit.payload or {})
            content = payload.pop("content", "")
            documents.append(Document(page_content=content, metadata=payload))
        return documents

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def delete_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            self.client.delete_collection(self.collection_name)


def get_vectorstore(user_id: int) -> QdrantVectorStore:
    return QdrantVectorStore(user_id)
