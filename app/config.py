from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    qdrant_api_key: str = ""

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "pdf_chat"

    @field_validator("qdrant_url", "qdrant_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def strip_whitespace(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    max_file_size_mb: int = 50
    max_total_upload_mb: int = 200
    max_files_per_upload: int = 10

    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 2048
    claude_temperature: float = 0.2

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    database_url: str = "sqlite:///./pdf_chat.db"

    rate_limit_chat: str = "20/minute"
    rate_limit_upload: str = "5/minute"


    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    enable_ocr: bool = True

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_total_upload_bytes(self) -> int:
        return self.max_total_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
