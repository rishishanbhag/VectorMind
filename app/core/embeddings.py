import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
