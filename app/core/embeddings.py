import gc
import logging
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_embedding_model: Optional[Any] = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
        )
    return _embedding_model


def unload_embedding_model() -> None:
    """Release embedding model from RAM (used in low-memory mode)."""
    global _embedding_model
    if _embedding_model is not None:
        logger.info("Unloading embedding model")
        del _embedding_model
        _embedding_model = None
        gc.collect()
