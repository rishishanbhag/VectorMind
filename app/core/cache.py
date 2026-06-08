import hashlib
import time
from functools import lru_cache
from typing import Optional

_answer_cache: dict[str, tuple[float, dict]] = {}
ANSWER_CACHE_TTL = 3600


@lru_cache(maxsize=256)
def cached_embed_query(query: str, embed_fn_id: int) -> tuple:
    """Cache query embeddings. embed_fn_id identifies the embedding function instance."""
    return query


def get_query_cache_key(user_id: int, question: str) -> str:
    return hashlib.sha256(f"{user_id}:{question.lower().strip()}".encode()).hexdigest()


def get_cached_answer(user_id: int, question: str) -> Optional[dict]:
    key = get_query_cache_key(user_id, question)
    entry = _answer_cache.get(key)
    if entry and (time.time() - entry[0]) < ANSWER_CACHE_TTL:
        return entry[1]
    return None


def set_cached_answer(user_id: int, question: str, answer: dict):
    key = get_query_cache_key(user_id, question)
    _answer_cache[key] = (time.time(), answer)
