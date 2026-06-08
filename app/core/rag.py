import json
import logging
from functools import lru_cache
from typing import AsyncGenerator, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.cache import get_cached_answer, set_cached_answer
from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)
settings = get_settings()

RAG_SYSTEM_INSTRUCTIONS = """You are a precise document assistant. Answer only using the provided context.

Formatting rules (strict):
- Do NOT use emojis or decorative symbols.
- Write in a clear, professional, academic tone.
- Use markdown headings (##, ###) and bullet points for structure.
- Use numbered sections when summarizing multiple topics.
- Keep paragraphs short and scannable.
- Do NOT paste raw source excerpts or chunk text into the answer.
- Do NOT list "source(s)" at the end — citations are handled separately by the UI.
- If the context only partially covers the question, say what is covered and what is missing.
- Prefer direct, well-organized explanations over conversational filler.
- Do not start with phrases like "Based on the provided context" unless the context is genuinely insufficient."""

# Per-user conversation memory (in-memory; history persisted in DB separately)
_user_memories: dict[int, ConversationBufferMemory] = {}


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.reranker_model)
    except Exception as e:
        logger.warning("Reranker unavailable: %s", e)
        return None


def rerank_documents(query: str, documents: List[Document], top_k: int) -> List[Document]:
    reranker = _get_reranker()
    if reranker is None or not documents:
        return documents[:top_k]

    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


def chunk_documents(pages: list, filename: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    documents = []
    for text, metadata in pages:
        if not text.strip():
            continue
        metadata = {**metadata, "filename": filename}
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk_index": idx},
                )
            )
    return documents


def get_llm(streaming: bool = False) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.claude_model,
        temperature=settings.claude_temperature,
        max_tokens=settings.claude_max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
        streaming=streaming,
    )


def _build_context(documents: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.metadata
        source = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        parts.append(f"[Source {i}: {source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _format_sources(documents: List[Document]) -> List[dict]:
    return [
        {
            "content": doc.page_content[:500],
            "metadata": {
                "filename": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
            },
        }
        for doc in documents
    ]


def _get_memory(user_id: int) -> ConversationBufferMemory:
    if user_id not in _user_memories:
        _user_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    return _user_memories[user_id]


def retrieve_context(user_id: int, question: str) -> tuple[List[Document], List[Document]]:
    store = get_vectorstore(user_id)
    if store.count() == 0:
        raise ValueError("No documents indexed. Please upload PDFs first.")

    candidates = store.similarity_search(question, k=settings.retrieval_top_k)
    reranked = rerank_documents(question, candidates, settings.rerank_top_k)
    return reranked, candidates


def _build_rag_prompt(question: str, context: str, history_text: str = "") -> str:
    history_block = f"\nPrevious conversation:\n{history_text}\n" if history_text.strip() else ""
    return f"""{RAG_SYSTEM_INSTRUCTIONS}

Context from uploaded documents:
{context}
{history_block}
Question: {question}

Answer:"""


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def ask_question(user_id: int, question: str, conversation_id: Optional[str] = None) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    cached = get_cached_answer(user_id, question)
    if cached:
        logger.info("Cache hit for user=%s", user_id)
        return cached

    reranked, _ = retrieve_context(user_id, question)
    context = _build_context(reranked)
    memory = _get_memory(user_id)
    history = memory.load_memory_variables({}).get("chat_history", [])

    history_text = ""
    for msg in history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Assistant: {msg.content}\n"

    prompt = _build_rag_prompt(question, context, history_text)

    input_tokens = estimate_tokens(prompt)
    logger.info("Estimated input tokens: %d (max output: %d)", input_tokens, settings.claude_max_tokens)

    llm = get_llm(streaming=False)
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    memory.save_context({"question": question}, {"answer": answer})

    sources = _format_sources(reranked)
    usage = getattr(response, "usage_metadata", None) or {}
    logger.info("Chat complete user=%s input_tokens=%d usage=%s", user_id, input_tokens, usage)

    result = {
        "answer": answer,
        "sources": sources,
        "conversation_id": conversation_id,
        "usage": {**usage, "estimated_input_tokens": input_tokens},
    }
    set_cached_answer(user_id, question, result)
    return result


async def ask_question_stream(
    user_id: int, question: str
) -> AsyncGenerator[str, None]:
    reranked, _ = retrieve_context(user_id, question)
    context = _build_context(reranked)
    memory = _get_memory(user_id)

    prompt = _build_rag_prompt(question, context)

    llm = get_llm(streaming=True)
    full_answer = ""
    async for chunk in llm.astream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_answer += token
            yield f"data: {json.dumps({'token': token})}\n\n"

    memory.save_context({"question": question}, {"answer": full_answer})
    sources = _format_sources(reranked)
    yield f"data: {json.dumps({'done': True, 'sources': sources})}\n\n"


def index_documents(user_id: int, documents: List[Document]) -> int:
    store = get_vectorstore(user_id)
    return store.add_documents(documents)
