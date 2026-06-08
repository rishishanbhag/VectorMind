from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings
from app.core.rag import chunk_documents, rerank_documents


def test_recursive_chunking():
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    text = "Hello world. " * 200
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= settings.chunk_size + 50 for c in chunks)


def test_chunk_documents_with_metadata():
    pages = [
        ("This is page one content about machine learning.", {"page": 1, "filename": "doc.pdf"}),
        ("This is page two content about neural networks.", {"page": 2, "filename": "doc.pdf"}),
    ]
    docs = chunk_documents(pages, "doc.pdf")
    assert len(docs) >= 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata["filename"] == "doc.pdf"
    assert "page" in docs[0].metadata
    assert "chunk_index" in docs[0].metadata


def test_rerank_documents_fallback():
    docs = [
        Document(page_content="machine learning basics", metadata={"page": 1}),
        Document(page_content="cooking recipes", metadata={"page": 2}),
    ]
    result = rerank_documents("machine learning", docs, top_k=1)
    assert len(result) == 1
