# VectorMind

VectorMind is a hybrid GraphRAG PDF Q&A system. Upload PDFs and it builds per-document registries (adaptive summaries, topics, keywords), a Neo4j knowledge graph of extracted entities and relations, and a FAISS child-level vector index. Retrieval is document-aware: a heuristic router selects lookup, synthesis, or cross-document modes; FAISS overfetch + CrossEncoder reranking finds precise excerpts; an LLM (Anthropic/Claude) generates structured, citation-backed answers with a mode-aware faithfulness check.

**Live app:** [vectormind.streamlit.app](https://vectormind.streamlit.app)

## Overview

- Per-document ingestion with parent (~2000 chars) and child (~400 chars) chunking.
- Adaptive, one-call document summaries used for routing and synthesis.
- Neo4j Aura knowledge graph storing `Document`, `Chunk`, and `Entity` nodes plus extracted relations.
- FAISS child-vector index (HuggingFace embeddings) with CrossEncoder reranking for high-precision excerpt selection.
- Retrieval modes: `lookup` (grounded excerpts + graph facts), `synthesis` (document summaries + best parents), and `cross_doc` (shared concepts across documents).
- Mode-aware faithfulness checks and a constrained regenerate flow for weaker-grounded synthesis/cross-doc answers.
- Index persistence: local `vectorstore.pkl` for FAISS + registry; Neo4j Aura persists the knowledge graph.

## How it works (brief)

1. Ingest: upload PDFs → extract text → split into parents and children (per-document) → extract triples and entities via the LLM → write entities/chunks to Neo4j and registry.
2. Index: child chunks are embedded and stored in FAISS; registry stores summaries, topics, keywords, and chunk counts per document.
3. Retrieve: a lightweight router chooses `lookup`, `synthesis`, or `cross_doc`. FAISS overfetches candidate children, a CrossEncoder reranker refines hits, and Neo4j provides graph neighborhood facts.
4. Answer: the LLM (Anthropic/Claude via LangChain) generates structured markdown answers with citations; a faithfulness-check step verifies grounding and may trigger constrained regeneration.

## Models & libraries

- LLM: Anthropic/Claude (via `langchain_anthropic`)
- Embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (via `langchain_community.embeddings`)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence_transformers.CrossEncoder`)
- Vector DB: FAISS (via `langchain_community.vectorstores`)
- Graph DB: Neo4j Aura (`neo4j` Python driver)
- PDF parsing: `PyPDF2`

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with these keys (examples):

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
CLAUDE_MODEL=claude-sonnet-4-6
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_aura_password

# Optional RAG tuning (defaults shown)
# PARENT_SIZE=2000
# PARENT_OVERLAP=200
# CHILD_SIZE=400
# CHILD_OVERLAP=50
# TOP_K_CHILDREN=8
# OVERFETCH_K=20
# MAX_PARENTS=6
# MAX_L2_DISTANCE=2.0
# SUMMARY_CHAR_BUDGET=12000
# SYNTHESIS_TOP_M=12
# CONTEXT_CHAR_BUDGET=14000
# INDEX_FILE=vectorstore.pkl
```

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) pointing at this repo, branch `main`, main file `app.py`, Python version **3.12**.
3. In the app's **Settings → Secrets**, add the same keys from `.env`.

## Persistence & limitations

- Neo4j Aura persists the knowledge graph across restarts; the FAISS index and in-repo registry are saved locally to `vectorstore.pkl` and must be reloaded on startup (or re-processed if missing).
- Summaries and shared-concept lists are model-derived and therefore weaker grounding than direct context excerpts — the faithfulness check aims to surface these risks. When in doubt, inspect `Sources` and `Document summaries` in the UI.

## Architecture (short)

Ingest → parent/child chunking → entity extraction → Neo4j registry + FAISS index → router → FAISS overfetch → CrossEncoder rerank → LLM answer + faithfulness check.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, chat, render sources/summaries/graph facts |
| `chatbot_core.py` | Document-aware hybrid GraphRAG pipeline and Neo4j/FAISS integration |
| `requirements.txt` | Python dependencies |

## License

MIT License
