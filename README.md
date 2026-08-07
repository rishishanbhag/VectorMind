# VectorMind

VectorMind is a hybrid GraphRAG chatbot: upload PDFs, ask questions in natural language, and get answers grounded in your documents. It combines **document-aware** parent-child FAISS retrieval, a Document Registry, Neo4j Aura for candidate discovery + graph facts, CrossEncoder reranking, and Claude with citations plus a mode-aware faithfulness check.

**Live app:** [vectormind.streamlit.app](https://vectormind.streamlit.app)

## How it works

1. Upload one or more PDFs and click **Process Documents**.
2. Each PDF is ingested **separately** with a `doc_id`. Text is split into **parent** (~2000 chars) and **child** (~400 chars) chunks. Children are embedded into FAISS with `doc_id` metadata.
3. Claude builds an **adaptive document summary** from parent chunks (one call per doc: full text if small, representative sampled parents if large), extracts entities into **Neo4j Aura**, and stores a **Document Registry** (summary, topics, entities, keywords, chunk counts).
4. On a question, a heuristic **router** chooses:
   - **Lookup:** Neo4j/registry candidate docs → FAISS over-fetch → CrossEncoder rerank → parent expansion → graph facts → Claude.
   - **Synthesis** (e.g. “each PDF”, “compare all”): Stage 1 candidate docs from registry/Neo4j → Stage 2 best parent per candidate + summaries → token-budgeted Claude answer (no map-reduce).
5. Answers are structured markdown with document-name citations. A faithfulness check is **mode-aware** (summaries count as grounding for synthesis).

## Run locally

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

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

Create a free Neo4j AuraDB instance at [console.neo4j.io](https://console.neo4j.io), then copy the connection URI and password into `.env`.

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io) pointing at this repo, branch `main`, main file `app.py`, Python version **3.12**.
3. In the app's **Settings → Secrets**, add:

```toml
ANTHROPIC_API_KEY = "your_anthropic_api_key"
CLAUDE_MODEL = "claude-sonnet-4-6"
NEO4J_URI = "neo4j+s://xxxx.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_aura_password"
```

**Persistence notes:**
- The knowledge graph lives in Neo4j Aura and persists across app restarts.
- The FAISS index + Document Registry are in-memory / local pickle on the Streamlit host and do **not** persist across restarts — re-upload PDFs after a restart.
- Re-processing documents clears and rebuilds the Aura graph for a clean index.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, chat, persisted answer, route/sources/summaries |
| `chatbot_core.py` | Document-aware hybrid GraphRAG pipeline |
| `requirements.txt` | Python dependencies |

## License

MIT License
