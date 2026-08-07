import os

import streamlit as st

_SECRET_KEYS = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_MODEL",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
)
try:
    for key in _SECRET_KEYS:
        if key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
except Exception:
    pass

from chatbot_core import chatbot

st.set_page_config(page_title="VectorMind", page_icon="📄")
st.title("VectorMind")


def _render_response(response: dict):
    answer = response.get("answer")
    if not (answer and str(answer).strip()):
        st.error("Received empty answer. Please try a different question.")
        return

    route = response.get("route", "lookup")
    st.caption(f"Route: **{route}**")
    candidate_docs = response.get("candidate_docs") or []
    if candidate_docs and chatbot.documents:
        names = [
            (chatbot.documents.get(d) or {}).get("doc_name", d) for d in candidate_docs
        ]
        st.caption("Candidate docs: " + ", ".join(names))

    shared = response.get("shared_concepts") or []
    if shared:
        st.caption(f"Shared concepts used: **{len(shared)}**")
        with st.expander("Shared concepts (Neo4j / registry)"):
            for c in shared:
                docs = ", ".join(c.get("doc_names") or c.get("docs") or [])
                st.write(f"- **{c.get('name')}** — {docs}")

    faith = response.get("faithfulness") or {}
    if faith.get("supported") is False and faith.get("caution"):
        st.warning(
            "Some claims may be weakly grounded — faithfulness still flagged issues "
            "after a constrained rewrite. Prefer citations in Sources."
        )
    elif faith.get("supported") is False and route in ("synthesis", "cross_doc"):
        st.warning(
            "Faithfulness check flagged weak grounding. Review Sources before trusting details."
        )

    st.markdown(answer)

    sources = response.get("sources") or []
    if sources:
        with st.expander("Sources"):
            for src in sources:
                label = src.get("doc_name") or src.get("id")
                concept = src.get("concept")
                header = f"**[{label}]** `{src.get('id', '')}`"
                if concept:
                    header += f" — concept: *{concept}*"
                st.markdown(header)
                excerpt = src.get("text", "")
                if len(excerpt) > 500:
                    excerpt = excerpt[:500] + "…"
                st.write(excerpt)

    summaries = response.get("summaries") or []
    if summaries:
        with st.expander("Document summaries used"):
            for s in summaries:
                st.markdown(f"**{s.get('doc_name', s.get('doc_id'))}**")
                st.write(s.get("summary") or "(empty)")

    graph_facts = response.get("graph_facts") or []
    if graph_facts:
        with st.expander("Graph context"):
            for fact in graph_facts:
                st.write(f"- {fact}")

    if faith:
        supported = faith.get("supported")
        reason = faith.get("reason", "")
        label = "supported" if supported else "not supported"
        st.caption(f"Faithfulness: {label}" + (f" — {reason}" if reason else ""))


def main():
    if "conversation_initialized" not in st.session_state:
        st.session_state.conversation_initialized = False
        if chatbot.initialize_from_saved_vectorstore():
            st.session_state.conversation_initialized = True

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    st.header("Chat with your documents")

    status = chatbot.get_status()
    if not status["neo4j_configured"]:
        st.warning(
            "Neo4j Aura is not configured. Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD "
            "in `.env` or Streamlit secrets before processing documents."
        )

    if not status["conversation_chain_initialized"]:
        st.info("Upload and process documents first")
    else:
        n_docs = status.get("documents_indexed", 0)
        st.success(f"Ready to answer questions ({n_docs} document(s) indexed)")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if not status["conversation_chain_initialized"]:
            st.error("Please process documents first before asking questions.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    response = chatbot.ask_question(user_question)
                    st.session_state.last_response = response
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.last_response:
        _render_response(st.session_state.last_response)

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type="pdf",
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner(
                    "Per-doc chunking, summaries, embeddings, and knowledge graph..."
                ):
                    try:
                        result = chatbot.process_documents(pdf_docs)
                        st.success(
                            f"Processed {result.get('files_processed', len(pdf_docs))} file(s): "
                            f"{result['chunks_created']} child chunks "
                            f"({result.get('parents_created', '?')} parents)"
                        )
                        st.session_state.conversation_initialized = True
                        st.session_state.last_response = None
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please upload PDF files first")

        if chatbot.documents:
            st.subheader("Document registry")
            for meta in chatbot.documents.values():
                st.write(
                    f"- **{meta.get('doc_name')}** "
                    f"({meta.get('n_parents')} parents / {meta.get('n_children')} children)"
                )

        st.subheader("System Status")
        status = chatbot.get_status()
        for key, value in status.items():
            icon = "OK" if value else "—"
            st.write(f"{icon} | {key}: {value}")


if __name__ == "__main__":
    main()
