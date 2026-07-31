import streamlit as st
from chatbot_core import chatbot

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("PDF Chatbot 🤖")


def main():
    if "conversation_initialized" not in st.session_state:
        st.session_state.conversation_initialized = False
        if chatbot.initialize_from_saved_vectorstore():
            st.session_state.conversation_initialized = True

    st.header("Chat with your documents")

    status = chatbot.get_status()
    if not status["conversation_chain_initialized"]:
        st.info("👆 Please upload and process documents first")
    else:
        st.success("✅ Ready to answer questions!")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        if not status["conversation_chain_initialized"]:
            st.error("Please process documents first before asking questions.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = chatbot.ask_question(user_question)
                    answer = response.get("answer")

                    if answer and answer.strip():
                        st.write("**Answer:**", answer)
                    else:
                        st.error("Received empty answer. Please try a different question.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with st.sidebar:
        st.subheader("Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    try:
                        result = chatbot.process_documents(pdf_docs)
                        st.success(f"✅ Processed {result['chunks_created']} chunks from {len(pdf_docs)} documents")
                        st.session_state.conversation_initialized = True
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please upload PDF files first")

        st.subheader("System Status")
        status = chatbot.get_status()
        for key, value in status.items():
            icon = "✅" if value else "❌"
            st.write(f"{icon} {key}: {value}")


if __name__ == "__main__":
    main()
