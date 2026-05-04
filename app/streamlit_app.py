import streamlit as st

from core.orchestrator import RAGOrchestrator

# ========= INIT =========
st.set_page_config(page_title="RAG Chatbot", layout="wide")

if "rag" not in st.session_state:
    st.session_state.rag = RAGOrchestrator()

if "user_id" not in st.session_state:
    st.session_state.user_id = "user_1"

if "messages" not in st.session_state:
    st.session_state.messages = []

rag = st.session_state.rag
user_id = st.session_state.user_id

# ========= SIDEBAR =========
with st.sidebar:
    st.title("📂 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if st.button("🚀 Ingest"):
        if uploaded_files:
            with st.spinner("Processing files..."):
                num_chunks = rag.ingest(uploaded_files, user_id)
                st.success(f"Indexed {num_chunks} chunks")
        else:
            st.warning("Please upload files first")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []

# ========= MAIN UI =========
st.title("🤖 RAG Chatbot")

# Hiển thị history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ========= INPUT =========
query = st.chat_input("Ask something...")

if query:
    # show user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = rag.query(query, user_id)
            except Exception as e:
                answer = f"Error: {e}"

            st.write(answer)

    # save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })