
"""
frontend/app.py
------------------
Streamlit frontend — wraps the entire backend pipeline (ingestion + hybrid retrieval
+ agentic routing + memory) in a simple chat UI.

How to run:
    streamlit run frontend/app.py

(Run from the project root folder, not from inside the "frontend" folder)
"""

import sys
import os

# Add root folder to path so that "app.xxx" imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import VectorStore
from app.retrieval.keyword_search import KeywordSearch
from app.memory.conversation_store import ConversationStore
from app.agents.router_agent import ask


st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🤖", layout="centered")


# --- Session state setup (Streamlit re-runs the entire script on every interaction,
# so persisting state here is necessary) ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "keyword_search" not in st.session_state:
    st.session_state.keyword_search = None
if "conversation_store" not in st.session_state:
    st.session_state.conversation_store = ConversationStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # for displaying in the UI (role, content, meta)
if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


def process_pdf(uploaded_file):
    """
    Passes the uploaded PDF through the entire ingestion pipeline:
    parse -> chunk -> embed -> create vector store + keyword index.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Reading PDF..."):
        pages_data = extract_text_from_pdf(tmp_path)

    with st.spinner("Breaking text into chunks..."):
        chunks = chunk_text(pages_data)

    with st.spinner("Creating embeddings (first time may take a moment)..."):
        embeddings = embed_chunks(chunks)

    store = VectorStore(dimension=embeddings.shape[1])
    store.add_chunks(embeddings, chunks)

    keyword_search = KeywordSearch(chunks)

    st.session_state.vector_store = store
    st.session_state.keyword_search = keyword_search
    st.session_state.document_ready = True

    os.remove(tmp_path)

    return len(chunks)


# --- Sidebar: document upload ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if uploaded_file is not None and st.button("Process Document", type="primary"):
        num_chunks = process_pdf(uploaded_file)
        st.success(f"Document ready! {num_chunks} chunks created.")

    st.divider()

    if st.session_state.document_ready:
        st.info("✅ Document loaded — you can now ask document-based questions.")
    else:
        st.warning("⚠️ No document uploaded yet. You can still ask questions — the system will use web search.")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation_store.clear_session("streamlit_session")
        st.rerun()


# --- Main chat interface ---
st.title("🤖 Agentic RAG Assistant")
st.caption("From documents or live web search — it fetches the right answer from wherever it's available.")

# Show previous chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            badge = "📄 Document" if meta["route"] == "document" else "🌐 Web Search"
            corrected_note = " · ✏️ Self-corrected" if meta.get("was_corrected") else ""
            st.caption(f"{badge}{corrected_note} | Sources: {', '.join(meta['sources'])}")

# New message input
user_question = st.chat_input("Ask anything...")

if user_question:
    # Show the user's message and add to history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # If no document is uploaded, create a dummy empty vector store/keyword
    # search so the agent can still run in web-only mode
    vector_store = st.session_state.vector_store or VectorStore(dimension=384)
    keyword_search = st.session_state.keyword_search or KeywordSearch([])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(
                user_question,
                vector_store,
                keyword_search,
                st.session_state.conversation_store,
                session_id="streamlit_session"
            )

        st.write(result["answer"])

        badge = "📄 Document" if result["route_used"] == "document" else "🌐 Web Search"
        corrected_note = " · ✏️ Self-corrected" if result.get("was_corrected") else ""
        st.caption(f"{badge}{corrected_note} | Sources: {', '.join(result['sources']) if result['sources'] else 'N/A'}")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": {
            "route": result["route_used"],
            "sources": result["sources"],
            "was_corrected": result.get("was_corrected", False)
        }
    })