
"""
frontend/app.py
------------------
Streamlit frontend — poore backend pipeline (ingestion + hybrid retrieval
+ agentic routing + memory) ko ek simple chat UI mein wrap karta hai.

Kaise chalayein:
    streamlit run frontend/app.py

(Project ke root folder se chalana, "frontend" folder ke andar se nahi)
"""

import sys
import os

# Root folder ko path mein add karo taaki "app.xxx" imports kaam karein
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


# --- Session state setup (Streamlit re-runs poori script har interaction pe,
# isliye state ko yahan persist karna zaroori hai) ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "keyword_search" not in st.session_state:
    st.session_state.keyword_search = None
if "conversation_store" not in st.session_state:
    st.session_state.conversation_store = ConversationStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # UI mein dikhane ke liye (role, content, meta)
if "document_ready" not in st.session_state:
    st.session_state.document_ready = False


def process_pdf(uploaded_file):
    """
    Uploaded PDF ko poori ingestion pipeline se guzarta hai:
    parse -> chunk -> embed -> vector store + keyword index bana do.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("PDF padh rahe hain..."):
        pages_data = extract_text_from_pdf(tmp_path)

    with st.spinner("Text ko chunks mein toda ja raha hai..."):
        chunks = chunk_text(pages_data)

    with st.spinner("Embeddings bana rahe hain (pehli baar thoda time lagega)..."):
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
    uploaded_file = st.file_uploader("Apna PDF upload karo", type=["pdf"])

    if uploaded_file is not None and st.button("Process Document", type="primary"):
        num_chunks = process_pdf(uploaded_file)
        st.success(f"Document ready! {num_chunks} chunks banaye gaye.")

    st.divider()

    if st.session_state.document_ready:
        st.info("✅ Document loaded — ab document-based sawaal pooch sakti ho.")
    else:
        st.warning("⚠️ Abhi tak koi document upload nahi hua. Bina document ke bhi sawaal pooch sakti ho — system web search use karega.")

    st.divider()
    if st.button("🗑️ Chat History Clear Karo"):
        st.session_state.chat_history = []
        st.session_state.conversation_store.clear_session("streamlit_session")
        st.rerun()


# --- Main chat interface ---
st.title("🤖 Agentic RAG Assistant")
st.caption("Document se ya live web search se — jahan se sahi jawab milega, wahan se laata hai.")

# Purani chat history dikhao
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            badge = "📄 Document" if meta["route"] == "document" else "🌐 Web Search"
            corrected_note = " · ✏️ Self-corrected" if meta.get("was_corrected") else ""
            st.caption(f"{badge}{corrected_note} | Sources: {', '.join(meta['sources'])}")

# Naya message input
user_question = st.chat_input("Kuch bhi pooch lo...")

if user_question:
    # User ka message dikhao aur history mein daalo
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Agar document upload nahi hua, to ek dummy empty vector store/keyword
    # search bana do taaki agent web-only mode mein bhi chal sake
    vector_store = st.session_state.vector_store or VectorStore(dimension=384)
    keyword_search = st.session_state.keyword_search or KeywordSearch([])

    with st.chat_message("assistant"):
        with st.spinner("Soch rahe hain..."):
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