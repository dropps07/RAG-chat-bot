"""Streamlit UI for the RAG chatbot.

This file only handles the interface: uploads, chat display and buttons.
All RAG work lives in the rag/ package.
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

# Streamlit Cloud secrets (if set) take priority over .env — read them
# before importing rag.config, which loads .env at import time.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    if "GEMINI_MODEL" in st.secrets:
        os.environ["GEMINI_MODEL"] = st.secrets["GEMINI_MODEL"]
except Exception:
    pass  # No secrets file: .env is used instead.

from rag import config
from rag.generation import GenerationError, answer_question
from rag.ingestion import IngestionError, ingest_files
from rag.retrieval import format_context, format_sources, retrieve
from rag.vectorstore import build_vectorstore, load_vectorstore, save_vectorstore

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")
st.caption("Ask questions about your own documents. Answers come only from what you upload.")

# --- session state: chat history, the search index, and what was indexed ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources"} dicts
if "history" not in st.session_state:
    st.session_state.history = []  # (question, answer) pairs for follow-ups
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# --- sidebar: upload and process documents ---
with st.sidebar:
    st.header("1. Add documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Process documents", type="primary"):
        if not uploaded:
            st.error("Upload at least one PDF or TXT file first.")
        else:
            with st.spinner("Reading files and building the search index..."):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        paths = []
                        for f in uploaded:
                            p = Path(tmp) / f.name
                            p.write_bytes(f.getbuffer())
                            paths.append(p)
                        chunks, skipped = ingest_files(paths)
                    vectorstore = build_vectorstore(chunks)
                    save_vectorstore(vectorstore)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.indexed_files = [f.name for f in uploaded]
                    st.success(
                        f"Indexed {len(chunks)} chunks from {len(uploaded)} file(s)."
                    )
                    for name, reason in skipped:
                        st.warning(f"Skipped {name}: {reason}")
                except IngestionError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Could not build the index. Details: {e}")

    if st.session_state.indexed_files:
        st.write("Indexed files:")
        for name in st.session_state.indexed_files:
            st.write(f"- {name}")

    st.header("2. Chat")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# Warn early if there is no API key, so the user is not surprised later.
if not config.GEMINI_API_KEY:
    st.sidebar.warning(
        "No Gemini API key found. Add GEMINI_API_KEY to your .env file "
        "(copy .env.example first) or to Streamlit secrets."
    )

# Reuse a saved index from a previous run, if one exists.
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vectorstore()

# --- show the chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    label = f"**{s['file']}**"
                    if s.get("page"):
                        label += f", page {s['page']}"
                    st.markdown(label)
                    st.markdown(f"> {s['text'][:600]}")


def _show_sources(sources):
    """Render the source list under an answer."""
    if not sources:
        return
    with st.expander("Sources"):
        for s in sources:
            label = f"**{s['file']}**"
            if s.get("page"):
                label += f", page {s['page']}"
            st.markdown(label)
            st.markdown(f"> {s['text'][:600]}")


# --- ask a question ---
if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            answer = "Upload and process a document first, then ask a question."
            sources = []
        elif not config.GEMINI_API_KEY:
            answer = (
                "No Gemini API key found. Add GEMINI_API_KEY to your .env file "
                "(copy .env.example first) or to Streamlit secrets, then restart the app."
            )
            sources = []
        else:
            with st.spinner("Searching your documents..."):
                try:
                    results = retrieve(st.session_state.vectorstore, question)
                    sources = format_sources(results)
                    context = format_context(results)
                    answer = answer_question(
                        question, context, history=st.session_state.history
                    )
                except GenerationError as e:
                    answer = str(e)
                    sources = []
                except Exception as e:
                    answer = f"Something went wrong while answering. Details: {e}"
                    sources = []
        st.markdown(answer)
        _show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.history.append((question, answer))
