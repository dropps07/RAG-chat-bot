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

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")

# Small visual polish, Claude + ChatGPT style: warm paper background, coral
# accent, serif headings, and no Deploy button or Streamlit footer, so the
# app reads like a finished product, not a demo.
st.markdown(
    """
    <style>
    .stAppDeployButton {display: none;}
    footer {visibility: hidden;}
    h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Chatbot")
st.markdown(
    "Upload a PDF or text file, then ask questions about it. "
    "Every answer comes only from your documents, and the source is shown "
    "under each answer."
)

# --- session state: chat history, the search index, and what was indexed ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources"} dicts
if "history" not in st.session_state:
    st.session_state.history = []  # (question, answer) pairs for follow-ups
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# Reuse a saved index from a previous run, if one exists.
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vectorstore()

indexed = st.session_state.vectorstore is not None


def _show_sources(sources):
    """Render the source list under an answer."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            label = f"**{s['file']}**"
            if s.get("page"):
                label += f", page {s['page']}"
            st.markdown(label)
            st.markdown(f"> {s['text'][:600]}")


# --- sidebar: documents and chat controls ---
with st.sidebar:
    st.header("Your documents")
    if not config.GEMINI_API_KEY:
        st.warning(
            "No Gemini API key found. Add GEMINI_API_KEY in Streamlit secrets "
            "(or in your .env file) to get answers."
        )
    uploaded = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Add one or more documents, then click Process documents.",
    )
    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded:
            st.error("Upload at least one PDF or TXT file first.")
        else:
            with st.spinner("Reading your files and building the search index..."):
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
                    indexed = True
                    st.success(
                        f"Ready — indexed {len(chunks)} chunks from "
                        f"{len(uploaded)} file(s). Ask a question below."
                    )
                    for name, reason in skipped:
                        st.warning(f"Skipped {name}: {reason}")
                except IngestionError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Could not build the index. Details: {e}")

    if st.session_state.indexed_files:
        st.subheader("Indexed")
        for name in st.session_state.indexed_files:
            st.write(f"✓ {name}")

    st.divider()
    st.header("Chat")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    with st.expander("How it works"):
        st.markdown(
            "1. **Upload** a PDF or TXT file above.\n"
            "2. Click **Process documents** to index it.\n"
            "3. **Ask** anything about it below.\n\n"
            "The app searches your documents and asks Gemini to answer "
            "only from what it finds. If the answer is not in your "
            "documents, it says so instead of guessing."
        )

# --- welcome message: shown until the first question is asked ---
if not st.session_state.messages:
    if not indexed:
        st.info(
            "Start by uploading a document in the sidebar, then click "
            "**Process documents**. Once it is indexed, ask your first "
            "question below."
        )
    else:
        st.info(
            "Your documents are indexed. Ask anything about them below — "
            "for example, “What is this document about?” or "
            "“Summarise the key points.”"
        )

# --- show the chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _show_sources(msg["sources"])

# --- ask a question ---
# The input stays disabled until a document is indexed, so the flow is
# obvious: upload first, then ask.
placeholder = (
    "Ask a question about your documents"
    if indexed
    else "Upload and process a document first"
)
if question := st.chat_input(placeholder, disabled=not indexed):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            answer = "Upload and process a document first, then ask a question."
            sources = []
        elif not config.GEMINI_API_KEY:
            answer = (
                "No Gemini API key found. Add GEMINI_API_KEY in Streamlit secrets "
                "(or in your .env file), then restart the app."
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
