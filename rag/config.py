"""Settings for the RAG chatbot.

Everything the user may want to change lives here or in .env.
No API key is ever written in code — it comes from .env or Streamlit secrets.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (if it exists).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- LLM (Google Gemini, free tier) ---
# Default model: gemini-3.5-flash-lite — the fast free-tier model. The
# full Flash model (gemini-3.8-flash) is slow and often overloaded
# (same lesson as Forkcast: lite answers in seconds, flash took minutes).
# Other free-tier options: gemini-2.5-flash, gemini-2.5-flash-lite,
# gemini-3.5-flash, gemini-3.8-flash.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Embeddings (local, runs on this machine, no key needed) ---
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))

# --- Storage ---
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
