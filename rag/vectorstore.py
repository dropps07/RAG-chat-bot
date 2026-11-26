"""Local embeddings (sentence-transformers) + FAISS vector store.

Embeddings run on this machine, so no key or internet is needed for this
step (after the model downloads once). FAISS stores the vectors for
fast similarity search.
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import EMBEDDING_MODEL, FAISS_INDEX_DIR


def get_embeddings(model_name=EMBEDDING_MODEL):
    """Create the local embedding model (downloads on first use)."""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(chunks, model_name=EMBEDDING_MODEL):
    """Turn chunks into vectors and put them in a FAISS index."""
    embeddings = get_embeddings(model_name)
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore, index_dir=FAISS_INDEX_DIR):
    """Save the FAISS index to disk so the app can reuse it."""
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))


def load_vectorstore(model_name=EMBEDDING_MODEL, index_dir=FAISS_INDEX_DIR):
    """Load a saved FAISS index. Returns None if no index exists yet."""
    index_dir = Path(index_dir)
    if not (index_dir / "index.faiss").exists():
        return None
    embeddings = get_embeddings(model_name)
    # allow_dangerous_deserialization is needed because FAISS stores pickled
    # document data alongside the index; the file is ours, so it is safe.
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )
