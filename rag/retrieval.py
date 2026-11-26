"""Retrieve the most relevant chunks for a question and format sources."""
from rag.config import TOP_K


def retrieve(vectorstore, question, top_k=TOP_K):
    """Return the top-k chunks most similar to the question.

    FAISS ranks by vector similarity, so the chunks about the question
    come first. Returns a list of (Document, score) pairs; the score is
    an L2 distance, so smaller means more similar.
    """
    return vectorstore.similarity_search_with_score(question, k=top_k)


def format_sources(results):
    """Turn retrieved chunks into a short, citable source list."""
    sources = []
    for doc, score in results:
        meta = doc.metadata
        sources.append(
            {
                "file": meta.get("source", "unknown file"),
                "page": meta.get("page"),  # None for TXT files
                "text": doc.page_content.strip(),
                "score": round(float(score), 3),
            }
        )
    return sources


def format_context(results):
    """Join the retrieved chunks into one text block for the LLM prompt."""
    parts = []
    for i, (doc, _score) in enumerate(results, 1):
        meta = doc.metadata
        label = meta.get("source", "unknown file")
        if meta.get("page"):
            label += f", page {meta['page']}"
        parts.append(f"[Source {i}: {label}]\n{doc.page_content.strip()}")
    return "\n\n".join(parts)
