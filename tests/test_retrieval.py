"""Tests for retrieval: relevant chunks rank first, sources carry file+page."""
import pytest
from langchain_core.documents import Document

from rag.retrieval import format_sources, retrieve
from rag.vectorstore import build_vectorstore

DOCS = [
    Document(
        page_content="The Eiffel Tower is in Paris, France. It was completed in 1889.",
        metadata={"source": "paris.pdf", "page": 1},
    ),
    Document(
        page_content="The Great Wall of China stretches over 21,000 kilometres.",
        metadata={"source": "china.pdf", "page": 2},
    ),
    Document(
        page_content="Photosynthesis lets plants turn sunlight into energy.",
        metadata={"source": "bio.txt"},
    ),
]


@pytest.fixture(scope="module")
def store():
    # Downloads the embedding model on first use, then embeds 3 short docs.
    return build_vectorstore(DOCS)


def test_relevant_chunk_ranks_first(store):
    results = retrieve(store, "Where is the Eiffel Tower?", top_k=3)
    assert results
    top_doc, _score = results[0]
    assert "Eiffel Tower" in top_doc.page_content


def test_sources_carry_file_and_page(store):
    results = retrieve(store, "Where is the Eiffel Tower?", top_k=1)
    sources = format_sources(results)
    assert sources[0]["file"] == "paris.pdf"
    assert sources[0]["page"] == 1
    assert "Eiffel Tower" in sources[0]["text"]


def test_unrelated_question_scores_worse(store):
    # FAISS returns L2 distance: smaller means more similar.
    related = retrieve(store, "Where is the Eiffel Tower?", top_k=1)
    unrelated = retrieve(store, "How do I bake sourdough bread?", top_k=1)
    assert related[0][1] < unrelated[0][1]
