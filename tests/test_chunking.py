"""Tests for chunking: sizes, overlap and page numbers are kept."""
from langchain_core.documents import Document

from rag.ingestion import split_documents


def test_chunks_do_not_exceed_size():
    text = " ".join(f"word{i}" for i in range(500))
    docs = [Document(page_content=text, metadata={"source": "a.txt"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert chunks
    assert all(len(c.page_content) <= 100 for c in chunks)


def test_overlap_keeps_shared_text():
    # Numbered sentences: text near a chunk boundary must appear in both
    # neighbouring chunks — that is what chunk_overlap is for.
    text = " ".join(f"Sentence {i}." for i in range(100))
    docs = [Document(page_content=text, metadata={"source": "a.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=100)
    assert len(chunks) >= 2
    # The tail of one chunk (minus a small margin for stripped whitespace)
    # appears inside the next chunk.
    shared = any(
        chunks[i].page_content[-50:-20] in chunks[i + 1].page_content
        for i in range(len(chunks) - 1)
    )
    assert shared


def test_page_numbers_become_one_based():
    # PyPDFLoader numbers pages from 0; the UI shows them from 1.
    text = "Some content. " * 100
    docs = [Document(page_content=text, metadata={"source": "a.pdf", "page": 0})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert chunks
    assert all(c.metadata["page"] == 1 for c in chunks)


def test_source_is_kept_on_every_chunk():
    text = "Some content. " * 100
    docs = [Document(page_content=text, metadata={"source": "notes.pdf", "page": 0})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)
    assert chunks
    assert all(c.metadata["source"] == "notes.pdf" for c in chunks)
