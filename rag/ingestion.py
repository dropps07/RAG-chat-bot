"""Load PDF/TXT files and split them into overlapping chunks.

Each chunk keeps its source file name and page number (PDFs) in its
metadata, so answers can cite exactly where they came from.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


class IngestionError(ValueError):
    """Raised when a file cannot be turned into text."""


def load_file(path):
    """Load one file and return a list of Documents (one per PDF page)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(
            f"Unsupported file type '{suffix}'. "
            "Only PDF (.pdf) and text (.txt) files are supported."
        )
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        raise IngestionError(
            f"Could not read '{path.name}': {e}"
        ) from e

    # PyPDFLoader gives one Document per page, with metadata "page" (0-based)
    # and "source" (file path). TextLoader gives one Document for the file.
    for doc in docs:
        doc.metadata["source"] = path.name

    # Drop pages with no text (e.g. scanned pages) but remember we saw them.
    text_docs = [d for d in docs if d.page_content.strip()]
    if not text_docs:
        raise IngestionError(
            f"No text could be read from '{path.name}'. "
            "The file may be empty or a scanned PDF (image only, no text layer). "
            "Try a PDF with selectable text or a .txt file."
        )
    return text_docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into overlapping chunks.

    Overlap keeps a sentence that straddles a boundary inside both chunks,
    so the retriever still finds it.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # Store page numbers 1-based, so the UI can say "page 3" like a human.
    for chunk in chunks:
        if "page" in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata["page"] + 1
    return chunks


def ingest_files(paths, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Load and split many files.

    Returns (chunks, skipped) where skipped is a list of
    (file name, reason) for files that could not be read.
    """
    chunks, skipped = [], []
    for path in paths:
        try:
            docs = load_file(path)
            chunks.extend(split_documents(docs, chunk_size, chunk_overlap))
        except IngestionError as e:
            skipped.append((Path(path).name, str(e)))
    if not chunks:
        raise IngestionError(
            "No usable text was found in any of the uploaded files. "
            + " ".join(reason for _, reason in skipped)
        )
    return chunks, skipped
