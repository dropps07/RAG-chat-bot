# RAG Chatbot — Presentation Content (7 slides)

## Slide 1 — Problem

- We have notes, PDFs and guides, but finding one fact inside them is slow.
- Asking a chatbot directly is risky: it answers from training data and can
  invent facts that sound right.
- For college work we need answers we can check — tied to a real page in a
  real document.
- This project fixes that: a chatbot that answers only from documents you
  upload, and shows its sources.

## Slide 2 — What RAG is

- RAG = Retrieval-Augmented Generation.
- Two steps: first **retrieve** the most relevant passages from your
  documents, then **generate** an answer from those passages only.
- The LLM never answers from memory alone — the documents are its only
  source of facts.
- If the documents don't contain the answer, the bot says so instead of
  guessing.
- Result: answers you can verify, because every answer lists its sources.

## Slide 3 — Architecture

- Upload (PDF/TXT) → Extract text per page → Split into overlapping chunks
  → Embed chunks locally → Store in FAISS → Retrieve top 4 chunks per
  question → Gemini answers from those chunks only → Answer + sources.
- Modules: `ingestion.py` (load + split), `vectorstore.py` (embeddings +
  FAISS), `retrieval.py` (top-k + sources), `generation.py` (Gemini +
  strict prompt), `app.py` (Streamlit UI only).
- The FAISS index is saved to disk (`faiss_index/`), so the app can reuse
  it after a restart.

## Slide 4 — Tech stack

- **Python + Streamlit** — the chat interface, file upload and session state.
- **LangChain** — document loaders (PyPDFLoader, TextLoader) and the
  RecursiveCharacterTextSplitter (1000 chars, 200 overlap).
- **FAISS (faiss-cpu)** — the vector store; fast similarity search over
  chunk embeddings.
- **sentence-transformers, all-MiniLM-L6-v2** — local embeddings, no API key
  needed, 384-dimensional vectors.
- **Google Gemini (free tier)** — the LLM; model name is configurable in
  `.env` (default `gemini-3.5-flash-lite`).
- **python-dotenv + pytest** — config from `.env`, tests for chunking and
  retrieval.

## Slide 5 — Workflow (how a question is answered)

1. User uploads documents and clicks **Process documents** — chunks are
   embedded and stored in FAISS.
2. User asks a question in the chat box.
3. The question is embedded with the same model and FAISS returns the 4
   closest chunks.
4. The chunks are pasted into a strict prompt: answer only from this
   context, otherwise say "I don't know".
5. Gemini replies (temperature 0, so answers are factual and repeatable).
6. The answer appears with its sources: file name, page number and the
   retrieved text. Follow-ups work because the last 4 turns are included
   in the prompt.

## Slide 6 — Demo

- Sample document: `data/sample.txt` — a library visitor guide (opening
  hours, membership, fees, rules).
- Question 1: "What are the library's opening hours?" — answered from the
  document, source shown.
- Follow-up: "What about Saturdays?" — answered using the chat history.
- Unrelated question: "Who won the 2024 cricket World Cup?" — the bot says
  the documents don't contain the answer.
- Error paths shown: missing API key, empty/scanned PDF, unsupported file.
- Tests: `pytest` — chunking (sizes, overlap, page numbers) and retrieval
  (relevant chunk ranks first, sources carry file + page).

## Slide 7 — Limitations and future scope

- **Limitations:** needs a Gemini API key; free-tier rate limits apply;
  scanned PDFs can't be read (no OCR); answers depend on the top 4 chunks,
  so a fact buried deep may be missed; one user and one index at a time;
  CPU embeddings are slow on very large document sets.
- **Future scope:** OCR for scanned PDFs; a persistent vector database
  instead of a local FAISS file; hybrid search (keyword + vector) and
  re-ranking; per-document chat and multi-user support; a "confidence"
  indicator next to each answer.
