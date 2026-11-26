# RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot. Upload your own PDF or
TXT files, ask questions about them, and get answers **only from those
documents** — with the source file, page number and text shown under every
answer. If the documents don't contain the answer, it says so instead of
guessing.

Built with Python, Streamlit, LangChain, FAISS, local sentence-transformers
embeddings and Google Gemini (free tier).

## Features

- Upload one or more PDF or TXT files, then process them with one click.
- Text is extracted per page (so page numbers survive), split into overlapping
  chunks (1000 characters, 200 overlap), embedded locally and stored in FAISS.
- Every question retrieves the 4 most relevant chunks and the LLM answers
  only from them.
- Sources under each answer: file name, page number (PDFs) and the retrieved
  text.
- Chat interface with history, so follow-up questions work, plus a
  **Clear chat** button.
- Clear error messages for a missing API key, empty or scanned PDFs,
  unsupported files and Gemini API failures — no raw tracebacks.

## Architecture

```text
                ┌─────────────────────────────────────────────┐
                │  app.py  (Streamlit UI: upload + chat)      │
                └──────┬──────────────────────────┬───────────┘
                       │                          │
            ┌──────────▼──────────┐      ┌────────▼────────┐
            │ ingestion.py        │      │ generation.py   │
            │ PyPDFLoader /       │      │ ChatGoogleGen-  │
            │ TextLoader → split  │      │ erativeAI       │
            │ into chunks         │      │ (Gemini,        │
            └──────────┬──────────┘      │ temp = 0)       │
                       │                 └────────▲────────┘
            ┌──────────▼──────────┐                │
            │ vectorstore.py      │      ┌────────┴────────┐
            │ HuggingFaceEmbed-   │      │ retrieval.py    │
            │ dings (all-MiniLM-  │      │ top-k chunks +  │
            │ L6-v2, local) →     │      │ sources         │
            │ FAISS index         │      └─────────────────┘
            │ (faiss_index/)      │
            └─────────────────────┘
```

How a question is answered:

1. The question is embedded with the same local model as the documents.
2. FAISS returns the 4 chunks closest to it in vector space.
3. Those chunks are pasted into a strict prompt: "answer only from this
   context; if it is not in the context, say you don't know."
4. Gemini answers, and the app shows the sources underneath.

## Project structure

```text
rag-chatbot/
├── app.py                  # Streamlit UI only (no RAG logic here)
├── rag/
│   ├── __init__.py
│   ├── config.py           # loads .env: model names, chunk size, top-k
│   ├── ingestion.py        # load PDF/TXT, extract text, split into chunks
│   ├── vectorstore.py      # local embeddings + FAISS build/save/load
│   ├── retrieval.py        # retrieve top-k chunks, format sources
│   └── generation.py       # Gemini call, strict "answer only from context" prompt
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
├── data/
│   └── sample.txt          # sample document for testing
├── .env.example
├── requirements.txt
├── README.md
├── PRESENTATION.md
└── VIVA.md
```

## Setup and run

```bash
git clone <your-repo-url>
cd rag-chatbot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and paste your key:
# GEMINI_API_KEY=your_key_here
# Get a free key at https://aistudio.google.com/apikey

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

### Try it with the sample document

1. Upload `data/sample.txt` and click **Process documents**.
2. Ask: "What are the library's opening hours?" — it should answer from the
   document, with the source shown.
3. Follow up: "What about Saturdays?" — it should use the chat history.
4. Ask something unrelated, e.g. "Who won the 2024 cricket World Cup?" — it
   should say the documents don't contain the answer.

## Deployment — Streamlit Community Cloud (free)

1. Push this repo to GitHub (do **not** commit `.env`).
2. Go to https://share.streamlit.io and click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`, and click **Deploy**.
4. In the app's settings, open **Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```
5. The first question takes a little longer: the embedding model
   (all-MiniLM-L6-v2, about 90 MB) downloads once, then it is cached.

Note: the FAISS index is rebuilt every time you upload documents, and it is
stored on the app's own disk, so each deploy starts with an empty index.

## Configuration (.env)

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_API_KEY` | _(empty)_ | Your free Gemini API key |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model. Free-tier alternatives: `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.5-flash` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Characters shared between neighbouring chunks |
| `TOP_K` | `4` | Chunks retrieved per question |

## Tests

```bash
pytest -v
```

`test_chunking.py` checks chunk sizes, overlap and that page numbers survive
splitting. `test_retrieval.py` checks that the relevant chunk ranks first,
that sources carry the file name and page, and that an unrelated question
scores worse than a related one.

## Limitations

- Needs a Gemini API key; without one, the app can index documents but
  cannot generate answers (it tells you this clearly).
- The free tier has rate limits; too many questions too fast may fail with
  an API error, which the app reports instead of crashing.
- Scanned PDFs (images with no text layer) cannot be read — use a PDF with
  selectable text, or a TXT file. OCR is not included.
- Answers are only as good as the retrieved chunks. If the right chunk is
  not in the top 4, the answer will be "I don't know" even if the document
  contains it somewhere.
- One user, one index at a time: uploading new documents replaces the
  current index, and chat history lives only in the browser session.
- Embeddings run on CPU by default, so indexing a very large set of
  documents takes a while.
