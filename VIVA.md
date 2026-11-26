# RAG Chatbot — 10 Likely Viva Questions

## 1. What is RAG, and why not just ask the LLM directly?

RAG means Retrieval-Augmented Generation. Instead of answering from its
training data, the model first retrieves the most relevant passages from
documents you uploaded, then answers only from those. Asking the LLM
directly is risky because it can invent facts. RAG grounds every answer in
your own documents, and the app shows the sources so you can check.

## 2. What is chunking, and why do you use overlap?

Long documents don't fit into an embedding model in one piece, so we split
them into chunks of 1000 characters. Overlap of 200 characters means a
sentence that falls across a boundary still appears whole inside a chunk.
Without overlap, a fact split across two chunks could be missed. This is
done by `RecursiveCharacterTextSplitter` in `rag/ingestion.py`.

## 3. What are embeddings?

Embeddings turn text into a list of numbers (a vector) that captures its
meaning. Texts with similar meanings get vectors that are close together.
We use the local `all-MiniLM-L6-v2` model, which makes 384-dimensional
vectors on this machine — no API key or internet needed after the first
download.

## 4. What is FAISS and what does it store?

FAISS (Facebook AI Similarity Search) is a library for fast similarity
search over vectors. It stores one vector per chunk, plus the chunk's text
and metadata (file name, page number). When a question comes in, FAISS
returns the chunks whose vectors are closest to the question's vector.
The index is saved in `faiss_index/` by `rag/vectorstore.py`.

## 5. How does the bot know when it doesn't know the answer?

The prompt in `rag/generation.py` is strict: "answer only from the
context; if the context does not contain the answer, reply exactly: I
don't know — the uploaded documents don't contain an answer to that
question." Temperature is 0, so the model follows the instruction
deterministically instead of guessing. If no context is retrieved at all,
the code returns that message without calling the API.

## 6. How do follow-up questions work?

The app keeps a `history` list of (question, answer) pairs in Streamlit
session state. When answering, the last 4 turns are added to the prompt,
so the model understands words like "it" or "that" from the earlier
question. History never overrides the rule — the answer must still come
from the retrieved context.

## 7. Where is the API key, and why isn't it in the code?

The key lives in a `.env` file (loaded by `rag/config.py`) or in Streamlit
secrets on deployment. `.env` is in `.gitignore`, so it can never be
committed. Only `.env.example` — with empty placeholders — is in the repo.
If the key is missing, the app shows a clear error instead of crashing.

## 8. What happens if I upload a scanned PDF?

A scanned PDF is images, not text. `PyPDFLoader` finds no text on its
pages, so `rag/ingestion.py` raises an `IngestionError` and the app shows:
"No text could be read… the file may be empty or a scanned PDF." The fix
would be OCR, which is listed as future scope — it is not in this version.

## 9. How do sources work — how do you know the page number?

`PyPDFLoader` loads one Document per PDF page, with metadata `page`
(0-based) and `source` (file name). Splitting copies that metadata onto
every chunk, and `ingestion.py` converts the page to 1-based for display.
`rag/retrieval.py` formats each retrieved chunk as file name, page number
and text, and the UI shows them under the answer in a "Sources" section.

## 10. What are the limitations of this project?

It needs a Gemini API key and the free tier has rate limits. Answers depend
on the top 4 retrieved chunks — if the right passage is not retrieved, the
bot says "I don't know". Scanned PDFs are unsupported. The FAISS index is
local and single-user: uploading new documents replaces the index, and chat
history lives only in the browser session. Future work: OCR, a persistent
vector database, hybrid keyword + vector search, and re-ranking.
