"""Ask Gemini to answer ONLY from the retrieved context.

The prompt is strict on purpose: if the documents do not contain the
answer, the model must say so instead of guessing.
"""
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import config

NOT_IN_DOCUMENTS = (
    "I don't know — the uploaded documents don't contain an answer to that question."
)

SYSTEM_PROMPT = """You answer questions using ONLY the context below, which comes
from documents the user uploaded. Rules:
- Answer only from the context. Do not use anything you know from elsewhere.
- If the context does not contain the answer, reply exactly: "I don't know — the uploaded documents don't contain an answer to that question."
- Keep the answer short and in plain language.
- If the question is a follow-up, use the earlier conversation to understand
  what "it", "that" or "they" refers to, but still answer only from the context.

Context:
{context}

Question: {question}
"""


class GenerationError(RuntimeError):
    """Raised when the Gemini API call fails."""


def get_llm(api_key=None, model=None):
    """Create the Gemini chat model. Raises if no API key is set."""
    api_key = api_key or config.GEMINI_API_KEY
    model = model or config.GEMINI_MODEL
    if not api_key:
        raise GenerationError(
            "No Gemini API key found. Add GEMINI_API_KEY to your .env file "
            "(copy .env.example first) or to Streamlit secrets, then restart the app."
        )
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


def answer_question(question, context, history=None, api_key=None, model=None):
    """Generate an answer from the retrieved context.

    history is a list of (question, answer) pairs for follow-up questions.
    Returns the answer text. Raises GenerationError on API failure.
    """
    if not context.strip():
        return NOT_IN_DOCUMENTS
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    if history:
        # Give the model the last few turns so follow-ups like "and what
        # about the second one?" make sense. History never overrides the
        # answer-only-from-context rule.
        past = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-4:])
        prompt = f"Earlier conversation:\n{past}\n\n{prompt}"
    llm = get_llm(api_key, model)
    try:
        reply = llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        raise GenerationError(
            "The Gemini API call failed. Check your internet connection and "
            f"that your API key and model ({model or config.GEMINI_MODEL}) are valid. "
            f"Details: {e}"
        ) from e
    return reply.content.strip()
