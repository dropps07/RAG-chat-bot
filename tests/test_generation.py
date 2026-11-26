"""Tests for answer generation, using a stub LLM (no API key needed)."""
from langchain_core.messages import AIMessage

from rag import generation


class _StubLLM:
    """Fake chat model returning a fixed reply content."""

    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return AIMessage(content=self._content)


def test_list_content_blocks_are_joined(monkeypatch):
    """Regression test (2026-11-26 live test): Gemini returned the answer
    as a list of content blocks and reply.content.strip() crashed."""
    monkeypatch.setattr(
        generation,
        "get_llm",
        lambda *args, **kwargs: _StubLLM(
            [{"type": "text", "text": "  Open 9 to 6. "}, {"type": "text", "text": "Closed Sundays."}]
        ),
    )
    answer = generation.answer_question(
        "What are the opening hours?", "some context", api_key="fake-key"
    )
    assert answer == "Open 9 to 6. Closed Sundays."


def test_string_content_still_works(monkeypatch):
    monkeypatch.setattr(
        generation, "get_llm", lambda *args, **kwargs: _StubLLM("  Plain answer. ")
    )
    answer = generation.answer_question(
        "What are the opening hours?", "some context", api_key="fake-key"
    )
    assert answer == "Plain answer."
