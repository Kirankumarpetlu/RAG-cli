from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_cli import answer


def test_weak_evidence_never_calls_chat(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda *args: [])

    def unexpected_chat(**kwargs):
        raise AssertionError("Chat should not be called")

    monkeypatch.setattr(answer, "ChatGroq", unexpected_chat)
    result = answer.answer_question("unknown", None, {"top_k": 4, "min_score": 0.5})
    assert result == {"answer": answer.REFUSAL, "sources": []}


def test_context_and_valid_answer(monkeypatch):
    document = Document(
        page_content="Refunds require a receipt.",
        metadata={"source": "policy.pdf", "page": 0, "chunk_id": "7"},
    )
    monkeypatch.setattr(answer, "retrieve", lambda *args: [(document, 0.9)])

    def respond(prompt):
        messages = prompt.to_messages()
        assert "untrusted" in messages[0].content
        assert "answer string itself MUST contain inline [ID]" in messages[0].content
        assert "Refunds require a receipt." in messages[1].content
        assert "page=1" in messages[1].content
        return {"answer": "A receipt is required. [1]", "citations": [1], "abstained": False}

    monkeypatch.setattr(answer, "ChatGroq", lambda **kwargs: SimpleNamespace(
        with_structured_output=lambda *args, **kwargs: RunnableLambda(respond)
    ))
    result = answer.answer_question("What is required?", None, {
        "top_k": 4, "min_score": 0.5, "chat_model": "test-model", "api_key": "fake",
    })
    assert result["answer"] == "A receipt is required. [1]"
    assert result["sources"][0]["label"] == "policy.pdf; page=1; chunk=7"


@pytest.mark.parametrize("result", [
    {"answer": "Invented. [99]", "citations": [99], "abstained": False},
    {"answer": "Uncited claim.", "citations": [], "abstained": False},
    {"answer": "Mismatch. [1]", "citations": [2], "abstained": False},
    {"answer": "Boolean ID. [1]", "citations": [True], "abstained": False},
    {"answer": "Unsupported.", "citations": [], "abstained": True},
    {"answer": 123, "citations": [1], "abstained": False},
    None,
])
def test_bad_response_or_abstention_refuses(result):
    assert answer.validate_answer(result, {1: {"id": 1}}) == {
        "answer": answer.REFUSAL, "sources": [],
    }


def test_provider_failure_stays_an_error(monkeypatch):
    document = Document(page_content="Evidence", metadata={"source": "a.md", "chunk_id": "0"})
    monkeypatch.setattr(answer, "retrieve", lambda *args: [(document, 0.9)])

    def fail(**kwargs):
        raise ConnectionError("simulated provider failure")

    monkeypatch.setattr(answer, "ChatGroq", fail)
    with pytest.raises(ConnectionError):
        answer.answer_question("Question", None, {
            "top_k": 4, "min_score": 0.5, "chat_model": "test-model", "api_key": "fake",
        })


def test_blank_question_is_rejected():
    with pytest.raises(ValueError, match="blank"):
        answer.answer_question("  ", None, {"top_k": 4, "min_score": 0.5})
