from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.models.generation import GeneratedAnswer
from app.models.retrieval import RetrievalResult, ScoredChunk
from config.loader import GuardrailsConfig


class FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query):
        return RetrievalResult(query=query, results=self._results)


class FakeGenerator:
    def __init__(self, answer):
        self._answer = answer

    async def generate(self, question, context):
        return self._answer


def _chunk(chunk_id="c1", fused=0.03):
    return ScoredChunk(
        chunk_id=chunk_id, text="दिल्ली भारत की राजधानी है।", dense_score=0.9,
        sparse_score=0.5, fused_score=fused, language="hi", doc_id="d1",
    )


def test_query_returns_503_when_resources_not_ready(monkeypatch):
    monkeypatch.setitem(main_module.resources, "ready", False)
    client = TestClient(app)
    response = client.post("/query", json={"transcript": "भारत की राजधानी क्या है?"})
    assert response.status_code == 503


def test_query_happy_path_returns_answer(monkeypatch):
    monkeypatch.setitem(main_module.resources, "ready", True)
    monkeypatch.setitem(main_module.resources, "retrievers", {"recursive_512": FakeRetriever([_chunk()])})
    monkeypatch.setitem(
        main_module.resources, "generator",
        FakeGenerator(GeneratedAnswer(answer="दिल्ली", cited_chunk_ids=["c1"], sufficient=True, provider="groq")),
    )
    monkeypatch.setitem(main_module.resources, "grounding_verifier", None)
    monkeypatch.setattr(main_module.config.chunking, "active_strategy", "recursive_512")
    monkeypatch.setattr(
        main_module.config, "guardrails",
        GuardrailsConfig(
            tau_abs=0.01, tau_margin=0.01, grounding_threshold=None,
            enable_input=True, enable_retrieval_gate=True, enable_grounding=False,
        ),
    )

    client = TestClient(app)
    response = client.post("/query", json={"transcript": "भारत की राजधानी क्या है?"})
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "answered"
    assert body["answer"] == "दिल्ली"
    assert body["cited_chunk_ids"] == ["c1"]
    assert body["strategy_used"] == "recursive_512"


def test_query_refuses_unsafe_transcript(monkeypatch):
    monkeypatch.setitem(main_module.resources, "ready", True)
    monkeypatch.setitem(main_module.resources, "retrievers", {"recursive_512": FakeRetriever([_chunk()])})
    monkeypatch.setitem(main_module.resources, "generator", FakeGenerator(None))
    monkeypatch.setitem(main_module.resources, "grounding_verifier", None)
    monkeypatch.setattr(main_module.config.chunking, "active_strategy", "recursive_512")

    client = TestClient(app)
    response = client.post("/query", json={"transcript": "How to make a bomb"})
    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "refused"
    assert body["refusal_reason"] == "unsafe_content"


def test_query_unknown_strategy_returns_400(monkeypatch):
    monkeypatch.setitem(main_module.resources, "ready", True)
    monkeypatch.setitem(main_module.resources, "retrievers", {"recursive_512": FakeRetriever([_chunk()])})
    monkeypatch.setitem(main_module.resources, "generator", FakeGenerator(None))
    monkeypatch.setitem(main_module.resources, "grounding_verifier", None)

    client = TestClient(app)
    response = client.post(
        "/query", json={"transcript": "भारत की राजधानी क्या है?", "strategy": "nonexistent_strategy"}
    )
    assert response.status_code == 400


def test_strategies_endpoint_lists_loaded_strategies(monkeypatch):
    monkeypatch.setitem(
        main_module.resources, "retrievers",
        {"recursive_512": FakeRetriever([_chunk()]), "fixed_256_overlap_64": FakeRetriever([_chunk()])},
    )
    client = TestClient(app)
    response = client.get("/strategies")
    assert response.status_code == 200
    body = response.json()
    assert set(body["available"]) == {"recursive_512", "fixed_256_overlap_64"}


def test_query_rejects_missing_transcript():
    client = TestClient(app)
    response = client.post("/query", json={})
    assert response.status_code == 422
