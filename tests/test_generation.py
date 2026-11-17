import asyncio
import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from contextdock.app import DEMO_TOKENS, create_app
from contextdock.generation import GenerationError, Generator
from contextdock.retrieval import LocalEncoder
from contextdock.store import Store

ALICE = {"Authorization": "Bearer " + DEMO_TOKENS["alice"]}
BOB = {"Authorization": "Bearer " + DEMO_TOKENS["bob"]}
TEXT = "Café invoices close on Friday. Late invoices wait until the next cycle."


def response(claims):
    return httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"claims": claims}),
    }}]})


def test_generated_answers_use_only_current_tenant_sources_and_exact_quotes(tmp_path):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        data = json.loads(payload["messages"][1]["content"])
        requests.append(data)
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert "Monday" not in request.content.decode()
        chunk = data["sources"][0]["chunk_id"]
        return response([
            {"text": "Invoices close on Friday.", "citations": [{"chunk_id": chunk, "quote": "Café invoices close on Friday."}]},
            {"text": "Unsupported source", "citations": [{"chunk_id": 999_999, "quote": "Secret Monday rule."}]},
            {"text": "Unsupported quote", "citations": [{"chunk_id": chunk, "quote": "Not present in the source."}]},
        ])

    generator = Generator("https://answer.invalid", "mock-answer", "secret-test-key", httpx2.MockTransport(respond))
    assert "secret-test-key" not in repr(generator)
    with TestClient(create_app(tmp_path / "context.sqlite", DEMO_TOKENS, LocalEncoder(), demo=True, generator=generator)) as client:
        alice = client.post("/api/documents", headers=ALICE, json={"collection": "policies", "title": "Friday rule", "text": TEXT}).json()["id"]
        bob = client.post("/api/documents", headers=BOB, json={"collection": "policies", "title": "Private Monday", "text": "Secret Monday rule. Invoices close on Monday."}).json()["id"]
        assert client.post("/api/search", json={"query": "invoice close"}).status_code == 401
        result = client.post("/api/search", headers=ALICE, json={"query": "café invoices close"}).json()
        assert result["answer_mode"] == "generative"
        assert result["generation_error"] is None
        assert len(result["generated"]["claims"]) == 1
        citation = result["generated"]["claims"][0]["citations"][0]
        assert citation["document_id"] == alice
        assert TEXT[citation["start"]:citation["end"]] == citation["quote"]
        assert all(source["document_id"] != bob for source in result["sources"])
        assert len(requests) == 1


@pytest.mark.parametrize("provider_error", [False, True])
def test_deletion_during_generation_removes_citations_and_fallback_excerpts(tmp_path, provider_error):
    path = tmp_path / "context.sqlite"
    identifier = None

    def respond(request):
        assert Store(path, LocalEncoder.identity).delete("alice", identifier)
        if provider_error:
            return httpx2.Response(429, text="sensitive upstream response")
        data = json.loads(json.loads(request.content)["messages"][1]["content"])
        return response([{"text": "Invoices close Friday.", "citations": [
            {"chunk_id": data["sources"][0]["chunk_id"], "quote": "Café invoices close on Friday."},
        ]}])

    generator = Generator("https://answer.invalid", "mock-answer", "test-only", httpx2.MockTransport(respond))
    with TestClient(create_app(path, DEMO_TOKENS, LocalEncoder(), demo=True, generator=generator)) as client:
        identifier = client.post("/api/documents", headers=ALICE, json={"collection": "policies", "title": "Friday", "text": TEXT}).json()["id"]
        result = client.post("/api/search", headers=ALICE, json={"query": "café invoices close"})
        assert result.status_code == 200
        body = result.json()
        assert body["sources"] == body["answer"] == []
        assert body["generated"] is None
        assert body["answer_mode"] == "extractive"
        assert "sensitive" not in result.text


def test_empty_retrieval_makes_no_provider_call(tmp_path):
    def unexpected(_request):
        raise AssertionError("No model request is authorized without retrieved evidence")

    generator = Generator("https://answer.invalid", "mock", "test-only", httpx2.MockTransport(unexpected))
    with TestClient(create_app(tmp_path / "context.sqlite", DEMO_TOKENS, LocalEncoder(), demo=True, generator=generator)) as client:
        body = client.post("/api/search", headers=ALICE, json={"query": "invoice close"}).json()
        assert body["sources"] == []
        assert body["generated"] is None
        assert body["generation_error"] is None


def test_provider_response_is_bounded_and_refusals_fail_safely():
    for upstream in [httpx2.Response(200, content=b"x" * 40_000),
                     httpx2.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "No"}}]})]:
        generator = Generator("https://answer.invalid", "mock", "test-only", httpx2.MockTransport(lambda _, value=upstream: value))
        with pytest.raises(GenerationError, match="answer_invalid_response"):
            asyncio.run(generator.generate("invoice", [{"chunk_id": 1, "text": TEXT}]))
