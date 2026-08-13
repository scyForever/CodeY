import json

import pytest

from CodeY.providers.embeddings import (
    OllamaEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)
from CodeY.cli import _build_skill_embedding_client, build_arg_parser


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_compatible_embedding_client_batches_and_orders_rows(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return StubResponse(
            {
                "model": "text-embedding-test",
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        )

    monkeypatch.setattr("CodeY.providers.embeddings.urllib.request.urlopen", urlopen)
    client = OpenAICompatibleEmbeddingClient(
        model="text-embedding-test",
        base_url="https://example.test/api",
        api_key="test-key",
        timeout=12,
        dimensions=2,
    )

    batch = client.embed(("alpha", "beta"))

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "https://example.test/api/v1/embeddings"
    assert request.get_header("Authorization") == "Bearer test-key"
    assert captured["timeout"] == 12
    assert payload == {
        "model": "text-embedding-test",
        "input": ["alpha", "beta"],
        "encoding_format": "float",
        "dimensions": 2,
    }
    assert batch.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert batch.metadata["prompt_tokens"] == 4


def test_openai_compatible_embedding_client_rejects_invalid_indices(monkeypatch):
    monkeypatch.setattr(
        "CodeY.providers.embeddings.urllib.request.urlopen",
        lambda request, timeout: StubResponse(
            {"data": [{"index": 1, "embedding": [1.0, 0.0]}]}
        ),
    )
    client = OpenAICompatibleEmbeddingClient(
        model="text-embedding-test",
        base_url="https://example.test/v1",
        api_key="",
        attempts=1,
    )

    with pytest.raises(RuntimeError, match="count/order"):
        client.embed(("alpha",))


def test_ollama_embedding_client_uses_native_batch_endpoint(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return StubResponse(
            {
                "model": "embeddinggemma",
                "embeddings": [[1.0, 0.0], [0.0, 1.0]],
                "prompt_eval_count": 7,
            }
        )

    monkeypatch.setattr("CodeY.providers.embeddings.urllib.request.urlopen", urlopen)
    client = OllamaEmbeddingClient(
        model="embeddinggemma",
        host="http://127.0.0.1:11434/",
        timeout=9,
    )

    batch = client.embed(("alpha", "beta"))

    request = captured["request"]
    assert request.full_url == "http://127.0.0.1:11434/api/embed"
    assert captured["timeout"] == 9
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "embeddinggemma",
        "input": ["alpha", "beta"],
        "truncate": True,
    }
    assert batch.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert batch.metadata["prompt_eval_count"] == 7


def test_embedding_endpoint_does_not_inherit_main_model_gateway_key(monkeypatch):
    monkeypatch.delenv("CODEY_SKILL_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("CODEY_OPENAI_API_KEY", "main-model-gateway-key")
    args = build_arg_parser().parse_args(["--skill-embedding-provider", "openai"])

    client = _build_skill_embedding_client(args)

    assert client.api_key == ""
