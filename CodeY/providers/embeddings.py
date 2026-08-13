"""Embedding provider adapters used by semantic Skill routing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from http.client import RemoteDisconnected
import urllib.error
import urllib.request

from .clients import (
    OPENAI_COMPATIBLE_USER_AGENT,
    _http_retry_delay,
    _is_retryable_http_error,
    _normalize_versioned_base_url,
)


@dataclass(frozen=True)
class EmbeddingBatch:
    """A request-scoped batch of dense vectors and provider metadata."""

    vectors: tuple[tuple[float, ...], ...]
    model: str
    metadata: dict


def normalize_embedding_batch(value, expected_count=None):
    if not isinstance(value, EmbeddingBatch):
        raise TypeError("embedding clients must return EmbeddingBatch")
    vectors = tuple(_normalize_vector(vector) for vector in value.vectors)
    if expected_count is not None and len(vectors) != int(expected_count):
        raise ValueError("embedding response count does not match input count")
    if vectors:
        dimensions = len(vectors[0])
        if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("embedding vectors must have one consistent non-zero dimension")
    model = str(value.model or "").strip()
    if not model:
        raise ValueError("embedding response model is required")
    if not isinstance(value.metadata, dict):
        raise TypeError("embedding response metadata must be an object")
    return EmbeddingBatch(vectors=vectors, model=model, metadata=dict(value.metadata))


class FakeEmbeddingClient:
    """Deterministic embedding client for tests and Python API examples."""

    def __init__(self, vectors_by_text, model="fake-embedding"):
        self.vectors_by_text = {
            str(text): tuple(float(value) for value in vector)
            for text, vector in dict(vectors_by_text).items()
        }
        self.model = str(model)
        self.inputs = []

    @property
    def identity(self):
        return f"fake:{self.model}"

    def embed(self, texts):
        values = _normalize_texts(texts)
        self.inputs.append(values)
        try:
            vectors = tuple(self.vectors_by_text[text] for text in values)
        except KeyError as exc:
            raise RuntimeError(f"fake embedding is missing text: {exc.args[0]}") from exc
        return normalize_embedding_batch(
            EmbeddingBatch(vectors=vectors, model=self.model, metadata={}),
            expected_count=len(values),
        )


class OpenAICompatibleEmbeddingClient:
    """Dense embeddings over an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        model,
        base_url,
        api_key,
        timeout=30,
        attempts=3,
        dimensions=None,
    ):
        self.model = str(model).strip()
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = str(api_key or "")
        self.timeout = max(0.1, float(timeout))
        self.attempts = max(1, int(attempts))
        self.dimensions = None if dimensions is None else int(dimensions)
        if not self.model:
            raise ValueError("embedding model is required")
        if self.dimensions is not None and self.dimensions < 1:
            raise ValueError("embedding dimensions must be at least 1")

    @property
    def identity(self):
        dimensions = self.dimensions if self.dimensions is not None else "native"
        return f"openai-compatible:{self.base_url}:{self.model}:{dimensions}"

    def embed(self, texts):
        values = _normalize_texts(texts)
        payload = {
            "model": self.model,
            "input": list(values),
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        body_text, attempt = _request_json_body(
            request,
            timeout=self.timeout,
            attempts=self.attempts,
            backend="OpenAI-compatible embedding",
        )
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI-compatible embedding backend returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("OpenAI-compatible embedding response must be an object")
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible embedding error: {data['error']}")
        rows = data.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("OpenAI-compatible embedding response is missing data")
        indexed = {}
        for row in rows:
            if not isinstance(row, dict) or isinstance(row.get("index"), bool):
                raise RuntimeError("OpenAI-compatible embedding row is invalid")
            index = row.get("index")
            if not isinstance(index, int) or index in indexed:
                raise RuntimeError("OpenAI-compatible embedding indices are invalid")
            indexed[index] = row.get("embedding")
        if set(indexed) != set(range(len(values))):
            raise RuntimeError("OpenAI-compatible embedding response count/order is invalid")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return normalize_embedding_batch(
            EmbeddingBatch(
                vectors=tuple(indexed[index] for index in range(len(values))),
                model=str(data.get("model") or self.model),
                metadata={
                    "request_attempts": attempt,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
            ),
            expected_count=len(values),
        )


class OllamaEmbeddingClient:
    """Dense embeddings over Ollama's native ``/api/embed`` endpoint."""

    def __init__(self, model, host, timeout=30):
        self.model = str(model).strip()
        self.host = str(host).rstrip("/")
        self.timeout = max(0.1, float(timeout))
        if not self.model:
            raise ValueError("embedding model is required")
        if not self.host:
            raise ValueError("Ollama embedding host is required")

    @property
    def identity(self):
        return f"ollama:{self.host}:{self.model}"

    def embed(self, texts):
        values = _normalize_texts(texts)
        payload = {
            "model": self.model,
            "input": list(values),
            "truncate": True,
        }
        request = urllib.request.Request(
            self.host + "/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama embedding request failed with HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, RemoteDisconnected, TimeoutError) as exc:
            raise RuntimeError(
                "Could not reach the Ollama embedding backend.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from exc
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama embedding backend returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Ollama embedding response must be an object")
        if data.get("error"):
            raise RuntimeError(f"Ollama embedding error: {data['error']}")
        vectors = data.get("embeddings")
        if not isinstance(vectors, list):
            raise RuntimeError("Ollama embedding response is missing embeddings")
        return normalize_embedding_batch(
            EmbeddingBatch(
                vectors=tuple(vectors),
                model=str(data.get("model") or self.model),
                metadata={
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                },
            ),
            expected_count=len(values),
        )


def _request_json_body(request, *, timeout, attempts, backend):
    for attempt in range(max(1, int(attempts))):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8"), attempt + 1
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if _is_retryable_http_error(exc, body) and attempt < attempts - 1:
                time.sleep(_http_retry_delay(exc, attempt, body))
                continue
            raise RuntimeError(f"{backend} request failed with HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, RemoteDisconnected, TimeoutError) as exc:
            if attempt < attempts - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Could not reach the {backend} backend") from exc
    raise RuntimeError(f"{backend} request exhausted retries")


def _normalize_texts(texts):
    if isinstance(texts, (str, bytes)):
        raise TypeError("embedding input must be a sequence of strings")
    values = tuple(str(text) for text in texts)
    if not values or any(not text.strip() for text in values):
        raise ValueError("embedding input requires non-empty strings")
    return values


def _normalize_vector(vector):
    if isinstance(vector, (str, bytes)):
        raise TypeError("embedding vector must be numeric")
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as exc:
        raise TypeError("embedding vector must be numeric") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("embedding vector must contain finite numbers")
    return values
