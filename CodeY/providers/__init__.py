"""Model provider adapters."""

from .clients import (
    AnthropicCompatibleModelClient,
    FakeModelClient,
    ModelCompletion,
    OllamaModelClient,
    OpenAICompatibleModelClient,
)
from .embeddings import (
    EmbeddingBatch,
    FakeEmbeddingClient,
    OllamaEmbeddingClient,
    OpenAICompatibleEmbeddingClient,
)

__all__ = [
    "AnthropicCompatibleModelClient",
    "FakeModelClient",
    "ModelCompletion",
    "OllamaModelClient",
    "OpenAICompatibleModelClient",
    "EmbeddingBatch",
    "FakeEmbeddingClient",
    "OllamaEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
]
