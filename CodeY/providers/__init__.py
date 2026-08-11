"""Model provider adapters."""

from .clients import (
    AnthropicCompatibleModelClient,
    FakeModelClient,
    ModelCompletion,
    OllamaModelClient,
    OpenAICompatibleModelClient,
)

__all__ = [
    "AnthropicCompatibleModelClient",
    "FakeModelClient",
    "ModelCompletion",
    "OllamaModelClient",
    "OpenAICompatibleModelClient",
]
