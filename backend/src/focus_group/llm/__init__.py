"""LLM provider abstraction layer."""

from focus_group.llm.base import BaseLLMProvider
from focus_group.llm.factory import create_provider
from focus_group.llm.ollama import (
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaProvider,
)
from focus_group.llm.openai_compat import (
    OpenAICompatibleProvider,
    OpenAIConnectionError,
    OpenAIGenerationError,
)

__all__ = [
    "BaseLLMProvider",
    "OllamaConnectionError",
    "OllamaGenerationError",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIConnectionError",
    "OpenAIGenerationError",
    "create_provider",
]
