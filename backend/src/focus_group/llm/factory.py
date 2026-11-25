"""Factory for creating LLM providers from configuration.

This module provides a simple factory function to create LLM provider
instances based on configuration. Supports Ollama (local models) and
OpenAI-compatible cloud providers (DeepSeek, OpenRouter, Together AI, etc.).
"""

from focus_group.llm.base import BaseLLMProvider
from focus_group.llm.ollama import OllamaProvider
from focus_group.llm.openai_compat import OpenAICompatibleProvider
from focus_group.models.config import CloudModelConfig, OllamaModelConfig


def create_provider(
    config: OllamaModelConfig | CloudModelConfig,
) -> BaseLLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: An OllamaModelConfig or CloudModelConfig.

    Returns:
        A configured LLM provider instance ready for use.

    Raises:
        TypeError: If config is not a recognized configuration type.

    Example:
        >>> from focus_group.llm import create_provider
        >>> from focus_group.models.config import OllamaModelConfig
        >>>
        >>> config = OllamaModelConfig(model_name="llama3.1:8b-instruct-q4_K_M")
        >>> provider = create_provider(config)
    """
    if isinstance(config, CloudModelConfig):
        return OpenAICompatibleProvider(config)

    if isinstance(config, OllamaModelConfig):
        return OllamaProvider(config)

    raise TypeError(
        f"Expected OllamaModelConfig or CloudModelConfig, got {type(config).__name__}"
    )
