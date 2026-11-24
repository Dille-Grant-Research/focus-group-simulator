"""Base interface for LLM providers.

This module defines the abstract base class that all LLM provider implementations
must inherit from. It ensures a consistent interface for interacting with different
LLM backends (Ollama, OpenAI, Anthropic, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageTotals:
    """Token usage accumulated across every completion this provider received."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    max_prompt_tokens: int = 0

    def record(self, usage: dict[str, Any] | None) -> None:
        """Add one response's `usage` block. A missing block still counts the call."""
        self.calls += 1
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens") or 0)
        self.prompt_tokens += prompt
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider implementations.

    This class defines the contract that all LLM providers must fulfill,
    ensuring consistent behavior across different backends.

    Implementations must provide:
    - Text generation with optional system prompts
    - Token counting for context management
    - Health check functionality
    - Model metadata (name, context limits)
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate text from the LLM based on the given prompt.

        This is the primary method for getting responses from the LLM.
        It should handle all provider-specific logic for making API calls
        and returning the generated text.

        Args:
            prompt: The user prompt/question to send to the LLM
            system_prompt: Optional system prompt to set context/behavior
            json_schema: Optional JSON Schema (an object schema) the reply must match.
                Providers map it to their structured-output option; callers must
                still parse defensively.

        Returns:
            The generated text response from the LLM

        Raises:
            Exception: Provider-specific exceptions for API errors, timeouts, etc.
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        This is used for context window management and determining when
        to trigger summarization. Different models may use different
        tokenization schemes.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens in the text
        """
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Get the maximum context window size in tokens.

        This represents the total number of tokens (prompt + completion)
        that the model can handle in a single request.

        Returns:
            Maximum number of tokens in the context window
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the model being used.

        This should return the full model identifier (e.g.,
        "llama3.1:8b-instruct-q4_K_M", "gpt-4", "claude-3-sonnet").

        Returns:
            The model name/identifier
        """
        pass

    @property
    def usage_totals(self) -> UsageTotals | None:
        """Token usage reported by the server, or None if the provider doesn't track it."""
        return None

    async def aclose(self) -> None:
        """Release any pooled connections held by the provider.

        Providers that keep a long-lived HTTP client override this. Safe to call
        more than once; a later generate() call may reopen resources.
        """
        return None

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the LLM provider is available and healthy.

        This should perform a lightweight health check to verify that
        the provider can be reached and is ready to handle requests.
        For local models, this might check if the server is running.
        For cloud providers, this might verify API credentials.

        Returns:
            True if the provider is available, False otherwise
        """
        pass
