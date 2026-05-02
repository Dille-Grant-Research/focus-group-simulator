"""Ollama LLM provider implementation.

This module provides the OllamaProvider class for interacting with local
Ollama models. Ollama allows running LLMs locally without requiring cloud
API access.
"""

import asyncio
import logging

import httpx

from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import OllamaModelConfig
from focus_group.utils.tokens import count_tokens as estimate_tokens

logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Raised when unable to connect to Ollama server."""

    pass


class OllamaGenerationError(Exception):
    """Raised when Ollama fails to generate a response."""

    pass


class OllamaProvider(BaseLLMProvider):
    """LLM provider for local Ollama models.

    This provider connects to a locally running Ollama server and uses it
    to generate text. It supports all Ollama configuration options including
    temperature, top_p, and context window size.

    Attributes:
        config: The Ollama configuration settings

    Example:
        >>> config = OllamaModelConfig(
        ...     model_name="llama3.1:8b-instruct-q4_K_M",
        ...     base_url="http://localhost:11434",
        ...     temperature=0.7
        ... )
        >>> provider = OllamaProvider(config)
        >>> if await provider.is_available():
        ...     response = await provider.generate("Hello!")
    """

    def __init__(
        self,
        config: OllamaModelConfig | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            config: Ollama configuration. If None, uses default settings.
            max_retries: Maximum number of retry attempts for transient failures.
            retry_delay: Initial delay between retries (uses exponential backoff).
        """
        self._config = config or OllamaModelConfig()
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def config(self) -> OllamaModelConfig:
        """Get the current configuration."""
        return self._config

    @property
    def model_name(self) -> str:
        """Get the name of the model being used.

        Returns:
            The Ollama model name (e.g., "llama3.1:8b-instruct-q4_K_M")
        """
        return self._config.model_name

    @property
    def max_context_tokens(self) -> int:
        """Get the maximum context window size in tokens.

        Returns:
            The configured num_ctx value from the Ollama config
        """
        return self._config.num_ctx

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Uses the utility function for token estimation. This is a conservative
        estimate that works across different models.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated number of tokens
        """
        return estimate_tokens(text)

    async def is_available(self) -> bool:
        """Check if the Ollama server is available and the model is loaded.

        This performs a lightweight check by calling the Ollama API's
        tag listing endpoint.

        Returns:
            True if Ollama is running and accessible, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._config.base_url}/api/tags")
                return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from the Ollama model.

        Sends the prompt to Ollama and returns the generated response.
        Implements retry logic with exponential backoff for transient failures.

        Args:
            prompt: The user prompt/question to send to the model
            system_prompt: Optional system prompt to set context/behavior

        Returns:
            The generated text response

        Raises:
            OllamaConnectionError: If unable to connect to Ollama server
            OllamaGenerationError: If the generation request fails
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._generate_once(prompt, system_prompt)
            except OllamaConnectionError:
                # Connection errors are not retryable
                raise
            except OllamaGenerationError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2**attempt)
                    logger.warning(
                        f"Ollama generation failed (attempt {attempt + 1}/{self._max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Ollama generation failed after {self._max_retries + 1} attempts")

        raise last_error or OllamaGenerationError("Generation failed with no error")

    async def _generate_once(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Perform a single generation request to Ollama.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The generated text

        Raises:
            OllamaConnectionError: If unable to connect to Ollama
            OllamaGenerationError: If the request fails
        """
        # Build the request payload
        payload: dict = {
            "model": self._config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "num_ctx": self._config.num_ctx,
            },
        }

        # Add GPU layers if specified
        if self._config.num_gpu != -1:
            payload["options"]["num_gpu"] = self._config.num_gpu

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{self._config.base_url}/api/generate",
                    json=payload,
                )

                if response.status_code == 404:
                    raise OllamaGenerationError(
                        f"Model '{self._config.model_name}' not found. "
                        f"Please run 'ollama pull {self._config.model_name}' first."
                    )

                if response.status_code != 200:
                    raise OllamaGenerationError(
                        f"Ollama API returned status {response.status_code}: " f"{response.text}"
                    )

                data = response.json()
                return data.get("response", "")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self._config.base_url}. "
                f"Is Ollama running? Start it with 'ollama serve'. Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaGenerationError(
                f"Ollama request timed out after 300 seconds. "
                f"The model may be loading or the prompt too long. Error: {e}"
            ) from e
        except httpx.RequestError as e:
            raise OllamaGenerationError(f"Failed to communicate with Ollama: {e}") from e

    async def check_model_available(self) -> bool:
        """Check if the configured model is available locally.

        This queries Ollama's model list to verify the model has been pulled.

        Returns:
            True if the model is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._config.base_url}/api/tags")
                if response.status_code != 200:
                    return False

                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "") for m in models]

                # Check for exact match or match without tag
                target = self._config.model_name
                target_base = target.split(":")[0]

                for name in model_names:
                    if name == target or name.startswith(f"{target_base}:"):
                        return True

                return False
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"OllamaProvider(model={self._config.model_name!r}, "
            f"base_url={self._config.base_url!r})"
        )
