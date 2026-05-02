"""OpenAI-compatible cloud LLM provider implementation.

This module provides the OpenAICompatibleProvider class for interacting with
cloud APIs that follow the OpenAI chat completions format. This includes
DeepSeek, OpenRouter, Together AI, Groq, and OpenAI itself.
"""

import asyncio
import logging
import os

import httpx

from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import CloudModelConfig
from focus_group.utils.tokens import count_tokens as estimate_tokens

logger = logging.getLogger(__name__)


class OpenAIConnectionError(Exception):
    """Raised when unable to connect to the cloud API or authentication fails."""

    pass


class OpenAIGenerationError(Exception):
    """Raised when the cloud API fails to generate a response."""

    pass


class OpenAICompatibleProvider(BaseLLMProvider):
    """LLM provider for OpenAI-compatible cloud APIs.

    This provider connects to any API that follows the OpenAI chat completions
    format (POST /v1/chat/completions). It supports DeepSeek, OpenRouter,
    Together AI, Groq, and OpenAI itself.

    Attributes:
        config: The cloud model configuration settings

    Example:
        >>> config = CloudModelConfig(
        ...     provider=LLMProviderType.OPENAI,
        ...     model_name="deepseek-chat",
        ...     api_key_env_var="DEEPSEEK_API_KEY",
        ...     base_url="https://api.deepseek.com",
        ... )
        >>> provider = OpenAICompatibleProvider(config)
        >>> if await provider.is_available():
        ...     response = await provider.generate("Hello!")
    """

    def __init__(
        self,
        config: CloudModelConfig,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the OpenAI-compatible provider.

        Args:
            config: Cloud model configuration.
            max_retries: Maximum number of retry attempts for transient failures.
            retry_delay: Initial delay between retries (uses exponential backoff).

        Raises:
            OpenAIConnectionError: If the API key environment variable is not set.
        """
        self._config = config
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Read API key from environment
        self._api_key = os.environ.get(config.api_key_env_var, "")
        if not self._api_key:
            logger.warning(
                f"API key environment variable '{config.api_key_env_var}' is not set. "
                f"Requests will fail with authentication errors."
            )

    @property
    def config(self) -> CloudModelConfig:
        """Get the current configuration."""
        return self._config

    @property
    def model_name(self) -> str:
        """Get the name of the model being used."""
        return self._config.model_name

    @property
    def max_context_tokens(self) -> int:
        """Get the maximum context window size in tokens."""
        return self._config.num_ctx

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Uses the utility function for token estimation.

        Args:
            text: The text to count tokens for

        Returns:
            Estimated number of tokens
        """
        return estimate_tokens(text)

    async def is_available(self) -> bool:
        """Check if the cloud API is reachable and authentication is valid.

        Sends a minimal request to verify connectivity and auth.

        Returns:
            True if the API is accessible and authenticated, False otherwise
        """
        if not self._api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._config.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._config.model_name,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                    },
                )
                # 200 = success, means API is reachable and auth is valid
                # 401/403 = auth failure
                # Other codes still mean the API is reachable
                if response.status_code in (401, 403):
                    logger.debug(f"API authentication failed: {response.status_code}")
                    return False
                return True
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.debug(f"API availability check failed: {e}")
            return False

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from the cloud model.

        Sends the prompt to the API and returns the generated response.
        Implements retry logic with exponential backoff for rate limits
        and server errors. Authentication errors are not retried.

        Args:
            prompt: The user prompt/question to send to the model
            system_prompt: Optional system prompt to set context/behavior

        Returns:
            The generated text response

        Raises:
            OpenAIConnectionError: If unable to connect or auth fails
            OpenAIGenerationError: If the generation request fails
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._generate_once(prompt, system_prompt)
            except OpenAIConnectionError:
                # Auth/connection errors are not retryable
                raise
            except OpenAIGenerationError as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2**attempt)
                    logger.warning(
                        f"API generation failed (attempt {attempt + 1}/{self._max_retries + 1}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API generation failed after {self._max_retries + 1} attempts")

        raise last_error or OpenAIGenerationError("Generation failed with no error")

    async def _generate_once(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Perform a single generation request to the API.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            The generated text

        Raises:
            OpenAIConnectionError: If unable to connect or auth fails
            OpenAIGenerationError: If the request fails
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._config.model_name,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self._config.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                if response.status_code in (401, 403):
                    raise OpenAIConnectionError(
                        f"Authentication failed (HTTP {response.status_code}). "
                        f"Check your API key in environment variable '{self._config.api_key_env_var}'."
                    )

                if response.status_code == 429:
                    raise OpenAIGenerationError(f"Rate limited (HTTP 429). {response.text}")

                if response.status_code >= 500:
                    raise OpenAIGenerationError(
                        f"Server error (HTTP {response.status_code}): {response.text}"
                    )

                if response.status_code != 200:
                    raise OpenAIGenerationError(
                        f"API returned status {response.status_code}: {response.text}"
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise OpenAIGenerationError("API returned no choices in response")

                return choices[0].get("message", {}).get("content", "")

        except httpx.ConnectError as e:
            raise OpenAIConnectionError(
                f"Cannot connect to API at {self._config.base_url}. "
                f"Check your network connection. Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OpenAIGenerationError(
                f"API request timed out after 120 seconds. Error: {e}"
            ) from e
        except httpx.RequestError as e:
            raise OpenAIGenerationError(f"Failed to communicate with API: {e}") from e

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"OpenAICompatibleProvider(model={self._config.model_name!r}, "
            f"base_url={self._config.base_url!r})"
        )
