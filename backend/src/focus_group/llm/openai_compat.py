"""OpenAI-compatible cloud LLM provider implementation.

This module provides the OpenAICompatibleProvider class for interacting with
cloud APIs that follow the OpenAI chat completions format. This includes
DeepSeek, OpenRouter, Together AI, Groq, and OpenAI itself.
"""

import asyncio
import logging
import os
import re

import httpx

from focus_group.llm.base import BaseLLMProvider
from focus_group.models.config import CloudModelConfig
from focus_group.utils.tokens import count_tokens as estimate_tokens

logger = logging.getLogger(__name__)

# Availability probes hit the lightweight model list, not a generation.
_AVAILABILITY_TIMEOUT_SECONDS = 10.0
_AUTH_FAILURE_STATUSES = (401, 403)

# Reasoning models (e.g. Qwen3) may emit their chain of thought inline as
# <think>...</think>. An unclosed block means the response was truncated mid-thought.
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def strip_reasoning(content: str) -> str:
    """Remove inline <think> reasoning blocks from model output."""
    return _THINK_BLOCK_RE.sub("", content).strip()


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

        If ``config.api_key_env_var`` is None the endpoint is treated as
        unauthenticated and no Authorization header is sent.
        """
        self._config = config
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._api_key = ""
        if config.api_key_env_var is not None:
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

    def _headers(self) -> dict[str, str]:
        """Build request headers, adding Bearer auth only when a key is configured."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def is_available(self) -> bool:
        """Check that the API is reachable, authorized, and serves the configured model.

        Uses GET /v1/models rather than a generation request so the probe is
        cheap even on large models.

        Returns:
            True if the model is listed by the API, False otherwise
        """
        if self._config.api_key_env_var is not None and not self._api_key:
            return False

        try:
            async with httpx.AsyncClient(timeout=_AVAILABILITY_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{self._config.base_url}/v1/models",
                    headers=self._headers(),
                )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.debug(f"API availability check failed: {e}")
            return False

        if response.status_code in _AUTH_FAILURE_STATUSES:
            logger.debug(f"API authentication failed: {response.status_code}")
            return False
        if response.status_code != 200:
            logger.debug(f"Model list returned HTTP {response.status_code}")
            return False

        try:
            model_ids = {m.get("id") for m in response.json().get("data", [])}
        except (ValueError, AttributeError):
            logger.debug("Model list response was not valid JSON")
            return False

        if self._config.model_name not in model_ids:
            logger.warning(
                f"Model '{self._config.model_name}' not served at {self._config.base_url}. "
                f"Available: {sorted(i for i in model_ids if i)}"
            )
            return False
        return True

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
            async with httpx.AsyncClient(timeout=self._config.request_timeout_seconds) as client:
                response = await client.post(
                    f"{self._config.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )

                if response.status_code in _AUTH_FAILURE_STATUSES:
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

                content = choices[0].get("message", {}).get("content") or ""
                return strip_reasoning(content)

        except httpx.ConnectError as e:
            raise OpenAIConnectionError(
                f"Cannot connect to API at {self._config.base_url}. "
                f"Check your network connection. Error: {e}"
            ) from e
        except httpx.TimeoutException as e:
            raise OpenAIGenerationError(
                f"API request timed out after {self._config.request_timeout_seconds} seconds. "
                f"Error: {e}"
            ) from e
        except httpx.RequestError as e:
            raise OpenAIGenerationError(f"Failed to communicate with API: {e}") from e

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"OpenAICompatibleProvider(model={self._config.model_name!r}, "
            f"base_url={self._config.base_url!r})"
        )
