"""Tests for the OpenAI-compatible cloud provider."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from focus_group.llm.factory import create_provider
from focus_group.llm.openai_compat import (
    OpenAICompatibleProvider,
    OpenAIConnectionError,
)
from focus_group.models.config import (
    CloudModelConfig,
    LLMProviderType,
    OllamaModelConfig,
    SessionConfig,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cloud_config() -> CloudModelConfig:
    """Cloud config for testing."""
    return CloudModelConfig(
        provider=LLMProviderType.OPENAI,
        model_name="test-model",
        api_key_env_var="TEST_OPENAI_KEY",
        base_url="https://api.test.com",
        max_tokens=1024,
        temperature=0.5,
        top_p=0.9,
        num_ctx=4096,
    )


@pytest.fixture
def provider_with_key(cloud_config: CloudModelConfig) -> OpenAICompatibleProvider:
    """Provider with API key set."""
    with patch.dict(os.environ, {"TEST_OPENAI_KEY": "sk-test-key-123"}):
        return OpenAICompatibleProvider(cloud_config)


@pytest.fixture
def provider_no_key(cloud_config: CloudModelConfig) -> OpenAICompatibleProvider:
    """Provider without API key set."""
    with patch.dict(os.environ, {}, clear=True):
        # Ensure the env var is not set
        os.environ.pop("TEST_OPENAI_KEY", None)
        return OpenAICompatibleProvider(cloud_config)


def _mock_response(
    status_code: int, json_data: dict | None = None, text: str = ""
) -> httpx.Response:
    """Create a mock httpx Response."""
    import json as json_lib

    if json_data is not None:
        content = json_lib.dumps(json_data).encode("utf-8")
    else:
        content = text.encode("utf-8")

    return httpx.Response(
        status_code=status_code,
        content=content,
        request=httpx.Request("POST", "https://api.test.com/v1/chat/completions"),
    )


def _success_response(content: str = "Hello from the model!") -> httpx.Response:
    """Create a successful chat completion response."""
    return _mock_response(
        200,
        json_data={
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


# ============================================================================
# Init Tests
# ============================================================================


class TestProviderInit:
    """Tests for provider initialization."""

    def test_init_with_api_key(self, cloud_config: CloudModelConfig) -> None:
        """Provider reads API key from environment."""
        with patch.dict(os.environ, {"TEST_OPENAI_KEY": "sk-test-123"}):
            provider = OpenAICompatibleProvider(cloud_config)
            assert provider._api_key == "sk-test-123"

    def test_init_without_api_key_warns(self, cloud_config: CloudModelConfig) -> None:
        """Provider warns when API key env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TEST_OPENAI_KEY", None)
            provider = OpenAICompatibleProvider(cloud_config)
            assert provider._api_key == ""

    def test_model_name_property(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """model_name returns the configured model name."""
        assert provider_with_key.model_name == "test-model"

    def test_max_context_tokens_property(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """max_context_tokens returns num_ctx from config."""
        assert provider_with_key.max_context_tokens == 4096

    def test_repr(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """repr shows model and base_url."""
        r = repr(provider_with_key)
        assert "test-model" in r
        assert "api.test.com" in r


# ============================================================================
# Generate Tests
# ============================================================================


class TestGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """Successful generation returns content."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_success_response("Test response"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            result = await provider_with_key.generate("Hello")

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """System prompt is included in messages array."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_success_response("OK"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider_with_key.generate("Hello", system_prompt="Be helpful")

        # Check the payload sent
        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert payload["messages"][1] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_generate_without_system_prompt(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """Without system prompt, only user message is sent."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_success_response("OK"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider_with_key.generate("Hello")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert len(payload["messages"]) == 1
        assert payload["messages"][0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_correct_payload_structure(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """Payload includes model, messages, max_tokens, temperature, top_p."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_success_response())

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider_with_key.generate("Hello")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["max_tokens"] == 1024
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_correct_auth_header(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """Authorization header uses Bearer token."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_success_response())

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider_with_key.generate("Hello")

        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-test-key-123"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_auth_error_raises_connection_error(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """401 raises OpenAIConnectionError and is not retried."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_mock_response(401, text="Unauthorized"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(OpenAIConnectionError, match="Authentication failed"):
                await provider_with_key.generate("Hello")

        # Should only be called once (no retries for auth errors)
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_retries(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """429 triggers retries with backoff."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=[
                _mock_response(429, text="Rate limited"),
                _mock_response(429, text="Rate limited"),
                _success_response("Finally!"),
            ]
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider_with_key.generate("Hello")

        assert result == "Finally!"
        assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_server_error_retries(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """5xx triggers retries."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=[
                _mock_response(500, text="Internal Server Error"),
                _success_response("Recovered!"),
            ]
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await provider_with_key.generate("Hello")

        assert result == "Recovered!"

    @pytest.mark.asyncio
    async def test_connection_refused_raises_connection_error(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """ConnectError raises OpenAIConnectionError and is not retried."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(OpenAIConnectionError, match="Cannot connect"):
                await provider_with_key.generate("Hello")

        assert mock_client.post.call_count == 1


# ============================================================================
# Availability Tests
# ============================================================================


class TestIsAvailable:
    """Tests for the is_available method."""

    @pytest.mark.asyncio
    async def test_available_with_valid_key(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """Returns True when API responds successfully."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_success_response())

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_with_key.is_available() is True

    @pytest.mark.asyncio
    async def test_unavailable_without_key(self, provider_no_key: OpenAICompatibleProvider) -> None:
        """Returns False when no API key is set."""
        assert await provider_no_key.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_on_auth_failure(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """Returns False when API returns 401."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=_mock_response(401, text="Unauthorized"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_with_key.is_available() is False

    @pytest.mark.asyncio
    async def test_unavailable_on_network_error(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """Returns False when network is unreachable."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_with_key.is_available() is False


# ============================================================================
# Token Counting Tests
# ============================================================================


class TestCountTokens:
    """Tests for token counting."""

    @pytest.mark.asyncio
    async def test_count_tokens(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """count_tokens returns a reasonable estimate."""
        count = await provider_with_key.count_tokens("Hello world test")
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens_empty(self, provider_with_key: OpenAICompatibleProvider) -> None:
        """count_tokens returns 0 for empty string."""
        count = await provider_with_key.count_tokens("")
        assert count == 0


# ============================================================================
# Factory Integration Tests
# ============================================================================


class TestFactoryIntegration:
    """Tests for factory creating the correct provider."""

    def test_factory_creates_openai_provider(self, cloud_config: CloudModelConfig) -> None:
        """Factory returns OpenAICompatibleProvider for CloudModelConfig."""
        with patch.dict(os.environ, {"TEST_OPENAI_KEY": "sk-test"}):
            provider = create_provider(cloud_config)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.model_name == "test-model"

    def test_factory_still_creates_ollama_provider(self) -> None:
        """Factory still returns OllamaProvider for OllamaModelConfig."""
        from focus_group.llm.ollama import OllamaProvider

        config = OllamaModelConfig(model_name="llama3.1:8b")
        provider = create_provider(config)
        assert isinstance(provider, OllamaProvider)


# ============================================================================
# Config Parsing Tests
# ============================================================================


class TestConfigParsing:
    """Tests for SessionConfig parsing with cloud model configs."""

    def test_session_config_parses_cloud_model(self) -> None:
        """SessionConfig accepts cloud model config via model_validator."""
        config = SessionConfig(
            demographics=["Test person, 30-40"],
            questions=["What do you think?"],
            model={
                "provider": "openai",
                "model_name": "deepseek-chat",
                "api_key_env_var": "DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
            },
        )
        assert isinstance(config.model, CloudModelConfig)
        assert config.model.model_name == "deepseek-chat"
        assert config.model.base_url == "https://api.deepseek.com"

    def test_session_config_backward_compat_ollama(self) -> None:
        """SessionConfig still works with Ollama model config (no provider key)."""
        config = SessionConfig(
            demographics=["Test person, 30-40"],
            questions=["What do you think?"],
            model={
                "model_name": "llama3.1:8b",
                "base_url": "http://localhost:11434",
                "num_ctx": 8192,
            },
        )
        assert isinstance(config.model, OllamaModelConfig)
        assert config.model.model_name == "llama3.1:8b"

    def test_session_config_default_model(self) -> None:
        """SessionConfig defaults to OllamaModelConfig when model is omitted."""
        config = SessionConfig(
            demographics=["Test person, 30-40"],
            questions=["What do you think?"],
        )
        assert isinstance(config.model, OllamaModelConfig)

    def test_cloud_config_defaults(self) -> None:
        """CloudModelConfig has correct defaults for optional fields."""
        config = CloudModelConfig(
            provider=LLMProviderType.OPENAI,
            model_name="test-model",
            api_key_env_var="MY_KEY",
        )
        assert config.base_url == "https://api.openai.com"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.num_ctx == 8192
