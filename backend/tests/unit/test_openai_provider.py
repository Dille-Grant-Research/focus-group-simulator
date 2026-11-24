"""Tests for the OpenAI-compatible cloud provider."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from focus_group.llm.factory import create_provider
from focus_group.llm.openai_compat import (
    OpenAICompatibleProvider,
    OpenAIConnectionError,
    OpenAIContextOverflowError,
    OpenAIGenerationError,
    strip_reasoning,
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


_TEST_TIMEOUT_SECONDS = 45.0


@pytest.fixture
def provider_unauthenticated() -> OpenAICompatibleProvider:
    """Provider for a keyless endpoint (api_key_env_var=None)."""
    return OpenAICompatibleProvider(
        CloudModelConfig(
            provider=LLMProviderType.OPENAI,
            model_name="test-model",
            base_url="http://cruncher.test:1234",
            request_timeout_seconds=_TEST_TIMEOUT_SECONDS,
        )
    )


def _mock_client(post: AsyncMock | None = None, get: AsyncMock | None = None) -> AsyncMock:
    """Create a mock httpx.AsyncClient usable as an async context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if post is not None:
        mock_client.post = post
    if get is not None:
        mock_client.get = get
    return mock_client


def _models_response(model_ids: list[str]) -> httpx.Response:
    """Create a /v1/models list response."""
    return _mock_response(
        200,
        json_data={"object": "list", "data": [{"id": i, "object": "model"} for i in model_ids]},
    )


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
    """Tests for the is_available method (GET /v1/models)."""

    @pytest.mark.asyncio
    async def test_available_when_model_listed(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        """Returns True when the configured model is in the model list."""
        mock_client = _mock_client(get=AsyncMock(return_value=_models_response(["test-model"])))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_with_key.is_available() is True

        mock_client.get.assert_called_once()
        assert mock_client.get.call_args[0][0] == "https://api.test.com/v1/models"

    @pytest.mark.asyncio
    async def test_unavailable_without_key(self, provider_no_key: OpenAICompatibleProvider) -> None:
        """Returns False when a key env var is configured but not set."""
        assert await provider_no_key.is_available() is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_mock",
        [
            pytest.param(AsyncMock(return_value=_models_response(["other-model"])), id="model-missing"),
            pytest.param(AsyncMock(return_value=_mock_response(401, text="Unauthorized")), id="auth-401"),
            pytest.param(AsyncMock(return_value=_mock_response(500, text="boom")), id="server-500"),
            pytest.param(AsyncMock(return_value=_mock_response(200, text="not json")), id="bad-json"),
            pytest.param(AsyncMock(side_effect=httpx.ConnectError("refused")), id="network-error"),
        ],
    )
    async def test_unavailable(
        self, provider_with_key: OpenAICompatibleProvider, get_mock: AsyncMock
    ) -> None:
        """Returns False when the model list can't confirm the configured model."""
        mock_client = _mock_client(get=get_mock)

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_with_key.is_available() is False


# ============================================================================
# Unauthenticated Endpoint Tests (e.g. self-hosted LM Studio)
# ============================================================================


class TestUnauthenticatedEndpoint:
    """Provider configured with no API key env var at all."""

    @pytest.mark.asyncio
    async def test_generate_sends_no_auth_header(
        self, provider_unauthenticated: OpenAICompatibleProvider
    ) -> None:
        mock_client = _mock_client(post=AsyncMock(return_value=_success_response()))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider_unauthenticated.generate("Hello")

        assert "Authorization" not in mock_client.post.call_args[1]["headers"]

    @pytest.mark.asyncio
    async def test_available_without_key(
        self, provider_unauthenticated: OpenAICompatibleProvider
    ) -> None:
        mock_client = _mock_client(get=AsyncMock(return_value=_models_response(["test-model"])))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_unauthenticated.is_available() is True

        assert "Authorization" not in mock_client.get.call_args[1]["headers"]

    def test_uses_configured_timeout(self, provider_unauthenticated: OpenAICompatibleProvider) -> None:
        assert provider_unauthenticated.config.request_timeout_seconds == _TEST_TIMEOUT_SECONDS


# ============================================================================
# Reasoning Stripping Tests
# ============================================================================


class TestStripReasoning:
    """Qwen3-style <think> blocks must not leak into returned content."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("plain answer", "plain answer", id="no-think"),
            pytest.param("<think>hmm</think>\n\nanswer", "answer", id="leading-think"),
            pytest.param("<think>a\nb\nc</think>[1, 2]", "[1, 2]", id="multiline-think"),
            pytest.param("<think>never closed", "", id="truncated-think"),
            pytest.param("<think></think>answer", "answer", id="empty-think"),
        ],
    )
    def test_strip_reasoning(self, raw: str, expected: str) -> None:
        assert strip_reasoning(raw) == expected

    @pytest.mark.asyncio
    async def test_generate_strips_reasoning(
        self, provider_with_key: OpenAICompatibleProvider
    ) -> None:
        mock_client = _mock_client(
            post=AsyncMock(return_value=_success_response("<think>reasoning</think>\n{\"a\": 1}"))
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider_with_key.generate("Hello") == '{"a": 1}'


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
        assert config.request_timeout_seconds == 120.0

    def test_cloud_config_api_key_optional(self) -> None:
        """CloudModelConfig allows unauthenticated endpoints."""
        config = CloudModelConfig(provider=LLMProviderType.OPENAI, model_name="test-model")
        assert config.api_key_env_var is None


# ============================================================================
# Provider Hardening (LM Studio / cruncher behaviour)
# ============================================================================

_LMSTUDIO_CONTEXT_OVERFLOW_BODY = (
    '{"error":"Context size has been exceeded."}'
)
_RETRIES = 2
_TOTAL_ATTEMPTS = _RETRIES + 1


def _completion(
    content: str | None,
    finish_reason: str = "stop",
    usage: dict | None = None,
) -> httpx.Response:
    """A chat completion response with explicit content, finish_reason and usage."""
    body: dict = {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ]
    }
    if usage is not None:
        body["usage"] = usage
    return _mock_response(200, json_data=body)


def _hardening_provider(**overrides) -> OpenAICompatibleProvider:
    """Unauthenticated provider with fast retries and optional config overrides."""
    config = CloudModelConfig(
        provider=LLMProviderType.OPENAI,
        model_name="test-model",
        base_url="http://cruncher.test:1234",
        **overrides,
    )
    return OpenAICompatibleProvider(config, max_retries=_RETRIES, retry_delay=0.0)


class TestOptionalSamplingParams:
    """reasoning_effort and seed are sent only when configured."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("overrides", "expected_present", "expected_absent"),
        [
            pytest.param({}, {}, {"reasoning_effort", "seed"}, id="neither-set"),
            pytest.param(
                {"reasoning_effort": "none"}, {"reasoning_effort": "none"}, {"seed"}, id="effort"
            ),
            pytest.param({"seed": 42}, {"seed": 42}, {"reasoning_effort"}, id="seed"),
            pytest.param(
                {"reasoning_effort": "none", "seed": 0},
                {"reasoning_effort": "none", "seed": 0},
                set(),
                id="both-including-zero-seed",
            ),
        ],
    )
    async def test_payload(
        self, overrides: dict, expected_present: dict, expected_absent: set[str]
    ) -> None:
        provider = _hardening_provider(**overrides)
        mock_client = _mock_client(post=AsyncMock(return_value=_success_response()))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider.generate("Hello")

        payload = mock_client.post.call_args.kwargs["json"]
        for key, value in expected_present.items():
            assert payload[key] == value
        assert not expected_absent & payload.keys()


class TestEmptyContent:
    """Blank content (e.g. reasoning ate max_tokens) is an error, not a valid answer."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("content", "finish_reason"),
        [
            pytest.param("", "stop", id="empty"),
            pytest.param(None, "stop", id="null"),
            pytest.param("   \n", "stop", id="whitespace"),
            pytest.param("<think>never closed", "length", id="truncated-reasoning"),
            pytest.param("", "length", id="length-empty"),
        ],
    )
    async def test_blank_content_raises_after_retries(
        self, content: str | None, finish_reason: str
    ) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(
            post=AsyncMock(return_value=_completion(content, finish_reason))
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(OpenAIGenerationError, match="empty content"):
                await provider.generate("Hello")

        assert mock_client.post.call_count == _TOTAL_ATTEMPTS

    @pytest.mark.asyncio
    async def test_length_error_mentions_truncation(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(post=AsyncMock(return_value=_completion("", "length")))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(OpenAIGenerationError, match="max_tokens"):
                await provider.generate("Hello")

    @pytest.mark.asyncio
    async def test_blank_then_success_recovers(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(
            post=AsyncMock(side_effect=[_completion(""), _completion("real answer")])
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider.generate("Hello") == "real answer"


class TestContextOverflow:
    """LM Studio's shared-context 400 is a distinct, retryable error."""

    @pytest.mark.asyncio
    async def test_overflow_raises_distinct_error_after_retries(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(
            post=AsyncMock(return_value=_mock_response(400, text=_LMSTUDIO_CONTEXT_OVERFLOW_BODY))
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(OpenAIContextOverflowError):
                await provider.generate("Hello")

        assert mock_client.post.call_count == _TOTAL_ATTEMPTS

    @pytest.mark.asyncio
    async def test_overflow_then_success_recovers(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(
            post=AsyncMock(
                side_effect=[
                    _mock_response(400, text=_LMSTUDIO_CONTEXT_OVERFLOW_BODY),
                    _completion("fits now"),
                ]
            )
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            assert await provider.generate("Hello") == "fits now"

    @pytest.mark.asyncio
    async def test_other_400_is_not_overflow(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(
            post=AsyncMock(return_value=_mock_response(400, text='{"error":"bad schema"}'))
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(OpenAIGenerationError) as exc_info:
                await provider.generate("Hello")

        assert not isinstance(exc_info.value, OpenAIContextOverflowError)


class TestUsageTotals:
    """Token usage reported by the server is accumulated per provider."""

    @pytest.mark.asyncio
    async def test_usage_accumulates(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(
            post=AsyncMock(
                side_effect=[
                    _completion("a", usage={"prompt_tokens": 100, "completion_tokens": 7}),
                    _completion("b", usage={"prompt_tokens": 300, "completion_tokens": 9}),
                ]
            )
        )

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider.generate("one")
            await provider.generate("two")

        totals = provider.usage_totals
        assert totals.calls == 2
        assert totals.prompt_tokens == 400
        assert totals.completion_tokens == 16
        assert totals.max_prompt_tokens == 300

    @pytest.mark.asyncio
    async def test_missing_usage_still_counts_call(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(post=AsyncMock(return_value=_completion("a")))

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await provider.generate("one")

        totals = provider.usage_totals
        assert (totals.calls, totals.prompt_tokens, totals.completion_tokens) == (1, 0, 0)

    def test_starts_empty(self) -> None:
        totals = _hardening_provider().usage_totals
        assert (totals.calls, totals.prompt_tokens, totals.completion_tokens) == (0, 0, 0)


class TestSharedClient:
    """One lazily created httpx client per provider, closed via aclose()."""

    @pytest.mark.asyncio
    async def test_client_reused_across_calls(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(post=AsyncMock(return_value=_success_response()))

        with patch(
            "focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client
        ) as client_cls:
            await provider.generate("one")
            await provider.generate("two")

        assert client_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_aclose_closes_and_next_call_reopens(self) -> None:
        provider = _hardening_provider()
        mock_client = _mock_client(post=AsyncMock(return_value=_success_response()))

        with patch(
            "focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client
        ) as client_cls:
            await provider.generate("one")
            await provider.aclose()
            mock_client.aclose.assert_awaited_once()
            await provider.generate("two")

        assert client_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_aclose_without_client_is_noop(self) -> None:
        await _hardening_provider().aclose()


class TestConcurrencyLimit:
    """In-flight requests are capped by max_parallel_requests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limit", [1, 2, 4])
    async def test_peak_in_flight_never_exceeds_limit(self, limit: int) -> None:
        import asyncio

        provider = _hardening_provider(max_parallel_requests=limit)
        in_flight = 0
        peak = 0

        async def slow_post(*_args, **_kwargs) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _success_response()

        mock_client = _mock_client(post=AsyncMock(side_effect=slow_post))
        requests = limit * 3

        with patch("focus_group.llm.openai_compat.httpx.AsyncClient", return_value=mock_client):
            await asyncio.gather(*(provider.generate(f"q{i}") for i in range(requests)))

        assert peak == limit
        assert mock_client.post.call_count == requests

    def test_default_limit(self) -> None:
        assert CloudModelConfig(
            provider=LLMProviderType.OPENAI, model_name="m"
        ).max_parallel_requests == 4


class TestBuildFromSettings:
    """api.config.build_llm_provider maps the LLM_* env vars onto the provider config."""

    @pytest.mark.parametrize(
        ("env", "expected_effort", "expected_parallel"),
        [
            pytest.param({}, None, 4, id="defaults"),
            pytest.param(
                {"LLM_REASONING_EFFORT": "none", "LLM_MAX_PARALLEL": "8"}, "none", 8, id="set"
            ),
        ],
    )
    def test_maps_env(
        self, env: dict[str, str], expected_effort: str | None, expected_parallel: int
    ) -> None:
        from api.config import Settings, build_llm_provider

        settings = Settings(
            _env_file=None,
            LLM_PROVIDER="openai",
            LLM_BASE_URL="http://cruncher.test:1234/",
            LLM_MODEL="test-model",
            ENTRA_TENANT_ID="t",
            ENTRA_CLIENT_ID="c",
            ENTRA_CLIENT_SECRET="s",
            ENTRA_REDIRECT_URI="http://localhost/cb",
            FRONTEND_URL="http://localhost:3000",
            SESSION_SECRET_KEY="k",
            **env,
        )
        provider = build_llm_provider(settings)

        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.config.reasoning_effort == expected_effort
        assert provider.config.max_parallel_requests == expected_parallel
