"""API configuration via environment variables.

All runtime settings are read from environment variables (or a .env file).
Sensitive values are validated at startup — the app will not start if they
are missing.
"""

import secrets
from dataclasses import dataclass
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from focus_group.llm.base import BaseLLMProvider
from focus_group.llm.factory import create_provider
from focus_group.models.config import (
    DEFAULT_MAX_PARALLEL_REQUESTS,
    CloudModelConfig,
    LLMProviderType,
    OllamaModelConfig,
)

DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 300
DEFAULT_LLM_MAX_TOKENS = 2048
DEFAULT_LLM_NUM_CTX = 8192
# Server-chosen seeds stay in the signed 32-bit range every backend accepts.
_SEED_UPPER_BOUND = 2**31 - 1
SUPPORTED_LLM_PROVIDERS = (LLMProviderType.OLLAMA, LLMProviderType.OPENAI)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Inference
    # "ollama" -> native Ollama API (/api/generate)
    # "openai" -> any OpenAI-compatible server (/v1/chat/completions), e.g. MSUM cruncher
    LLM_PROVIDER: LLMProviderType
    LLM_BASE_URL: str  # scheme://host:port, no trailing /v1
    LLM_MODEL: str
    LLM_REQUEST_TIMEOUT_SECONDS: int = DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    # OpenAI-compatible only. "none" stops reasoning models (Qwen3 on cruncher)
    # from spending max_tokens on reasoning and returning empty content.
    LLM_REASONING_EFFORT: str | None = None
    # OpenAI-compatible only. Max chat completions in flight at once. Cruncher's
    # loaded context is one pool shared by concurrent requests, so keep this low.
    LLM_MAX_PARALLEL: int = DEFAULT_MAX_PARALLEL_REQUESTS
    # Default and ceiling for max_tokens per reply. Submitted values above it are
    # lowered to it, since cruncher's context is shared by all requests in flight.
    LLM_MAX_TOKENS: int = DEFAULT_LLM_MAX_TOKENS
    # The context size the server has loaded (cruncher: 8192 today). Used for
    # near-limit warnings and context budgeting; it does not change the server.
    LLM_NUM_CTX: int = DEFAULT_LLM_NUM_CTX

    # Microsoft Entra SSO
    ENTRA_TENANT_ID: str
    ENTRA_CLIENT_ID: str
    ENTRA_CLIENT_SECRET: str
    ENTRA_REDIRECT_URI: str

    # App secrets & origins
    FRONTEND_URL: str
    SESSION_SECRET_KEY: str
    CORS_ORIGINS: str = ""  # comma-separated list, parsed via property below

    # Data & job settings
    DB_PATH: str = "/data/focusgroup.db"
    MAX_JOBS_PER_HOUR: int = 5
    JOB_TIMEOUT_SECONDS: int = 600

    # Logging
    LOG_LEVEL: str = "INFO"

    # Dev-only: bypass MSAL entirely and inject a fixed local user_id.
    # NEVER set this to true in production.
    DEV_AUTH_BYPASS: bool = False

    # --- derived properties ---

    @property
    def llm_base_url(self) -> str:
        return self.LLM_BASE_URL.rstrip("/")

    @property
    def entra_authority(self) -> str:
        return f"https://login.microsoftonline.com/{self.ENTRA_TENANT_ID}"

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS:
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @model_validator(mode="after")
    def _check_llm_provider(self) -> "Settings":
        if self.LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:
            supported = ", ".join(p.value for p in SUPPORTED_LLM_PROVIDERS)
            raise ValueError(f"LLM_PROVIDER must be one of: {supported}")
        return self

    @model_validator(mode="after")
    def _require_secrets(self) -> "Settings":
        if not self.ENTRA_CLIENT_SECRET:
            raise ValueError("ENTRA_CLIENT_SECRET must not be empty")
        if not self.SESSION_SECRET_KEY:
            raise ValueError("SESSION_SECRET_KEY must not be empty")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


@dataclass(frozen=True)
class SamplingParams:
    """Sampling settings for one job: what was submitted, capped and defaulted by the server."""

    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None


_DEFAULT_TEMPERATURE: float = CloudModelConfig.model_fields["temperature"].default
_DEFAULT_TOP_P: float = CloudModelConfig.model_fields["top_p"].default


def resolve_sampling(
    settings: Settings, submitted: OllamaModelConfig | CloudModelConfig | None
) -> SamplingParams:
    """Apply a job's submitted sampling settings over server defaults.

    Only fields the submitter actually set are honoured. Model, URL and context
    size stay server-controlled. max_tokens is capped at LLM_MAX_TOKENS, and a
    seed is chosen by the server when none is submitted so every run is reproducible.
    """
    given = submitted.model_fields_set if submitted is not None else set()

    def _pick(name: str, default):  # noqa: ANN001, ANN202
        value = getattr(submitted, name, None) if name in given else None
        return default if value is None else value

    seed = _pick("seed", None)
    return SamplingParams(
        temperature=_pick("temperature", _DEFAULT_TEMPERATURE),
        top_p=_pick("top_p", _DEFAULT_TOP_P),
        max_tokens=min(_pick("max_tokens", settings.LLM_MAX_TOKENS), settings.LLM_MAX_TOKENS),
        seed=seed if seed is not None else secrets.randbelow(_SEED_UPPER_BOUND),
    )


def build_llm_provider(
    settings: Settings, sampling: SamplingParams | None = None
) -> BaseLLMProvider:
    """Build the inference provider used by the worker, /generate and /health.

    Without sampling (health checks, /generate) server defaults apply and no seed is sent.
    """
    if sampling is None:
        sampling = SamplingParams(
            temperature=_DEFAULT_TEMPERATURE,
            top_p=_DEFAULT_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS,
            seed=None,
        )
    if settings.LLM_PROVIDER == LLMProviderType.OPENAI:
        return create_provider(
            CloudModelConfig(
                provider=LLMProviderType.OPENAI,
                model_name=settings.LLM_MODEL,
                base_url=settings.llm_base_url,
                request_timeout_seconds=settings.LLM_REQUEST_TIMEOUT_SECONDS,
                reasoning_effort=settings.LLM_REASONING_EFFORT or None,
                max_parallel_requests=settings.LLM_MAX_PARALLEL,
                num_ctx=settings.LLM_NUM_CTX,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed,
            )
        )
    return create_provider(
        OllamaModelConfig(
            model_name=settings.LLM_MODEL,
            base_url=settings.llm_base_url,
            num_ctx=settings.LLM_NUM_CTX,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )
    )
