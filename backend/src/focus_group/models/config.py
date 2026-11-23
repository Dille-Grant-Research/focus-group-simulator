"""Configuration models for the focus group simulator."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class OllamaModelConfig(BaseModel):
    """Configuration for Ollama local models."""

    model_name: str = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        description="Name of the Ollama model to use",
    )
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama API",
    )
    num_ctx: int = Field(
        default=8192,
        description="Context window size in tokens",
        ge=1024,
        le=131072,
    )
    num_gpu: int = Field(
        default=-1,
        description="Number of GPU layers to use (-1 = all)",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=0.9,
        description="Top-p (nucleus) sampling",
        ge=0.0,
        le=1.0,
    )


class CloudModelConfig(BaseModel):
    """Configuration for cloud LLM providers (OpenAI-compatible APIs)."""

    provider: LLMProviderType = Field(
        description="The cloud provider type",
    )
    model_name: str = Field(
        description="Name of the model to use",
    )
    api_key_env_var: str | None = Field(
        default=None,
        description=(
            "Name of environment variable containing the API key. "
            "None for unauthenticated endpoints (e.g. a self-hosted LM Studio server)."
        ),
    )
    base_url: str = Field(
        default="https://api.openai.com",
        description="Base URL for the API (e.g., https://api.deepseek.com)",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in the response",
        ge=1,
        le=32768,
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=0.9,
        description="Top-p (nucleus) sampling",
        ge=0.0,
        le=1.0,
    )
    num_ctx: int = Field(
        default=8192,
        description="Context window size in tokens",
        ge=1024,
        le=131072,
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        description="Per-request HTTP timeout for chat completions",
        gt=0,
    )


class ModelConfig(BaseModel):
    """Combined model configuration with primary and optional fallback."""

    primary: OllamaModelConfig = Field(
        default_factory=OllamaModelConfig,
        description="Primary model configuration (Ollama)",
    )
    fallback: CloudModelConfig | None = Field(
        default=None,
        description="Optional fallback cloud model configuration",
    )
    use_fallback_on_error: bool = Field(
        default=True,
        description="Whether to use fallback model when primary fails",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=0,
        le=10,
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        description="Initial delay between retries (exponential backoff)",
        ge=0.1,
        le=60.0,
    )


class ContextConfig(BaseModel):
    """Configuration for context window management."""

    max_tokens: int = Field(
        default=6000,
        description="Maximum total tokens for context",
        ge=1000,
        le=128000,
    )
    summarize_after_tokens: int = Field(
        default=4000,
        description="Token threshold to trigger summarization",
        ge=500,
        le=128000,
    )
    reserved_for_persona: int = Field(
        default=500,
        description="Tokens reserved for persona description",
        ge=100,
        le=2000,
    )
    reserved_for_question: int = Field(
        default=200,
        description="Tokens reserved for current question",
        ge=50,
        le=500,
    )
    reserved_for_recent_response: int = Field(
        default=1000,
        description="Tokens reserved for most recent response",
        ge=200,
        le=4000,
    )


class MemoryConfig(BaseModel):
    """Configuration for the three-tier memory system.

    When provided in SessionConfig, enables structured memory for participants
    instead of raw context accumulation. When absent (None), the system
    behaves identically to the pre-memory baseline.

    Attributes:
        working_memory_window: Max recent turns retained in working memory
        enable_stance_tracking: Whether Research Assistant tracks stances
        question_history_limit: Max past questions included in prompt context
    """

    working_memory_window: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of recent turns in working memory",
    )
    enable_stance_tracking: bool = Field(
        default=True,
        description="Whether to enable Research Assistant stance tracking",
    )
    question_history_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum past questions included in prompt context",
    )


class OutputConfig(BaseModel):
    """Configuration for output files."""

    directory: str = Field(
        default="./output",
        description="Output directory for logs and transcripts",
    )
    save_debug_log: bool = Field(
        default=True,
        description="Whether to save full debug log with non-verbals",
    )
    save_clean_transcript: bool = Field(
        default=True,
        description="Whether to save clean transcript (verbal only)",
    )
    log_format: str = Field(
        default="jsonl",
        description="Format for log files (jsonl, yaml, or json)",
        pattern="^(jsonl|yaml|json)$",
    )


class SessionConfig(BaseModel):
    """Configuration for a focus group session.

    This is the main configuration object that defines all parameters
    for running a focus group simulation.
    """

    demographics: list[str] = Field(
        description="List of demographic specifications for participants",
        min_length=1,
        max_length=15,
    )
    participant_names: list[str | None] | None = Field(
        default=None,
        description="Optional name overrides per participant, parallel to demographics. None entries = autogenerate.",
        max_length=15,
    )
    questions: list[str] = Field(
        description="List of focus group questions to discuss (in order)",
        min_length=1,
        max_length=20,
    )
    purpose: str | None = Field(
        default=None,
        description="Optional introductory statement/purpose for the focus group",
    )
    closing_question: str = Field(
        default="Do you have any concluding thoughts about participating in this focus group?",
        description="The closing question to ask at the end",
    )
    model: OllamaModelConfig | CloudModelConfig = Field(
        default_factory=OllamaModelConfig,
        description="Model configuration for the LLM provider",
    )

    @model_validator(mode="before")
    @classmethod
    def _route_model_config(cls, data: Any) -> Any:
        """Route model config to correct type based on 'provider' key."""
        if isinstance(data, dict) and "model" in data:
            model_data = data["model"]
            if isinstance(model_data, dict) and "provider" in model_data:
                data["model"] = CloudModelConfig(**model_data)
        return data

    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context window management configuration",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output file configuration",
    )
    memory: MemoryConfig | None = Field(
        default=None,
        description="Optional memory config. None = legacy behavior.",
    )
    validate_personas: bool = Field(
        default=True,
        description="Whether to validate generated personas",
    )
    exclude_nonverbals: bool = Field(
        default=True,
        description="Strip non-verbal reactions from the stored result. "
                    "The simulation still generates them internally for dual-context purposes.",
    )
    session_name: str | None = Field(
        default=None,
        description="Optional name for the session (used in output filenames)",
    )
