"""API configuration via environment variables.

All runtime settings are read from environment variables (or a .env file).
Sensitive values are validated at startup — the app will not start if they
are missing.
"""

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama / inference
    OLLAMA_HOST: str
    OLLAMA_PORT: int = 11434
    OLLAMA_MODEL: str

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

    # --- derived properties ---

    @property
    def ollama_base_url(self) -> str:
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

    @property
    def entra_authority(self) -> str:
        return f"https://login.microsoftonline.com/{self.ENTRA_TENANT_ID}"

    @property
    def cors_origins_list(self) -> list[str]:
        if not self.CORS_ORIGINS:
            return []
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

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
