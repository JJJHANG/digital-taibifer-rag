"""Validated database settings; Compose supplies environment variables."""

from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    postgres_host: str = "db"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "taibifer"
    postgres_user: str = "taibifer"
    postgres_password: SecretStr

    @property
    def database_url(self) -> URL:
        """Build a URL without manually escaping credentials."""
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


class EmbeddingSettings(BaseSettings):
    """Separate from DB settings: API startup does not require an API key."""

    model_config = SettingsConfigDict(extra="ignore")

    openai_api_key: SecretStr | None = None
    embedding_model: Literal["text-embedding-3-small"] = "text-embedding-3-small"
    # Environment values arrive as strings; parse as int before enforcing
    # the one dimension supported by the existing VECTOR(1536) schema.
    embedding_dimensions: int = Field(default=1536, ge=1536, le=1536)
    embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    embedding_timeout_seconds: float = Field(default=30, gt=0)
    embedding_max_retries: int = Field(default=2, ge=0, le=5)


class ChatSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openai_api_key: SecretStr | None = None
    chat_model: str = Field(default="gpt-6-luna", min_length=1)
    chat_tokenizer: Literal["cl100k_base", "o200k_base"] = "o200k_base"
    chat_max_input_tokens: int = Field(default=8000, gt=0)
    chat_max_output_tokens: int = Field(default=1500, ge=16, le=128000)
    chat_context_window: int = Field(default=1050000, gt=0)
    chat_framing_token_reserve: int = Field(default=256, ge=0)
    chat_timeout_seconds: float = Field(default=60, gt=0)
    chat_max_retries: int = Field(default=2, ge=0, le=5)
    chat_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "none"

    @model_validator(mode="after")
    def check_context_allowance(self) -> "ChatSettings":
        allowance = self.chat_max_input_tokens + self.chat_max_output_tokens + self.chat_framing_token_reserve
        if allowance > self.chat_context_window:
            raise ValueError("Chat input, output and framing allowance exceed the configured context window")
        return self
