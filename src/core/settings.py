import os
from enum import StrEnum
from json import loads
from typing import Annotated, Any, List

from pathlib import Path

from functools import lru_cache

from dotenv import find_dotenv
from pydantic import (
    BeforeValidator,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    # AnthropicModelName,
    # AWSModelName,
    AzureOpenAIModelName,
    # DeepseekModelName,
    FakeModelName,
    # GoogleModelName,
    GroqModelName,
    OllamaModelName,
    # OpenAICompatibleName,
    OpenAIModelName,
    # OpenRouterModelName,
    Provider,
    # VertexAIModelName,
)


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"
    MONGO = "mongo"


def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    MODE: str | None = None

    HOST: str = "http://localhost"  # "0.0.0.0"
    PORT: int = 8080

    AUTH_SECRET: SecretStr | None = None

    WHITELIST: List[str] = Field(default_factory=list, env="WHITELIST")

    # —— Entra ID OAuth2 settings ——
    AZURE_AD_TENANT_ID: str | None = None
    AZURE_AD_CLIENT_ID: str | None = None
    AZURE_AD_CLIENT_SECRET: SecretStr | None = None
    AZURE_AD_API_CLIENT_ID: str | None = None  # used in SCOPE: api://<API_CLIENT_ID>/access_as_user
    AUTH_ENABLED: bool = True
    # ——

    OPENAI_API_KEY: SecretStr | None = None
    # DEEPSEEK_API_KEY: SecretStr | None = None
    # ANTHROPIC_API_KEY: SecretStr | None = None
    # GOOGLE_API_KEY: SecretStr | None = None
    # GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    # USE_AWS_BEDROCK: bool = False
    OLLAMA_MODEL: str | None = None
    OLLAMA_BASE_URL: str | None = None
    USE_FAKE_MODEL: bool = False
    # OPENROUTER_API_KEY: str | None = None

    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    # Set openai compatible api, mainly used for proof of concept
    COMPATIBLE_MODEL: str | None = None
    COMPATIBLE_API_KEY: SecretStr | None = None
    COMPATIBLE_BASE_URL: str | None = None

    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT: str = "default"
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    LANGFUSE_TRACING: bool = False
    LANGFUSE_HOST: Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # Database Configuration
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # Options: DatabaseType.SQLITE or DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "checkpoints.db"

    # PostgreSQL Configuration
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_DB: str | None = None
    POSTGRES_APPLICATION_NAME: str = "agent-service-toolkit"
    POSTGRES_MIN_CONNECTIONS_PER_POOL: int = 1
    POSTGRES_MAX_CONNECTIONS_PER_POOL: int = 1
    PGVECTOR_URL: str | None = None

    # MongoDB Configuration
    MONGO_HOST: str | None = None
    MONGO_PORT: int | None = None
    MONGO_DB: str | None = None
    MONGO_USER: str | None = None
    MONGO_PASSWORD: SecretStr | None = None
    MONGO_AUTH_SOURCE: str | None = None

    # Azure OpenAI Settings
    AZURE_OPENAI_API_KEY: SecretStr | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_EMBEDDER: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str] = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    # LLM Settings
    TEMPERATURE: float = 0.05


    # -- Uploads --
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_MB: int = 100
    ALLOWED_MIME_TYPES: set[str] = {
        # docs
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        # images
        "image/png",
        "image/jpeg",
        "image/webp",
        # code-ish
        "application/x-python-code",
    }
    AUTO_INGEST_UPLOADS: bool = False



    def model_post_init(self, __context: Any) -> None:
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            # Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            # Provider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            # Provider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            # Provider.GOOGLE: self.GOOGLE_API_KEY,
            # Provider.VERTEXAI: self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ: self.GROQ_API_KEY,
            # Provider.AWS: self.USE_AWS_BEDROCK,
            Provider.OLLAMA: self.OLLAMA_MODEL,
            Provider.FAKE: self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI: self.AZURE_OPENAI_API_KEY,
            # Provider.OPENROUTER: self.OPENROUTER_API_KEY,
        }
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")

        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                # case Provider.OPENAI_COMPATIBLE:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                #     self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                # case Provider.DEEPSEEK:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                #     self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                # case Provider.ANTHROPIC:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = AnthropicModelName.HAIKU_3
                #     self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                # case Provider.GOOGLE:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                #     self.AVAILABLE_MODELS.update(set(GoogleModelName))
                # case Provider.VERTEXAI:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                #     self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                # case Provider.AWS:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                #     self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                # case Provider.OPENROUTER:
                #     if self.DEFAULT_MODEL is None:
                #         self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                #     self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # Validate Azure OpenAI settings if Azure provider is available
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # Parse deployment map if it's a string
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # Validate required deployments exist
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    raise ValueError(f"Unknown provider: {provider}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def JWKS_URL(self) -> str | None:
        if self.AZURE_AD_TENANT_ID:
            return f"https://login.microsoftonline.us/{self.AZURE_AD_TENANT_ID}/discovery/v2.0/keys"
        return None

    def is_dev(self) -> bool:
        return self.MODE == "dev"


settings = Settings()

# ensure upload root exists eagerly
Path(settings.UPLOAD_DIR).resolve().mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1)
def get_settings() -> Settings:  # pragma: no cover
    """Return a cached instance so settings are evaluated once only."""

    return Settings()
