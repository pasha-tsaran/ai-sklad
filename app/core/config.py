from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://support:local-demo-db@localhost:5432/support"
    api_token: SecretStr = SecretStr("local-demo-token")
    llm_backend: Literal["mock", "openwebui"] = "mock"
    openwebui_base_url: str = "http://host.docker.internal:3000"
    openwebui_api_key: SecretStr = SecretStr("")
    openwebui_model: str = ""
    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)
    seed_demo_knowledge: bool = True
    support_email: str = "support@example.invalid"
    rag_min_score: float = Field(default=0.30, ge=0, le=1)

    @model_validator(mode="after")
    def validate_credentials(self):
        if not self.api_token.get_secret_value():
            raise ValueError("API_TOKEN must not be empty")
        if self.llm_backend == "openwebui" and (
            not self.openwebui_model or not self.openwebui_api_key.get_secret_value()
        ):
            raise ValueError("Open WebUI requires OPENWEBUI_MODEL and OPENWEBUI_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
