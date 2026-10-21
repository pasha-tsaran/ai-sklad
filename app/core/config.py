from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://support:local-demo-db@localhost:5432/support"
    api_token: SecretStr = SecretStr("local-demo-token")
    llm_backend: Literal["mock", "openwebui"] = "mock"
    openwebui_base_url: str = "https://ai.collant.ru"
    openwebui_api_key: SecretStr = SecretStr("")
    openwebui_model: str = "ai-support"
    knowledge_backend: Literal["openwebui", "demo"] = "openwebui"
    openwebui_knowledge_ids: list[str] = Field(default_factory=list)
    knowledge_max_chars: int = Field(default=20000, ge=1000, le=24000)
    nextcloud_backend: Literal["mock", "talk"] = "mock"
    nextcloud_base_url: str = ""
    nextcloud_bot_secret: SecretStr = SecretStr("")
    nextcloud_allowed_rooms: list[str] = Field(default_factory=list)
    integration_timeout_seconds: float = Field(default=20, gt=0, le=120)
    mail_backend: Literal["mock", "smtp"] = "mock"
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_security: Literal["starttls", "ssl"] = "starttls"
    smtp_username: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_from: str = ""
    llm_timeout_seconds: float = Field(default=60, gt=0, le=300)
    seed_demo_knowledge: bool = False
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
        if self.knowledge_backend == "openwebui" and self.seed_demo_knowledge:
            raise ValueError("Demo knowledge must be disabled with Open WebUI knowledge")
        if self.openwebui_knowledge_ids and not self.openwebui_api_key.get_secret_value():
            raise ValueError("Knowledge collections require OPENWEBUI_API_KEY")
        if self.nextcloud_backend == "talk":
            if (
                not self.nextcloud_base_url.startswith("https://")
                or len(self.nextcloud_bot_secret.get_secret_value()) < 32
                or not self.nextcloud_allowed_rooms
            ):
                raise ValueError("Talk requires HTTPS URL, >=32 character secret and allowed rooms")
        if self.mail_backend == "smtp":
            if not self.smtp_host or not self.smtp_from or self.support_email.endswith(".invalid"):
                raise ValueError("SMTP requires host, from and a real support address")
            for address in (self.smtp_from, self.support_email):
                if "@" not in address or any(c in address for c in "\r\n,;"):
                    raise ValueError("Configure a single valid mailbox address")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
