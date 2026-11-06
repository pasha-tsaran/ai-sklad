from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)
    database_url: str = "postgresql+psycopg://support:local-demo-db@localhost:5432/support"
    api_token: SecretStr = SecretStr("local-demo-token")
    llm_backend: Literal["mock", "openwebui"] = "mock"
    openwebui_base_url: str = "https://ai.collant.ru"
    openwebui_api_key: SecretStr = SecretStr("")
    openwebui_model: str = "ai-support"
    knowledge_backend: Literal["openwebui", "demo"] = "openwebui"
    openwebui_knowledge_ids: list[str] = Field(default_factory=list)
    knowledge_max_chars: int = Field(default=200000, ge=1000, le=200000)
    knowledge_max_file_bytes: int = Field(default=30_000_000, ge=1000, le=50_000_000)
    nextcloud_backend: Literal["mock", "talk", "account"] = "mock"
    nextcloud_base_url: str = ""
    nextcloud_username: str = ""
    nextcloud_app_password: SecretStr = SecretStr("")
    nextcloud_poll_interval_seconds: float = Field(default=3, ge=1, le=60)
    nextcloud_batch_size: int = Field(default=100, ge=1, le=100)
    device_collection_base_url: str = ""
    nextcloud_bot_secret: SecretStr = SecretStr("")
    nextcloud_allowed_rooms: list[str] = Field(default_factory=list)
    nextcloud_files_username: str = ""
    nextcloud_files_app_password: SecretStr = SecretStr("")
    integration_timeout_seconds: float = Field(default=20, gt=0, le=120)
    mail_backend: Literal["mock", "smtp", "mailto"] = "mock"
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
    memory_max_messages: int = Field(default=12, ge=2, le=40)
    memory_max_chars: int = Field(default=12000, ge=2000, le=24000)
    memory_retention_days: int = Field(default=90, ge=1, le=365)

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
        if self.nextcloud_backend == "account":
            parsed = urlsplit(self.nextcloud_base_url)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or any(c.isspace() for c in self.nextcloud_base_url)):
                raise ValueError("Talk account requires an HTTPS URL without credentials, query or fragment")
            if (not self.nextcloud_username.strip() or len(self.nextcloud_username) > 255
                    or any(ord(c) < 32 for c in self.nextcloud_username)
                    or not self.nextcloud_app_password.get_secret_value()
                    or any(c in self.nextcloud_app_password.get_secret_value() for c in "\r\n")):
                raise ValueError("Talk account requires a login and a separate app password")
        if self.device_collection_base_url:
            base = urlsplit(self.device_collection_base_url)
            if (self.nextcloud_backend != "account" or len(self.device_collection_base_url) > 2048
                    or base.scheme != "https" or not base.hostname or base.username or base.password
                    or base.query or base.fragment or "\\" in self.device_collection_base_url
                    or any(c.isspace() or ord(c) < 32 for c in self.device_collection_base_url)
                    or any(part in {".", ".."} for part in base.path.split("/"))):
                raise ValueError("Device collection requires an account-mode HTTPS API base URL")
        if bool(self.nextcloud_files_username) != bool(self.nextcloud_files_app_password.get_secret_value()):
            raise ValueError("Nextcloud attachments require both files username and app password")
        if self.mail_backend == "smtp":
            if not self.smtp_host or not self.smtp_from or self.support_email.endswith(".invalid"):
                raise ValueError("SMTP requires host, from and a real support address")
            for address in (self.smtp_from, self.support_email):
                if "@" not in address or any(c in address for c in "\r\n,;"):
                    raise ValueError("Configure a single valid mailbox address")
        if self.mail_backend == "mailto":
            from app.services.mail.draft import validate_recipient

            validate_recipient(self.support_email)
            if self.support_email.casefold().endswith(".invalid"):
                raise ValueError("Mail drafts require a real SUPPORT_EMAIL recipient")
            if self.nextcloud_backend == "talk":
                raise ValueError("Mail drafts require account-mode Talk or the local API")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
