"""Durable account-mode Talk ingestion, scoped to one authenticated server user."""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.entities import Base, Entity, now


class TalkAccount(Entity, Base):
    __tablename__ = "talk_accounts"
    server_url: Mapped[str] = mapped_column(String(1024))
    username: Mapped[str] = mapped_column(String(255))  # Actual /cloud/user UID, not a login alias.
    activated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    __table_args__ = (UniqueConstraint("server_url", "username"),)


class TalkRoom(Entity, Base):
    __tablename__ = "talk_rooms"
    account_id: Mapped[str] = mapped_column(ForeignKey("talk_accounts.id"))
    room_token: Mapped[str] = mapped_column(String(255))
    employee_id: Mapped[str] = mapped_column(String(255))
    cursor: Mapped[int] = mapped_column(BigInteger, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    flow: Mapped[dict] = mapped_column(JSON, default=dict)
    profile: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
    incident_id: Mapped[str | None] = mapped_column(ForeignKey("incidents.id"))
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    error_code: Mapped[str | None] = mapped_column(String(100))
    __table_args__ = (UniqueConstraint("account_id", "room_token"), CheckConstraint("cursor >= 0"))


class TalkInbox(Entity, Base):
    __tablename__ = "talk_inbox"
    room_id: Mapped[str] = mapped_column(ForeignKey("talk_rooms.id"))
    remote_message_id: Mapped[str] = mapped_column(String(100))
    remote_sequence: Mapped[int] = mapped_column(BigInteger)
    sender_id: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    tries: Mapped[int] = mapped_column(default=0)
    error_code: Mapped[str | None] = mapped_column(String(100))
    __table_args__ = (
        UniqueConstraint("room_id", "remote_message_id"),
        CheckConstraint("remote_sequence > 0"),
        CheckConstraint("status IN ('PENDING','RUNNING','DONE','FAILED','IGNORED')"),
        Index("ix_talk_inbox_status_sequence", "status", "remote_sequence"),
    )
    progress: Mapped[dict] = mapped_column(JSON, default=dict, server_default=text("'{}'"))
