"""Versioned source-derived instruction cards and verified parse snapshots."""

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.entities import Base, Entity, now


class KnowledgeIndex(Entity, Base):
    __tablename__ = "knowledge_index"
    scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    file_id: Mapped[str] = mapped_column(String(36))
    knowledge_id: Mapped[str] = mapped_column(String(36))
    filename: Mapped[str] = mapped_column(String(255))
    source_sha256: Mapped[str] = mapped_column(String(64))
    parser_version: Mapped[str] = mapped_column(String(64))
    remote_revision: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(80))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[dict] = mapped_column(JSON)
    document_meta: Mapped[dict] = mapped_column(JSON)
    candidate_meta: Mapped[dict] = mapped_column(JSON)
    card: Mapped[dict] = mapped_column(JSON)
    sections: Mapped[list] = mapped_column(JSON)
    evidence: Mapped[dict] = mapped_column(JSON)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    __table_args__ = (UniqueConstraint("scope_hash", "file_id"),)


class KnowledgeIndexImage(Entity, Base):
    __tablename__ = "knowledge_index_images"
    index_id: Mapped[str] = mapped_column(ForeignKey("knowledge_index.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(255))
    caption: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(100))
    sha256: Mapped[str] = mapped_column(String(64))
    data: Mapped[bytes] = mapped_column(LargeBinary)
    __table_args__ = (UniqueConstraint("index_id", "number"),)
