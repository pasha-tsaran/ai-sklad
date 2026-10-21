import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now():
    return datetime.now(timezone.utc)


def new_id():
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Entity:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class User(Entity, Base):
    __tablename__ = "users"
    external_id: Mapped[str] = mapped_column(String(255), unique=True)
    display_name: Mapped[str] = mapped_column(String(255))


class Conversation(Entity, Base):
    __tablename__ = "conversations"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True)
    channel: Mapped[str] = mapped_column(String(32), default="mock_nextcloud")
    room_token: Mapped[str | None] = mapped_column(String(255))


class Incident(Entity, Base):
    __tablename__ = "incidents"
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    state: Mapped[str] = mapped_column(String(32), default="NEW")
    subject: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    problem_type: Mapped[str | None] = mapped_column(String(100))
    equipment_model: Mapped[str | None] = mapped_column(String(255))
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[int] = mapped_column(Integer, default=0)
    pending_action: Mapped[str | None] = mapped_column(String(64))
    result: Mapped[str | None] = mapped_column(String(32))
    escalation_reason: Mapped[str | None] = mapped_column(String(64))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        CheckConstraint("failed_attempts BETWEEN 0 AND 5"),
        CheckConstraint(
            "state IN ('NEW','DIAGNOSING','SOLUTION_PROPOSED','WAITING_FEEDBACK','RESOLVED','ESCALATED')"
        ),
        Index(
            "uq_active_incident",
            "conversation_id",
            unique=True,
            postgresql_where=text("state NOT IN ('RESOLVED','ESCALATED')"),
            sqlite_where=text("state NOT IN ('RESOLVED','ESCALATED')"),
        ),
    )


class Message(Entity, Base):
    __tablename__ = "messages"
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    delivery_status: Mapped[str] = mapped_column(String(32), default="RECEIVED")
    __table_args__ = (UniqueConstraint("incident_id", "sequence"),)


class KnowledgeDocument(Entity, Base):
    __tablename__ = "knowledge_documents"
    title: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer, default=1)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)


class KnowledgeChunk(Entity, Base):
    __tablename__ = "knowledge_chunks"
    document_id: Mapped[str] = mapped_column(ForeignKey("knowledge_documents.id"))
    content: Mapped[str] = mapped_column(Text)
    search_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list] = mapped_column(Vector(128).with_variant(JSON(), "sqlite"))
    embedding_model: Mapped[str] = mapped_column(String(100), default="demo-hash-v1")


class Solution(Entity, Base):
    __tablename__ = "solutions"
    chunk_id: Mapped[str] = mapped_column(ForeignKey("knowledge_chunks.id"))
    title: Mapped[str] = mapped_column(String(255))
    problem_type: Mapped[str] = mapped_column(String(100))
    steps: Mapped[list] = mapped_column(JSON)
    version: Mapped[int] = mapped_column(Integer, default=1)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class SolutionAttempt(Entity, Base):
    __tablename__ = "solution_attempts"
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"))
    solution_id: Mapped[str] = mapped_column(ForeignKey("solutions.id"))
    ordinal: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="WAITING_FEEDBACK")
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"))
    evidence: Mapped[dict] = mapped_column(JSON)
    retrieval_score: Mapped[float] = mapped_column(Float)
    llm_backend: Mapped[str] = mapped_column(String(255))
    __table_args__ = (
        UniqueConstraint("incident_id", "ordinal"),
        UniqueConstraint("incident_id", "solution_id"),
        CheckConstraint("ordinal BETWEEN 1 AND 5"),
        Index(
            "uq_pending_attempt",
            "incident_id",
            unique=True,
            postgresql_where=text("status = 'WAITING_FEEDBACK'"),
            sqlite_where=text("status = 'WAITING_FEEDBACK'"),
        ),
    )


class Feedback(Entity, Base):
    __tablename__ = "feedback"
    attempt_id: Mapped[str] = mapped_column(ForeignKey("solution_attempts.id"), unique=True)
    helped: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str] = mapped_column(Text, default="")


class Escalation(Entity, Base):
    __tablename__ = "escalations"
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    reason: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSON)
    email: Mapped[dict] = mapped_column(JSON)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Job(Entity, Base):
    __tablename__ = "jobs"
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"))
    kind: Mapped[str] = mapped_column(String(32))
    expected_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    tries: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(100))


class RequestReceipt(Entity, Base):
    __tablename__ = "request_receipts"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    key: Mapped[str] = mapped_column(String(100))
    fingerprint: Mapped[str] = mapped_column(String(64))
    response: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("user_id", "key"),)


class AuditEvent(Entity, Base):
    __tablename__ = "audit_events"
    incident_id: Mapped[str | None] = mapped_column(ForeignKey("incidents.id"), index=True)
    event: Mapped[str] = mapped_column(String(100))
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class Delivery(Entity, Base):
    """Durable Talk outbox; SENDING is never automatically retried after a crash."""

    __tablename__ = "deliveries"
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), unique=True)
    room_token: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    tries: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(100))
    remote_id: Mapped[str | None] = mapped_column(String(100))
