"""Private pointers to instructions explicitly confirmed as helpful."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.entities import Base, Entity


class SuccessfulRetrieval(Entity, Base):
    __tablename__ = "successful_retrievals"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"))
    scope_hash: Mapped[str] = mapped_column(String(64))
    query_hash: Mapped[str] = mapped_column(String(64))
    context_hash: Mapped[str] = mapped_column(String(64))
    source: Mapped[dict] = mapped_column(JSON)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    __table_args__ = (UniqueConstraint("user_id", "scope_hash", "query_hash", "context_hash"),)
