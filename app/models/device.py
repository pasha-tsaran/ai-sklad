"""Short-lived, incident-bound device collection capabilities; raw tokens aren't stored."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.entities import Base, Entity


class DeviceCollection(Entity, Base):
    __tablename__ = "device_collections"
    account_id: Mapped[str] = mapped_column(ForeignKey("talk_accounts.id"))
    room_id: Mapped[str] = mapped_column(ForeignKey("talk_rooms.id"))
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id"))
    service_uid: Mapped[str] = mapped_column(String(255))
    requester_uid: Mapped[str] = mapped_column(String(255))
    token_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    computer_name: Mapped[str | None] = mapped_column(String(63))
    __table_args__ = (
        UniqueConstraint("token_hash", name="uq_device_collections_token_hash"),
        Index("ix_device_collections_room_incident", "room_id", "incident_id"),
    )
