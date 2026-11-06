"""Add optional one-use device collection without changing existing conversations."""

from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "device_collections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("account_id", sa.String(36), sa.ForeignKey("talk_accounts.id"), nullable=False),
        sa.Column("room_id", sa.String(36), sa.ForeignKey("talk_rooms.id"), nullable=False),
        sa.Column("incident_id", sa.String(36), sa.ForeignKey("incidents.id"), nullable=False),
        sa.Column("service_uid", sa.String(255), nullable=False),
        sa.Column("requester_uid", sa.String(255), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("computer_name", sa.String(63), nullable=True),
        sa.UniqueConstraint("token_hash", name="uq_device_collections_token_hash"),
    )
    op.create_index("ix_device_collections_room_incident", "device_collections", ["room_id", "incident_id"])


def downgrade():
    op.drop_index("ix_device_collections_room_incident", table_name="device_collections")
    op.drop_table("device_collections")
