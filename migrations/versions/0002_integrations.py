"""Talk routing and durable outbound delivery. Existing incidents are preserved."""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("conversations", sa.Column("room_token", sa.String(255), nullable=True))
    op.create_table(
        "deliveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("message_id", sa.String(36), sa.ForeignKey("messages.id"), nullable=False, unique=True),
        sa.Column("room_token", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("tries", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(100)),
        sa.Column("remote_id", sa.String(100)),
    )
    op.create_index("ix_deliveries_status", "deliveries", ["status"])


def downgrade():
    op.drop_table("deliveries")
    op.drop_column("conversations", "room_token")
