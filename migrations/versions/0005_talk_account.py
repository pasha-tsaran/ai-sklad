"""Persist personal Talk account activation, private rooms and ordered inbound messages."""

from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("deliveries", sa.Column("provenance", sa.JSON(), nullable=True))
    op.add_column("deliveries", sa.Column("account_id", sa.String(36), nullable=True))
    op.create_index("ix_deliveries_account_id", "deliveries", ["account_id"])
    op.create_table(
        "talk_accounts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("server_url", sa.String(1024), nullable=False),
        sa.Column("username", sa.String(255), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.UniqueConstraint("server_url", "username"),
    )
    op.create_table(
        "talk_rooms",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("account_id", sa.String(36), sa.ForeignKey("talk_accounts.id"), nullable=False),
        sa.Column("room_token", sa.String(255), nullable=False),
        sa.Column("employee_id", sa.String(255), nullable=False),
        sa.Column("cursor", sa.BigInteger(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("flow", sa.JSON(), nullable=False),
        sa.Column("incident_id", sa.String(36), sa.ForeignKey("incidents.id")),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_code", sa.String(100)),
        sa.UniqueConstraint("account_id", "room_token"),
        sa.CheckConstraint("cursor >= 0"),
    )
    op.create_table(
        "talk_inbox",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("room_id", sa.String(36), sa.ForeignKey("talk_rooms.id"), nullable=False),
        sa.Column("remote_message_id", sa.String(100), nullable=False),
        sa.Column("remote_sequence", sa.BigInteger(), nullable=False),
        sa.Column("sender_id", sa.String(255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("tries", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(100)),
        sa.UniqueConstraint("room_id", "remote_message_id"),
        sa.CheckConstraint("remote_sequence > 0"),
        sa.CheckConstraint("status IN ('PENDING','RUNNING','DONE','FAILED','IGNORED')"),
    )
    op.create_index("ix_talk_inbox_status_sequence", "talk_inbox", ["status", "remote_sequence"])


def downgrade():
    op.drop_index("ix_talk_inbox_status_sequence", table_name="talk_inbox")
    op.drop_table("talk_inbox")
    op.drop_table("talk_rooms")
    op.drop_table("talk_accounts")
    op.drop_column("deliveries", "provenance")
    op.drop_index("ix_deliveries_account_id", table_name="deliveries")
    op.drop_column("deliveries", "account_id")
