"""Remember source pointers for private, explicitly successful retrievals."""

from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "successful_retrievals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("incident_id", sa.String(36), sa.ForeignKey("incidents.id"), nullable=False),
        sa.Column("scope_hash", sa.String(64), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=False),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "scope_hash", "query_hash", "context_hash"),
    )
    op.create_index("ix_successful_retrievals_user_id", "successful_retrievals", ["user_id"])
    op.create_index("ix_successful_retrievals_expires_at", "successful_retrievals", ["expires_at"])


def downgrade():
    op.drop_index("ix_successful_retrievals_expires_at", table_name="successful_retrievals")
    op.drop_index("ix_successful_retrievals_user_id", table_name="successful_retrievals")
    op.drop_table("successful_retrievals")
