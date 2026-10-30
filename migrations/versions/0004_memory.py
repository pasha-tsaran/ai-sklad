"""Persist bounded incident context and principal-scoped confirmed resolution facts."""

from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "incident_memory",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("incident_id", sa.String(36), sa.ForeignKey("incidents.id"), nullable=False, unique=True),
        sa.Column("through_sequence", sa.Integer(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "resolution_memory",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("incident_id", sa.String(36), sa.ForeignKey("incidents.id"), nullable=False, unique=True),
        sa.Column("facts", sa.JSON(), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_resolution_memory_user_id", "resolution_memory", ["user_id"])
    op.create_index("ix_resolution_memory_expires_at", "resolution_memory", ["expires_at"])


def downgrade():
    op.drop_index("ix_resolution_memory_expires_at", table_name="resolution_memory")
    op.drop_index("ix_resolution_memory_user_id", table_name="resolution_memory")
    op.drop_table("resolution_memory")
    op.drop_table("incident_memory")
