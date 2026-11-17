"""Persist source-grounded instruction cards and original parse snapshots."""

from alembic import op
import sqlalchemy as sa

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "knowledge_index",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scope_hash", sa.String(64), nullable=False),
        sa.Column("file_id", sa.String(36), nullable=False),
        sa.Column("knowledge_id", sa.String(36), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("source_sha256", sa.String(64), nullable=False),
        sa.Column("parser_version", sa.String(64), nullable=False),
        sa.Column("remote_revision", sa.String(64), nullable=True),
        sa.Column("error_code", sa.String(80), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("document_meta", sa.JSON(), nullable=False),
        sa.Column("candidate_meta", sa.JSON(), nullable=False),
        sa.Column("card", sa.JSON(), nullable=False),
        sa.Column("sections", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope_hash", "file_id"),
    )
    op.create_index("ix_knowledge_index_scope_hash", "knowledge_index", ["scope_hash"])
    op.create_table(
        "knowledge_index_images",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("index_id", sa.String(36), sa.ForeignKey("knowledge_index.id", ondelete="CASCADE"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("caption", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.UniqueConstraint("index_id", "number"),
    )
    op.create_index("ix_knowledge_index_images_index_id", "knowledge_index_images", ["index_id"])


def downgrade():
    op.drop_index("ix_knowledge_index_images_index_id", table_name="knowledge_index_images")
    op.drop_table("knowledge_index_images")
    op.drop_index("ix_knowledge_index_scope_hash", table_name="knowledge_index")
    op.drop_table("knowledge_index")
