"""Initial immutable schema. No imports from application models."""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def table(name, *columns):
    op.create_table(
        name,
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        *columns,
    )


def col(name, type_, nullable=False, **kw):
    return sa.Column(name, type_, nullable=nullable, **kw)


def ref(name, target, unique=False):
    return sa.Column(name, sa.String(36), sa.ForeignKey(target), nullable=False, unique=unique)


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    table("users", col("external_id", sa.String(255), unique=True), col("display_name", sa.String(255)))
    table("conversations", ref("user_id", "users.id", unique=True), col("channel", sa.String(32)))
    table(
        "incidents",
        ref("conversation_id", "conversations.id"),
        col("state", sa.String(32)),
        col("subject", sa.String(255)),
        col("description", sa.Text()),
        col("problem_type", sa.String(100), True),
        col("equipment_model", sa.String(255), True),
        col("failed_attempts", sa.Integer()),
        col("version", sa.Integer()),
        col("pending_action", sa.String(64), True),
        col("result", sa.String(32), True),
        col("escalation_reason", sa.String(64), True),
        col("closed_at", sa.DateTime(timezone=True), True),
        sa.CheckConstraint("failed_attempts BETWEEN 0 AND 5"),
        sa.CheckConstraint(
            "state IN ('NEW','DIAGNOSING','SOLUTION_PROPOSED','WAITING_FEEDBACK','RESOLVED','ESCALATED')"
        ),
    )
    op.create_index(
        "uq_active_incident",
        "incidents",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("state NOT IN ('RESOLVED','ESCALATED')"),
    )
    table(
        "messages",
        ref("incident_id", "incidents.id"),
        col("sequence", sa.Integer()),
        col("role", sa.String(20)),
        col("content", sa.Text()),
        col("delivery_status", sa.String(32)),
        sa.UniqueConstraint("incident_id", "sequence"),
    )
    op.create_index("ix_messages_incident_id", "messages", ["incident_id"])
    table(
        "knowledge_documents",
        col("title", sa.String(255)),
        col("version", sa.Integer()),
        col("approved", sa.Boolean()),
        col("is_demo", sa.Boolean()),
    )
    table(
        "knowledge_chunks",
        ref("document_id", "knowledge_documents.id"),
        col("content", sa.Text()),
        col("search_text", sa.Text()),
        col("embedding", Vector(128)),
        col("embedding_model", sa.String(100)),
    )
    table(
        "solutions",
        ref("chunk_id", "knowledge_chunks.id"),
        col("title", sa.String(255)),
        col("problem_type", sa.String(100)),
        col("steps", sa.JSON()),
        col("version", sa.Integer()),
        col("priority", sa.Integer()),
        col("active", sa.Boolean()),
    )
    table(
        "solution_attempts",
        ref("incident_id", "incidents.id"),
        ref("solution_id", "solutions.id"),
        col("ordinal", sa.Integer()),
        col("status", sa.String(32)),
        ref("message_id", "messages.id"),
        col("evidence", sa.JSON()),
        col("retrieval_score", sa.Float()),
        col("llm_backend", sa.String(255)),
        sa.UniqueConstraint("incident_id", "ordinal"),
        sa.UniqueConstraint("incident_id", "solution_id"),
        sa.CheckConstraint("ordinal BETWEEN 1 AND 5"),
    )
    op.create_index(
        "uq_pending_attempt",
        "solution_attempts",
        ["incident_id"],
        unique=True,
        postgresql_where=sa.text("status = 'WAITING_FEEDBACK'"),
    )
    table(
        "feedback",
        ref("attempt_id", "solution_attempts.id", unique=True),
        col("helped", sa.Boolean()),
        col("comment", sa.Text()),
    )
    table(
        "escalations",
        ref("incident_id", "incidents.id", unique=True),
        col("status", sa.String(32)),
        col("reason", sa.String(64)),
        col("details", sa.JSON()),
        col("email", sa.JSON()),
        col("sent_at", sa.DateTime(timezone=True), True),
    )
    table(
        "jobs",
        ref("incident_id", "incidents.id"),
        col("kind", sa.String(32)),
        col("status", sa.String(32)),
        col("expected_version", sa.Integer()),
        col("tries", sa.Integer()),
        col("error_code", sa.String(100), True),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])
    table(
        "request_receipts",
        ref("user_id", "users.id"),
        col("key", sa.String(100)),
        col("fingerprint", sa.String(64)),
        col("response", sa.JSON()),
        sa.UniqueConstraint("user_id", "key"),
    )
    table(
        "audit_events",
        sa.Column("incident_id", sa.String(36), sa.ForeignKey("incidents.id")),
        col("event", sa.String(100)),
        col("details", sa.JSON()),
    )
    op.create_index("ix_audit_events_incident_id", "audit_events", ["incident_id"])


def downgrade():
    for name in (
        "audit_events",
        "request_receipts",
        "jobs",
        "escalations",
        "feedback",
        "solution_attempts",
        "solutions",
        "knowledge_chunks",
        "knowledge_documents",
        "messages",
        "incidents",
        "conversations",
        "users",
    ):
        op.drop_table(name)
    # Extension may be shared with another application; leave it installed.
