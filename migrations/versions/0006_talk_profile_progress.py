"""Store trusted chat profile and best-effort reply progress separately from dialogue."""

from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("talk_rooms", sa.Column("profile", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    op.add_column("talk_inbox", sa.Column("progress", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))


def downgrade():
    op.drop_column("talk_inbox", "progress")
    op.drop_column("talk_rooms", "profile")
