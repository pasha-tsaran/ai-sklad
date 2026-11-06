"""Queue private photo references and mark letters requiring manual attachments."""

from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("talk_inbox", sa.Column("media", sa.JSON(), nullable=False, server_default=sa.text("'{}'")))
    op.add_column("incidents", sa.Column("has_user_photos", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade():
    op.drop_column("incidents", "has_user_photos")
    op.drop_column("talk_inbox", "media")
