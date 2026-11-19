"""Allow Talk to keep the history of more than five solutions."""

from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def _replace(table, name, previous, current):
    if op.get_context().dialect.name == "sqlite":
        definition = sa.Table(table, sa.MetaData(), autoload_with=op.get_bind())
        constraint = next(item for item in definition.constraints if isinstance(item, sa.CheckConstraint)
                          and str(item.sqltext).strip() == previous)
        constraint.name = name
        with op.batch_alter_table(table, copy_from=definition, recreate="always") as batch:
            batch.drop_constraint(name, type_="check")
            batch.create_check_constraint(name, current)
    else:
        op.drop_constraint(name, table, type_="check")
        op.create_check_constraint(name, table, current)


def upgrade():
    _replace("incidents", "incidents_failed_attempts_check", "failed_attempts BETWEEN 0 AND 5", "failed_attempts >= 0")
    _replace("solution_attempts", "solution_attempts_ordinal_check", "ordinal BETWEEN 1 AND 5", "ordinal >= 1")


def downgrade():
    _replace("solution_attempts", "solution_attempts_ordinal_check", "ordinal >= 1", "ordinal BETWEEN 1 AND 5")
    _replace("incidents", "incidents_failed_attempts_check", "failed_attempts >= 0", "failed_attempts BETWEEN 0 AND 5")
