import importlib.util

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, text

from app.models import Base


def test_additive_migration_preserves_conversation(tmp_path):
    # Construct the prior schema from the unchanged model fields, then apply the actual migration.
    old = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name != "deliveries":
            table.to_metadata(old)
    old.tables["conversations"]._columns.remove(old.tables["conversations"].c.room_token)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    old.create_all(engine)
    spec = importlib.util.spec_from_file_location("migration", "migrations/versions/0002_integrations.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (id,created_at,external_id,display_name) "
                "VALUES ('u',CURRENT_TIMESTAMP,'old-user','Old user')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO conversations (id,created_at,user_id,channel) "
                "VALUES ('c',CURRENT_TIMESTAMP,'u','mock_nextcloud')"
            )
        )
        context = MigrationContext.configure(conn)
        with Operations.context(context):
            migration.upgrade()
        assert conn.scalar(text("SELECT user_id FROM conversations WHERE id='c'")) == "u"
        assert compare_metadata(context, Base.metadata) == []
    engine.dispose()
