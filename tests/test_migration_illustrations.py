import importlib.util

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, text

from app.models import Base


def test_illustration_migration_preserves_existing_messages(tmp_path):
    expected = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name != "successful_retrievals":
            table.to_metadata(expected)
    previous = MetaData()
    for table in expected.sorted_tables:
        if table.name != "message_attachments":
            table.to_metadata(previous)
    engine = create_engine(f"sqlite:///{tmp_path / 'images-migration.db'}")
    previous.create_all(engine)
    spec = importlib.util.spec_from_file_location("images_migration", "migrations/versions/0003_illustrations.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO users (id,created_at,external_id,display_name) "
                                "VALUES ('old',CURRENT_TIMESTAMP,'old','Existing employee')"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
        assert connection.scalar(text("SELECT display_name FROM users WHERE id='old'")) == "Existing employee"
        assert compare_metadata(context, expected) == []
    engine.dispose()
