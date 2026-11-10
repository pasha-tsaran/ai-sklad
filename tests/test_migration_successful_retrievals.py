"""The private source shortcut is additive to the actual 0008 production schema."""

import importlib.util
from datetime import timedelta
from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, event, select, text

from app.models import Base, SuccessfulRetrieval, now
from tests.test_migration_inbound_photos import seed_old_writer


def _migration():
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0009_successful_retrievals.py"
    spec = importlib.util.spec_from_file_location("successful_retrieval_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == "0009" and module.down_revision == "0008"
    return module


def _snapshot(connection, metadata):
    return {table.name: [dict(row._mapping) for row in connection.execute(select(table).order_by(table.c.id))]
            for table in metadata.sorted_tables}


def test_success_migration_preserves_actual_prior_state_and_removes_only_new_cache_on_downgrade(tmp_path):
    previous = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name != "successful_retrievals":
            table.to_metadata(previous)
    engine = create_engine(f"sqlite:///{tmp_path / 'successful-retrieval-migration.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    migration = _migration()
    try:
        with engine.begin() as connection:
            previous.create_all(connection)
            seed_old_writer(connection)
            connection.execute(text("UPDATE talk_inbox SET media=:media WHERE id='inbox'"),
                               {"media": '{"files":[456]}'})
            connection.execute(text("UPDATE incidents SET has_user_photos=true WHERE id='incident'"))
            before = _snapshot(connection, previous)
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                migration.upgrade()
            assert compare_metadata(context, Base.metadata) == []
            assert _snapshot(connection, previous) == before
            assert connection.execute(select(SuccessfulRetrieval)).all() == []

            # An older writer doesn't know the cache table and can still append
            # text/metadata without adding a cache entry or changing its state.
            connection.execute(text(
                "INSERT INTO talk_inbox "
                "(id,created_at,room_id,remote_message_id,remote_sequence,sender_id,content,occurred_at,status,tries) "
                "VALUES ('legacy-inbox',CURRENT_TIMESTAMP,'room','13',13,'employee','Новый вопрос',"
                "CURRENT_TIMESTAMP,'PENDING',0)"
            ))
            legacy = _snapshot(connection, previous)
            source = {"file_id": "00000000-0000-4000-8000-000000000033", "filename": "Инструкция.docx",
                      "knowledge_id": "85cda3aa-b854-4c26-9948-12da742277ef", "sha256": "a" * 64}
            instant = now()
            connection.execute(SuccessfulRetrieval.__table__.insert().values(
                id="private-shortcut", created_at=instant, user_id="employee", incident_id="incident",
                scope_hash="b" * 64, query_hash="c" * 64, context_hash="d" * 64,
                source=source, expires_at=instant + timedelta(days=30),
            ))
            stored = connection.execute(select(SuccessfulRetrieval.__table__)).one()
            assert stored.user_id == "employee" and stored.incident_id == "incident" and stored.source == source
            assert _snapshot(connection, previous) == legacy
            with Operations.context(context):
                migration.downgrade()
            assert "successful_retrievals" not in connection.dialect.get_table_names(connection)
            assert compare_metadata(context, previous) == []
            assert _snapshot(connection, previous) == legacy
            with Operations.context(context):
                migration.upgrade()
            assert compare_metadata(context, Base.metadata) == []
            assert connection.execute(select(SuccessfulRetrieval)).all() == []
            assert _snapshot(connection, previous) == legacy
    finally:
        engine.dispose()
