"""Actual 0007 rows survive the additive photo-metadata migration and rollback."""

import importlib.util
from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, select, text

from app.models import Base, Incident, TalkInbox


def migration():
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0008_inbound_photo_metadata.py"
    spec = importlib.util.spec_from_file_location("inbound_photo_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.revision == "0008" and module.down_revision == "0007"
    return module


def expected_metadata():
    expected = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name != "successful_retrievals":
            table.to_metadata(expected)
    return expected


def old_metadata():
    previous = expected_metadata()
    for table, name in [("talk_inbox", "media"), ("incidents", "has_user_photos")]:
        previous.tables[table]._columns.remove(previous.tables[table].c[name])
    return previous


def snapshot(connection, previous):
    names = ["users", "conversations", "incidents", "talk_accounts", "talk_rooms", "talk_inbox"]
    return {name: [dict(row._mapping) for row in connection.execute(select(previous.tables[name]).order_by(
        previous.tables[name].c.id,
    ))] for name in names}


def seed_old_writer(connection):
    # Raw SQL models legacy writers: neither Python ORM defaults nor new fields.
    commands = [
        "INSERT INTO users(id,created_at,external_id,display_name) VALUES('employee',CURRENT_TIMESTAMP,'legacy-uid','Имя')",
        "INSERT INTO conversations(id,created_at,user_id,channel,room_token) "
        "VALUES('conversation',CURRENT_TIMESTAMP,'employee','nextcloud_talk','private-room')",
        "INSERT INTO incidents(id,created_at,conversation_id,state,subject,description,failed_attempts,version,pending_action) "
        "VALUES('incident',CURRENT_TIMESTAMP,'conversation','DIAGNOSING','Принтер','Не печатает',1,7,'COLLECT_ESCALATION_DETAILS')",
        "INSERT INTO talk_accounts(id,created_at,server_url,username,activated_at,enabled) "
        "VALUES('account',CURRENT_TIMESTAMP,'https://cloud.example.invalid','bot-uid',CURRENT_TIMESTAMP,1)",
        "INSERT INTO talk_rooms(id,created_at,account_id,room_token,employee_id,cursor,enabled,flow,profile,incident_id,status) "
        "VALUES('room',CURRENT_TIMESTAMP,'account','private-room','employee',12,1,'{\"step\":2}',"
        "'{\"nextcloud_display_name\":\"Имя\"}','incident','ACTIVE')",
        "INSERT INTO talk_inbox(id,created_at,room_id,remote_message_id,remote_sequence,sender_id,content,occurred_at,status,tries,progress) "
        "VALUES('inbox',CURRENT_TIMESTAMP,'room','12',12,'employee','Подпись фотографии',CURRENT_TIMESTAMP,'PENDING',0,"
        "'{\"state\":\"CLEANED\"}')",
    ]
    for command in commands:
        connection.exec_driver_sql(command)


def test_inbound_migration_preserves_real_prior_rows_and_legacy_writer_defaults(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'inbound-photo-migration.db'}")
    previous = old_metadata()
    expected = expected_metadata()
    change = migration()
    with engine.begin() as connection:
        previous.create_all(connection)
        seed_old_writer(connection)
        before = snapshot(connection, previous)
        assert "media" not in {column["name"] for column in connection.dialect.get_columns(connection, "talk_inbox")}
        assert "has_user_photos" not in {column["name"] for column in connection.dialect.get_columns(connection, "incidents")}
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            change.upgrade()
        assert compare_metadata(context, expected) == []
        assert "successful_retrievals" not in connection.dialect.get_table_names(connection)
        assert snapshot(connection, previous) == before
        assert connection.scalar(select(TalkInbox.media).where(TalkInbox.id == "inbox")) == {}
        assert connection.scalar(select(Incident.has_user_photos).where(Incident.id == "incident")) is False
        # Older deployment code can continue inserting without knowing the columns.
        connection.exec_driver_sql(
            "INSERT INTO incidents(id,created_at,conversation_id,state,subject,description,failed_attempts,version) "
            "VALUES('legacy-incident',CURRENT_TIMESTAMP,'conversation','ESCALATED','Старая тема','Старый текст',0,0)"
        )
        connection.exec_driver_sql(
            "INSERT INTO talk_inbox(id,created_at,room_id,remote_message_id,remote_sequence,sender_id,content,occurred_at,status,tries) "
            "VALUES('legacy-inbox',CURRENT_TIMESTAMP,'room','13',13,'employee','Обычный текст',CURRENT_TIMESTAMP,'PENDING',0)"
        )
        assert connection.scalar(select(TalkInbox.media).where(TalkInbox.id == "legacy-inbox")) == {}
        assert connection.scalar(select(Incident.has_user_photos).where(Incident.id == "legacy-incident")) is False
        assert not next(column for column in connection.dialect.get_columns(connection, "talk_inbox")
                        if column["name"] == "media")["nullable"]
        assert not next(column for column in connection.dialect.get_columns(connection, "incidents")
                        if column["name"] == "has_user_photos")["nullable"]
        after_legacy = snapshot(connection, previous)
        connection.execute(text("UPDATE talk_inbox SET media=:media WHERE id='inbox'"), {"media": '{"files":[456]}'})
        connection.execute(text("UPDATE incidents SET has_user_photos=true WHERE id='incident'"))
        with Operations.context(context):
            change.downgrade()
        assert compare_metadata(context, previous) == []
        assert snapshot(connection, previous) == after_legacy
        with Operations.context(context):
            change.upgrade()
        assert compare_metadata(context, expected) == []
        assert snapshot(connection, previous) == after_legacy
        assert connection.scalar(select(TalkInbox.media).where(TalkInbox.id == "inbox")) == {}
        assert connection.scalar(select(Incident.has_user_photos).where(Incident.id == "incident")) is False
    engine.dispose()
