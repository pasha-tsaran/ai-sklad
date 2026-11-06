"""Additive Talk profile/progress migration preserves the durable private inbox."""

import importlib.util
import json
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, inspect, text

from app.models import Base


def _migration():
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0006_talk_profile_progress.py"
    spec = importlib.util.spec_from_file_location("profile_progress_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


@pytest.mark.parametrize("status", ["PENDING", "FAILED"])
def test_profile_progress_migration_preserves_existing_private_queue_and_roundtrips(tmp_path, status):
    expected = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name != "device_collections":
            table.to_metadata(expected)
    previous = MetaData()
    for table in expected.sorted_tables:
        table.to_metadata(previous)
    previous.tables["talk_rooms"]._columns.remove(previous.tables["talk_rooms"].c.profile)
    previous.tables["talk_inbox"]._columns.remove(previous.tables["talk_inbox"].c.progress)
    engine = create_engine(f"sqlite:///{tmp_path / 'profile-progress.db'}")
    previous.create_all(engine)
    migration = _migration()
    flow = {"source": {"file_id": "original-approved-document"}, "shown": [1, 2], "next_offset": 2}
    report = "Нужно настроить сканер на моём рабочем месте"
    try:
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO talk_accounts (id,created_at,server_url,username,activated_at,enabled) "
                "VALUES ('account',CURRENT_TIMESTAMP,'https://nextcloud.example','service-user',CURRENT_TIMESTAMP,1)"
            ))

            def add_legacy_rows(suffix):
                connection.execute(text(
                    "INSERT INTO talk_rooms "
                    "(id,created_at,account_id,room_token,employee_id,cursor,enabled,flow,status) "
                    "VALUES (:room,CURRENT_TIMESTAMP,'account',:token,'employee',81,1,:flow,'ACTIVE')"
                ), {"room": "room" + suffix, "token": "private" + suffix, "flow": json.dumps(flow)})
                connection.execute(text(
                    "INSERT INTO talk_inbox "
                    "(id,created_at,room_id,remote_message_id,remote_sequence,sender_id,content,occurred_at,status,tries,error_code) "
                    "VALUES (:inbox,CURRENT_TIMESTAMP,:room,'81',81,'employee',:content,CURRENT_TIMESTAMP,:status,2,:error)"
                ), {"inbox": "inbox" + suffix, "room": "room" + suffix, "content": report,
                    "status": status, "error": "PreviousReadFailure" if status == "FAILED" else None})

            def assert_legacy_rows(suffix):
                room = connection.execute(text(
                    "SELECT room_token,employee_id,cursor,flow FROM talk_rooms WHERE id=:id"
                ), {"id": "room" + suffix}).one()
                inbox = connection.execute(text(
                    "SELECT remote_message_id,remote_sequence,sender_id,content,status,tries,error_code "
                    "FROM talk_inbox WHERE id=:id"
                ), {"id": "inbox" + suffix}).one()
                assert tuple(room[:3]) == ("private" + suffix, "employee", 81) and json.loads(room.flow) == flow
                assert tuple(inbox[:6]) == ("81", 81, "employee", report, status, 2)
                assert inbox.error_code == ("PreviousReadFailure" if status == "FAILED" else None)

            def assert_empty_defaults(suffix):
                assert json.loads(connection.scalar(text(
                    "SELECT profile FROM talk_rooms WHERE id=:id"
                ), {"id": "room" + suffix})) == {}
                assert json.loads(connection.scalar(text(
                    "SELECT progress FROM talk_inbox WHERE id=:id"
                ), {"id": "inbox" + suffix})) == {}

            add_legacy_rows("old")
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                migration.upgrade()
            assert compare_metadata(context, expected) == []
            assert "device_collections" not in inspect(connection).get_table_names()
            assert_legacy_rows("old")
            assert_empty_defaults("old")
            # Raw SQL omits the new fields: defaults must also work for writers
            # still running the previous application during the rollout.
            add_legacy_rows("new")
            assert_empty_defaults("new")
            profile = {"nextcloud_user_id": "employee", "nextcloud_display_name": "Павел Царан"}
            progress = {"reference_id": "private-reply-progress", "started": True}
            connection.execute(text("UPDATE talk_rooms SET profile=:value WHERE id='roomold'"),
                               {"value": json.dumps(profile)})
            connection.execute(text("UPDATE talk_inbox SET progress=:value WHERE id='inboxold'"),
                               {"value": json.dumps(progress)})
            assert json.loads(connection.scalar(text("SELECT profile FROM talk_rooms WHERE id='roomold'"))) == profile
            assert json.loads(connection.scalar(text("SELECT progress FROM talk_inbox WHERE id='inboxold'"))) == progress
            assert_legacy_rows("old")
            with Operations.context(context):
                migration.downgrade()
            assert "profile" not in {column["name"] for column in inspect(connection).get_columns("talk_rooms")}
            assert "progress" not in {column["name"] for column in inspect(connection).get_columns("talk_inbox")}
            assert compare_metadata(context, previous) == []
            assert_legacy_rows("old")
            assert_legacy_rows("new")
            with Operations.context(context):
                migration.upgrade()
            assert compare_metadata(context, expected) == []
            for suffix in ("old", "new"):
                assert_empty_defaults(suffix)
                assert_legacy_rows(suffix)
    finally:
        engine.dispose()
