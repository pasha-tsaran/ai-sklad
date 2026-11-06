"""Identity comes from the validated employee roster; devices expire with incidents."""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.models import TalkInbox, TalkRoom
from app.services.chat.channel_profile import channel_profile, remember_computer, update_nextcloud_profile
from app.services.nextcloud.poller import poll_once
from tests.test_talk_account import EMPLOYEE, ROOM, SERVICE, Server, configuration, factory as factory, message


def room(**profile):
    return SimpleNamespace(employee_id=EMPLOYEE, profile=profile)


def test_identity_refresh_keeps_current_confirmed_device_and_only_trusted_nextcloud_fields():
    target = room(device_name="WS-001", device_incident_id="current", device_source="employee_in_current_incident")
    assert update_nextcloud_profile(target, {"user_id": EMPLOYEE, "display_name": "  Иванов Иван  ",
                                             "computer_name": "SERVER", "windows_login": "administrator"})
    assert channel_profile(target, "current") == {
        "account_holder_name": "Иванов Иван", "account_login": EMPLOYEE, "computer_name": "WS-001",
    }
    assert channel_profile(target, "next") == {"account_holder_name": "Иванов Иван", "account_login": EMPLOYEE}
    assert "windows_login" not in target.profile and "computer_name" not in target.profile


@pytest.mark.parametrize("observed", [None, {}, {"user_id": SERVICE, "display_name": "Бот"},
                                      {"user_id": "another", "display_name": "Чужой"}])
def test_unbound_profile_cannot_change_employee_identity_or_device(observed):
    target = room(device_name="WS-001", device_incident_id="current")
    saved = dict(target.profile)
    assert not update_nextcloud_profile(target, observed) and target.profile == saved
    assert "account_holder_name" not in channel_profile(target, "current")


@pytest.mark.parametrize("name", ["...", "---", "___", "WS\nOTHER", "WS\x85OTHER", " WS 001 ", "x" * 64])
def test_invalid_or_control_containing_device_never_becomes_a_current_workstation(name):
    target = room(device_name="WS-001", device_incident_id="previous")
    before = dict(target.profile)
    assert not remember_computer(target, "current", name) and target.profile == before
    target.profile.update(device_name=name, device_incident_id="current")
    assert "computer_name" not in channel_profile(target, "current")


def test_workstation_is_remembered_only_for_current_incident_without_overwriting_identity():
    target = room()
    update_nextcloud_profile(target, {"user_id": EMPLOYEE, "display_name": "Иванов Иван"})
    assert remember_computer(target, "current", "terminal76-sk")
    assert channel_profile(target, "current")["computer_name"] == "terminal76-sk"
    assert "computer_name" not in channel_profile(target, "previous")
    assert target.profile["device_source"] == "employee_in_current_incident"
    assert channel_profile(target, "current")["account_holder_name"] == "Иванов Иван"


def test_polling_stores_roster_identity_with_durable_employee_text_but_not_server_device(factory):
    server = Server()
    server.participants[ROOM][1]["displayName"] = "Иванов Иван"
    server.history[ROOM] = [message(1)]
    assert poll_once(factory, configuration(), server.client())
    with factory() as session:
        target = session.scalar(select(TalkRoom))
        assert channel_profile(target, "incident") == {"account_holder_name": "Иванов Иван", "account_login": EMPLOYEE}
        assert session.scalar(select(TalkInbox)).status == "PENDING"
        assert "device_name" not in target.profile


def test_missing_optional_metadata_never_discards_text_or_prevents_cursor_commit(factory):
    server = Server()
    server.history[ROOM] = [message(1)]
    client = server.client()

    def without_profile(token, expected_employee=None):
        return client.validate_room(token, expected_employee=expected_employee), None

    client.validate_room_with_profile = without_profile
    assert poll_once(factory, configuration(), client)
    with factory() as session:
        target = session.scalar(select(TalkRoom))
        assert target.cursor == 1 and target.enabled and target.profile == {}
        assert session.scalar(select(TalkInbox)).content == "Нужно настроить сканер"


def test_polling_never_fetches_a_separate_profile_endpoint_or_saves_bot_echo_metadata(factory):
    server = Server()
    client = server.client()

    def unexpected(*args, **kwargs):
        pytest.fail("Optional profile must never add a separate network request before durable ingestion")

    client.get_employee_profile = unexpected
    assert not poll_once(factory, configuration(), client)
    server.history[ROOM] = [message(1, actor=SERVICE)]
    assert not poll_once(factory, configuration(), client)
    with factory() as session:
        assert session.scalar(select(TalkRoom)).profile == {}
    server.history[ROOM].append(message(2))
    assert poll_once(factory, configuration(), client)
    participant_reads = [request for request in server.requests if request.url.path.endswith("/participants")]
    assert len(participant_reads) == 3  # One mandatory private-room check per page, no profile refetch.
