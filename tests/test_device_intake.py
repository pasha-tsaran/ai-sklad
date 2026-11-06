"""A managed-PC callback advances only its requesting Talk form, without a model."""

import json
import re

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.core.config import get_settings
from app.main import app
from app.models import DeviceCollection, Escalation, TalkRoom
from app.services.chat.memory import load_context
from app.services.nextcloud.device_collection import capture_computer
from tests.test_talk_dialogue import Knowledge, NoGeneration, assistants, talk as talk


def start_collection(talk):
    talk.settings.nextcloud_backend = "account"
    talk.settings.device_collection_base_url = "https://support.example.invalid/ai-support"
    with talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, talk.room_id)
        room.profile = {"nextcloud_user_id": room.employee_id, "nextcloud_display_name": "Павел Царан",
                        "identity_source": "nextcloud_participants"}
    for sequence, message in enumerate(["оператор", "1"], 1):
        talk.incoming(message, sequence)
        assert talk.process(Knowledge(), NoGeneration())
    state = talk.snapshot()
    text = assistants(state)[-1][0]
    match = re.search(r"https://support\.example\.invalid/ai-support/device/collect#token=([A-Za-z0-9_-]{43})", text)
    assert match and state["flow"]["escalation"]["stage"] == "DEVICE"
    assert "API_TOKEN" not in text and "5 минут" in text
    return match[1]


def test_successful_pc_capture_continues_with_topic_then_exact_required_description(talk):
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    token = start_collection(talk)
    with talk.env.factory() as session, session.begin():
        assert capture_computer(session, talk.settings, token, "terminal76-sk") == {"status": "complete"}
    state = talk.snapshot()
    text = assistants(state)[-1][0]
    assert "ФИО: Павел Царан" in text and "Компьютер: terminal76-sk" in text
    assert "Укажите тему письма." in text and "Логин:" not in text
    assert state["flow"]["escalation"]["intake_requested_field"] == "subject"
    talk.incoming("Громкость сканера", 3)
    talk.process(Knowledge(), NoGeneration())
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
    talk.incoming("Сканер громко пищит.\nПрошу уменьшить громкость.", 4)
    talk.process(Knowledge(), NoGeneration())
    with talk.env.factory() as session:
        letter = session.scalar(select(Escalation))
        assert letter.email["subject"] == "[AI-Support] Громкость сканера"
        assert letter.email["body"] == "Сканер громко пищит.\nПрошу уменьшить громкость.\n\nФИО: Павел Царан\nКомпьютер: terminal76-sk\n"
        assert session.scalar(select(DeviceCollection)).consumed_at is not None


@pytest.mark.parametrize("message", ["Отмена", "terminal77-sk"])
def test_cancel_or_manual_pc_reply_invalidates_pending_collection_without_overwriting_employee_choice(talk, message):
    token = start_collection(talk)
    talk.incoming(message, 3)
    talk.process(Knowledge(), NoGeneration())
    before = talk.snapshot()
    with talk.env.factory() as session, session.begin():
        with pytest.raises(HTTPException):
            capture_computer(session, talk.settings, token, "WRONG-PC")
    assert talk.snapshot()["messages"] == before["messages"]
    with talk.env.factory() as session:
        profile = session.get(TalkRoom, talk.room_id).profile
        assert profile.get("device_name") != "WRONG-PC"
        if message == "terminal77-sk":
            assert profile["device_name"] == message


def test_collection_link_is_delivered_to_employee_but_never_put_into_model_context(talk):
    token = start_collection(talk)
    with talk.env.factory() as session, session.begin():
        incident = talk.snapshot()["incident"]
        context = load_context(session, incident, talk.settings)
        assert token not in json.dumps(context) and "REDACTED" in json.dumps(context)
    assert token in assistants(talk.snapshot())[-1][0]


def test_callback_routes_work_without_global_api_key_and_disabled_default_has_no_landing(env):
    assert env.client.get("/device/collect").status_code == 404
    assert env.client.get("/api/device-collection", headers={"Authorization": "Bearer " + "A" * 43}).status_code == 404


def test_http_callback_and_status_use_only_the_bound_one_use_capability(talk):
    token = start_collection(talk)
    app.dependency_overrides[get_settings] = lambda: talk.settings
    try:
        headers = {"Authorization": "Bearer " + token}
        client = talk.env.client
        assert client.get("/api/device-collection", headers=headers).json() == {"status": "pending"}
        result = client.post("/api/device-collection", json={"computer_name": "WS-001"}, headers=headers)
        assert result.status_code == 200 and result.json() == {"status": "complete"}
        assert client.get("/api/device-collection", headers=headers).json() == {"status": "complete"}
        assert client.post("/api/device-collection", json={"computer_name": "OTHER"}, headers=headers).status_code == 409
    finally:
        app.dependency_overrides.pop(get_settings, None)
