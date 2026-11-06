"""Optional progress and trusted Talk identity; no live accounts or network."""

import copy
import json

import httpx
import pytest

from app.services.nextcloud.account import PROGRESS_MESSAGE, AccountClient, AccountError
from tests.test_talk_account import EMPLOYEE, ROOM, SERVICE, Server, configuration, message, participant


class ProgressServer(Server):
    def __init__(self):
        super().__init__()
        self.features = ["delete-messages", "edit-messages"]
        self.target = message(901, actor=SERVICE, content=PROGRESS_MESSAGE)
        self.change_status = 200
        self.change_timeout = False
        self.after_read = None
        self.capability_failure = False

    def request(self, request):
        path = request.url.path.removeprefix("/office")
        if path == "/ocs/v2.php/cloud/capabilities":
            self.requests.append(request)
            if self.capability_failure:
                return httpx.Response(500)
            return self.ocs({"capabilities": {"spreed": {"features": self.features}}})
        chat = "/ocs/v2.php/apps/spreed/api/v1/chat/" + ROOM
        if path == chat and request.method == "GET" and request.url.params.get("includeLastKnown") == "1":
            self.requests.append(request)
            page = [] if self.target is None else [copy.deepcopy(self.target)]
            if self.after_read:
                self.after_read()
            return self.ocs(page)
        if path == chat + "/901" and request.method in {"DELETE", "PUT"}:
            self.requests.append(request)
            if self.change_timeout:
                raise httpx.ReadTimeout("simulated ambiguous cleanup", request=request)
            if self.change_status not in {200, 202}:
                return httpx.Response(self.change_status)
            return self.ocs({"id": 902, "parent": {"id": 901}}, status=self.change_status)
        return super().request(request)

    def client(self, **kwargs):
        return AccountClient(configuration(**kwargs), transport=self.transport)


def writes(server):
    return [request for request in server.requests if request.method != "GET"]


def test_employee_profile_uses_exact_private_participant_not_privileged_profile_endpoint():
    server = ProgressServer()
    server.participants[ROOM][0]["displayName"] = "Служебный бот"
    server.participants[ROOM][1]["displayName"] = "  Иван Иванов  "
    client = server.client()
    assert client.get_employee_profile(ROOM, expected_employee=EMPLOYEE) == {
        "user_id": EMPLOYEE, "display_name": "Иван Иванов",
    }
    assert client.validate_room(ROOM, expected_employee=EMPLOYEE) == {
        "token": ROOM, "type": 1, "service_id": SERVICE, "employee_id": EMPLOYEE,
    }
    assert all(request.method == "GET" for request in server.requests)
    assert not any("/cloud/users/" in request.url.path for request in server.requests)


def test_room_with_profile_uses_one_current_roster_without_additional_requests():
    server = ProgressServer()
    server.participants[ROOM][1]["displayName"] = "Иван Иванов"
    client = server.client()
    checked, profile = client.validate_room_with_profile(ROOM, expected_employee=EMPLOYEE)
    assert checked == {"token": ROOM, "type": 1, "service_id": SERVICE, "employee_id": EMPLOYEE}
    assert profile == {"user_id": EMPLOYEE, "display_name": "Иван Иванов"}
    assert [request.url.path.removeprefix("/office") for request in server.requests] == [
        "/ocs/v2.php/cloud/user", "/ocs/v2.php/apps/spreed/api/v4/room/" + ROOM,
        "/ocs/v2.php/apps/spreed/api/v4/room/" + ROOM + "/participants",
    ]
    assert all(request.method == "GET" for request in server.requests)
    server.requests.clear()
    server.participants[ROOM][1]["displayName"] = "Иван Петров"
    assert client.validate_room_with_profile(ROOM, expected_employee=EMPLOYEE)[1] == {
        "user_id": EMPLOYEE, "display_name": "Иван Петров",
    }
    assert len(server.requests) == 2
    server.requests.clear()
    server.participants[ROOM][1]["displayName"] = "Пётр Петров"
    assert client.get_employee_profile(ROOM, expected_employee=EMPLOYEE) == {
        "user_id": EMPLOYEE, "display_name": "Пётр Петров",
    }
    assert len(server.requests) == 2  # Standalone access also revalidates; no stale profile cache.


@pytest.mark.parametrize("name", [None, 3, "x" * 256, "Имя\nПодмена", "Имя\x00", "Имя\x7f",
                                 "Имя\x85Подмена", "Имя\x9f"],
                         ids=["missing", "number", "long", "newline", "nul", "delete-control",
                              "c1-newline", "c1-control"])
def test_invalid_display_name_stays_empty_and_never_invents_windows_information(name):
    server = ProgressServer()
    server.participants[ROOM][1]["displayName"] = name
    assert server.client().get_employee_profile(ROOM, expected_employee=EMPLOYEE) == {
        "user_id": EMPLOYEE, "display_name": "",
    }
    assert server.client().validate_room_with_profile(ROOM, expected_employee=EMPLOYEE)[1] == {
        "user_id": EMPLOYEE, "display_name": "",
    }


@pytest.mark.parametrize("change", ["group", "employee", "foreign", "guest", "extra"])
@pytest.mark.parametrize("method", ["get_employee_profile", "validate_room_with_profile"])
def test_profile_cannot_read_different_or_nonprivate_roster(change, method):
    server = ProgressServer()
    if change == "group":
        server.conversations[ROOM]["type"] = 2
    elif change == "employee":
        server.participants[ROOM][1] = participant("someone-else")
    elif change == "foreign":
        server.conversations[ROOM]["remoteServer"] = "foreign.example"
    elif change == "guest":
        server.participants[ROOM][1] = participant(EMPLOYEE, "guests")
    else:
        server.participants[ROOM].append(participant("someone-else"))
    with pytest.raises(AccountError):
        getattr(server.client(), method)(ROOM, expected_employee=EMPLOYEE)
    assert not writes(server)


@pytest.mark.parametrize("features, expected", [([], False), (["chat-get-context"], False),
                         (["delete-messages"], True), (["edit-messages"], True),
                         (["delete-messages", "edit-messages"], True)])
def test_cleanup_capabilities_are_read_only_and_cached(features, expected):
    server = ProgressServer()
    server.features = features
    client = server.client()
    assert client.supports_progress_cleanup() is expected
    assert client.supports_progress_cleanup() is expected
    assert len(server.requests) == 1 and not writes(server)


@pytest.mark.parametrize("invalid", [None, {}, [4], ["x" * 129], ["x"] * 1001],
                         ids=["null", "object", "number", "long-feature", "too-many"])
def test_invalid_optional_capabilities_disable_progress_without_failing_answer(invalid):
    server = ProgressServer()
    server.features = invalid
    assert server.client().supports_progress_cleanup() is False
    assert not writes(server)


@pytest.mark.parametrize("status", [200, 202])
def test_delete_verifies_author_literal_marker_and_private_roster_twice(status):
    server = ProgressServer()
    server.change_status = status
    assert server.client().delete_message(ROOM, "901", expected_employee=EMPLOYEE) is True
    assert [request.method for request in writes(server)] == ["DELETE"]
    assert writes(server)[0].url.path.endswith("/chat/roomone/901")
    reads = [request for request in server.requests if request.url.params.get("includeLastKnown") == "1"]
    assert len(reads) == 1
    assert dict(reads[0].url.params) == {
        "lookIntoFuture": "1", "lastKnownMessageId": "901", "limit": "1", "timeout": "0",
        "includeLastKnown": "1", "setReadMarker": "0", "noStatusUpdate": "1",
        "markNotificationsAsRead": "0",
    }
    assert len([request for request in server.requests if request.url.path.endswith("/participants")]) == 2


@pytest.mark.parametrize("changed", ["employee-author", "other-bot", "different-room", "different-id",
                         "normal-own-answer", "file-message", "system", "missing", "malformed"])
def test_cleanup_never_deletes_employee_history_or_other_service_content(changed):
    server = ProgressServer()
    if changed == "employee-author":
        server.target["actorId"] = EMPLOYEE
    elif changed == "other-bot":
        server.target["actorId"] = "another-bot"
    elif changed == "different-room":
        server.target["token"] = "otherroom"
    elif changed == "different-id":
        server.target["id"] = 902
    elif changed == "normal-own-answer":
        server.target["message"] = "Вот ваша техническая инструкция."
    elif changed == "file-message":
        server.target["messageType"] = "system"
    elif changed == "system":
        server.target["systemMessage"] = "message_deleted"
    elif changed == "missing":
        server.target = None
    else:
        server.target["id"] = "../901"
    assert server.client().delete_message(ROOM, 901, expected_employee=EMPLOYEE) is False
    assert not writes(server)


@pytest.mark.parametrize("operation", ["delete", "edit"])
def test_roster_revalidated_after_marker_read_before_optional_write(operation):
    server = ProgressServer()
    server.after_read = lambda: server.participants[ROOM].append(participant("another-employee"))
    client = server.client()
    result = (client.delete_message(ROOM, 901, expected_employee=EMPLOYEE) if operation == "delete"
              else client.edit_message(ROOM, 901, "Обработка завершена.", expected_employee=EMPLOYEE))
    assert result is False and not writes(server)


def test_edit_only_server_replaces_marker_with_fixed_completion_and_never_changes_instruction():
    server = ProgressServer()
    server.features = ["edit-messages"]
    client = server.client()
    assert client.supports_progress_cleanup() is True
    assert client.delete_message(ROOM, 901, expected_employee=EMPLOYEE) is False
    assert client.edit_message(ROOM, 901, "Обработка завершена.", expected_employee=EMPLOYEE) is True
    assert len(writes(server)) == 1
    assert writes(server)[0].method == "PUT"
    assert json.loads(writes(server)[0].content) == {"message": "Обработка завершена."}
    assert client.edit_message(ROOM, 901, "Удалить чужие данные", expected_employee=EMPLOYEE) is False
    assert client.edit_message(ROOM, 901, "Ответ подготовлен.", expected_employee=EMPLOYEE) is False
    assert len(writes(server)) == 1


@pytest.mark.parametrize("invalid", [None, "", "someone-else"])
def test_cleanup_requires_exact_bound_employee(invalid):
    server = ProgressServer()
    assert server.client().delete_message(ROOM, 901, expected_employee=invalid) is False
    assert not writes(server)


@pytest.mark.parametrize("case", ["capability-error", "too-old", "forbidden", "missing", "network"])
def test_optional_cleanup_errors_return_false_once_without_retries_or_reply_failure(case):
    server = ProgressServer()
    server.capability_failure = case == "capability-error"
    server.change_status = {"too-old": 400, "forbidden": 403, "missing": 404}.get(case, 200)
    server.change_timeout = case == "network"
    assert server.client().delete_message(ROOM, 901, expected_employee=EMPLOYEE) is False
    assert len(writes(server)) == (0 if case == "capability-error" else 1)


def test_unsupported_server_uses_no_dummy_typing_or_progress_mutation():
    server = ProgressServer()
    server.features = []
    client = server.client()
    assert client.supports_progress_cleanup() is False
    assert client.delete_message(ROOM, 901, expected_employee=EMPLOYEE) is False
    assert client.edit_message(ROOM, 901, "Обработка завершена.", expected_employee=EMPLOYEE) is False
    assert not hasattr(client, "set_typing")
    assert not writes(server)
