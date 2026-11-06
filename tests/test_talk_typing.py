"""Real signaling wire contracts exercised with HTTP/WS mocks, never a live account."""

import copy
import json
import threading
import time
from urllib.parse import parse_qs

import httpx
import pytest

from app.services.nextcloud.typing import MAX_ROOMS, NativeTyping
from tests.test_talk_account import EMPLOYEE, ROOM, SERVICE, Server, participant

INTERNAL_POLL = NativeTyping._start_internal_poll
SOCKET_POLL = NativeTyping._start_socket_poll


class SignalingServer(Server):
    def __init__(self):
        super().__init__()
        self.mode = "internal"
        self.privacy = 0
        self.features = ["signaling-v3", "typing-privacy"]
        self.settings_uid = SERVICE
        self.settings_server = "https://signal.example.invalid/hpb/"
        self.sessions = {}
        self.jar_count = 0
        self.conflict = False
        self.join_timeout = False
        self.after_join = None
        self.alter_join_id = None
        self.peer_ids = ["employee-web", "employee-desktop"]

    def request(self, request):
        path = request.url.path.removeprefix("/office")
        cookie = request.headers.get("cookie", "")
        if path == "/ocs/v2.php/cloud/user":
            self.requests.append(request)
            self.jar_count += 1
            return self.ocs({"id": self.uid}, headers={"Set-Cookie": f"native=jar{self.jar_count}; Path=/; Secure"})
        if path == "/ocs/v2.php/cloud/capabilities":
            self.requests.append(request)
            return self.ocs({"capabilities": {"spreed": {
                "features": self.features, "config": {"chat": {"typing-privacy": self.privacy}},
            }}})
        if path == "/ocs/v2.php/apps/spreed/api/v3/signaling/settings":
            self.requests.append(request)
            return self.ocs({"signalingMode": self.mode, "userId": self.settings_uid,
                             "server": self.settings_server, "helloAuthParams": {
                                 "1.0": {"userid": SERVICE, "ticket": "short-lived-ticket"},
                                 "2.0": {"token": "short-lived-jwt"},
                             }})
        room_prefix = "/ocs/v2.php/apps/spreed/api/v4/room/"
        if path.startswith(room_prefix):
            token, _, suffix = path.removeprefix(room_prefix).partition("/")
            self.requests.append(request)
            if token not in self.conversations:
                return httpx.Response(404)
            if suffix == "participants/active":
                if request.method == "POST":
                    assert cookie and json.loads(request.content) == {"force": False}
                    if self.conflict:
                        return httpx.Response(409)
                    self.sessions[(cookie, token)] = "owned-" + cookie.split("=")[-1]
                    if self.join_timeout:
                        raise httpx.ReadTimeout("simulated join ambiguity", request=request)
                    if self.after_join:
                        self.after_join()
                else:
                    assert request.method == "DELETE" and (cookie, token) in self.sessions
                    self.sessions.pop((cookie, token))
                    return self.ocs(None)
            if suffix == "participants":
                people = copy.deepcopy(self.participants[token])
                for person in people:
                    if person["actorId"] == SERVICE:
                        person["sessionIds"] = ["existing-bot-desktop"] + [
                            sid for (_, room), sid in self.sessions.items() if room == token
                        ]
                    else:
                        person["sessionIds"] = self.peer_ids
                return self.ocs(people)
            room = copy.deepcopy(self.conversations[token])
            room["sessionId"] = self.sessions.get((cookie, token), "0")
            if suffix == "participants/active" and self.alter_join_id:
                room["sessionId"] = self.alter_join_id
            return self.ocs(room)
        if path.startswith("/ocs/v2.php/apps/spreed/api/v3/signaling/"):
            self.requests.append(request)
            token = path.rsplit("/", 1)[-1]
            assert self.sessions.get((cookie, token))
            if request.method == "POST":
                items = json.loads(parse_qs(request.content.decode())["messages"][0])
                assert all(item["sessionId"] == self.sessions[(cookie, token)] for item in items)
            return self.ocs(None if request.method == "POST" else [])
        return super().request(request)


class Socket:
    def __init__(self, *, version2=True, hello_uid=SERVICE, room=ROOM):
        self.incoming = [
            {"type": "welcome", "welcome": {"features": ["hello-v2"] if version2 else []}},
            {"type": "hello", "hello": {"sessionid": "hpb-own-session", "userid": hello_uid}},
            {"type": "room", "room": {"roomid": room}},
        ]
        self.sent, self.calls, self.closed = [], [], False
        self.welcome_timeout = False

    def recv(self, *, timeout):
        assert 0 < timeout <= 3
        if self.welcome_timeout:
            self.welcome_timeout = False
            raise TimeoutError
        if not self.incoming:
            raise TimeoutError
        return json.dumps(self.incoming.pop(0))

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def close(self):
        self.closed = True

    def connect(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self


@pytest.fixture
def native(monkeypatch):
    # A unit test must not leave background long-polls or socket receiver threads.
    monkeypatch.setattr(NativeTyping, "_start_internal_poll", lambda *a: None)
    monkeypatch.setattr(NativeTyping, "_start_socket_poll", lambda *a: None)
    server, socket = SignalingServer(), Socket()
    account = server.client()
    client = NativeTyping(account.settings, account, websocket_connect=socket.connect)
    yield server, socket, client
    client.close()


def internal_frames(server):
    return [json.loads(item["fn"]) for request in server.requests if request.method == "POST"
            and "/api/v3/signaling/" in request.url.path
            for item in json.loads(parse_qs(request.content.decode())["messages"][0])]


def assert_no_chat_writes(server):
    assert not any("/chat/" in request.url.path or "/call/" in request.url.path
                   or "/remote.php/" in request.url.path for request in server.requests)


def test_internal_start_refresh_stop_is_native_private_and_uses_own_persistent_cookie(native):
    server, socket, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=False)
    assert internal_frames(server) == [
        {"to": peer, "type": kind} for kind in ["startedTyping", "startedTyping", "stoppedTyping"]
        for peer in server.peer_ids
    ]
    joins = [request for request in server.requests if request.method == "POST"
             and request.url.path.endswith("/participants/active")]
    leaves = [request for request in server.requests if request.method == "DELETE"]
    assert len(joins) == len(leaves) == 1
    assert joins[0].headers["cookie"] == leaves[0].headers["cookie"]
    assert not server.sessions and not socket.calls
    assert_no_chat_writes(server)


def test_each_room_owns_fresh_cookie_jar_and_never_replaces_desktop_session(native):
    server, _, client = native
    desktop_key = ("native=existing-browser", ROOM)
    server.sessions[desktop_key] = "desktop-session"
    other = "secondroom"
    server.conversations[other] = {"token": other, "type": 1}
    server.participants[other] = [participant(SERVICE), participant(EMPLOYEE)]
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert client.set_typing(other, expected_employee=EMPLOYEE, active=True)
    joins = [request for request in server.requests if request.method == "POST"
             and request.url.path.endswith("/participants/active")]
    cookies = {request.headers["cookie"] for request in joins}
    assert len(cookies) == 2 and desktop_key[0] not in cookies
    assert all(json.loads(request.content) == {"force": False} for request in joins)
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=False)
    client.close()
    assert server.sessions == {desktop_key: "desktop-session"}
    assert_no_chat_writes(server)


@pytest.mark.parametrize("mode,version2", [("external", True), ("external", False), ("conversation_cluster", True)])
def test_hpb_hello_php_join_typing_and_bye_use_official_websocket_contract(native, mode, version2):
    server, socket, client = native
    server.mode = mode
    socket.incoming[0]["welcome"]["features"] = ["hello-v2"] if version2 else []
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=False)
    url, arguments = socket.calls[0]
    assert url == "wss://signal.example.invalid/hpb/spreed"
    assert arguments["origin"] == "https://nextcloud.example"
    assert arguments["proxy"] is None and arguments["open_timeout"] == 3
    assert "additional_headers" not in arguments
    expected_auth = {"token": "short-lived-jwt"} if version2 else {"userid": SERVICE, "ticket": "short-lived-ticket"}
    assert socket.sent[0] == {"type": "hello", "hello": {
        "version": "2.0" if version2 else "1.0", "auth": {
            "url": "https://nextcloud.example/office/ocs/v2.php/apps/spreed/api/v3/signaling/backend",
            "params": expected_auth,
        },
    }}
    assert socket.sent[1]["type"] == "room" and socket.sent[1]["room"]["roomid"] == ROOM
    assert socket.sent[1]["room"]["sessionid"].startswith("owned-jar")
    assert socket.sent[2:4] == [{"type": "message", "message": {
        "recipient": {"type": "room"}, "data": {"type": kind},
    }} for kind in ["startedTyping", "stoppedTyping"]]
    assert socket.sent[-2:] == [{"type": "room", "room": {"roomid": ""}}, {"type": "bye"}]
    assert "test-only-app-password" not in json.dumps(socket.sent) and socket.closed
    assert not server.sessions and not internal_frames(server)
    assert_no_chat_writes(server)


def test_older_hpb_without_welcome_uses_bounded_hello1_fallback(native):
    server, socket, client = native
    server.mode = "external"
    socket.incoming.pop(0)
    socket.welcome_timeout = True
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert socket.sent[0]["hello"]["version"] == "1.0"
    assert socket.sent[0]["hello"]["auth"]["params"] == {"userid": SERVICE, "ticket": "short-lived-ticket"}
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=False)
    assert_no_chat_writes(server)


@pytest.mark.parametrize("privacy", [1, "0", False, None])
def test_missing_private_or_malformed_privacy_never_joins_or_signals(native, privacy):
    server, socket, client = native
    server.privacy = privacy
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert not any(request.method != "GET" for request in server.requests) and not socket.calls


@pytest.mark.parametrize("change", ["group", "foreign", "guest", "extra", "employee", "identity"])
def test_never_joins_or_broadcasts_outside_exact_authenticated_private_room(native, change):
    server, socket, client = native
    if change == "group":
        server.conversations[ROOM]["type"] = 2
    elif change == "foreign":
        server.conversations[ROOM]["remoteServer"] = "elsewhere.example"
    elif change == "guest":
        server.participants[ROOM][1] = participant(EMPLOYEE, "guests")
    elif change == "extra":
        server.participants[ROOM].append(participant("third-user"))
    elif change == "employee":
        server.participants[ROOM][1] = participant("another-employee")
    else:
        client.account._uid = "another-service"
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert not any(request.method != "GET" for request in server.requests) and not socket.calls


@pytest.mark.parametrize("change", ["group", "employee", "privacy"])
def test_rechecks_before_refresh_and_leaves_only_owned_session_when_access_changes(native, change):
    server, _, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    before = list(internal_frames(server))
    if change == "group":
        server.conversations[ROOM]["type"] = 2
    elif change == "employee":
        server.participants[ROOM][1] = participant("another-employee")
    else:
        server.privacy = 1
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert internal_frames(server) == before and not server.sessions
    assert len([request for request in server.requests if request.method == "DELETE"]) == 1
    assert_no_chat_writes(server)


def test_stop_after_privacy_change_closes_without_broadcasting_or_removing_chat_membership(native):
    server, _, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    before = list(internal_frames(server))
    server.privacy = 1
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=False)
    assert internal_frames(server) == before and not server.sessions
    assert all(request.url.path.endswith("/participants/active") for request in server.requests if request.method == "DELETE")


@pytest.mark.parametrize("case", ["conflict", "join-timeout", "settings-user", "mode", "own-session", "bad-peer"])
def test_unsupported_conflicting_or_invalid_sessions_fail_open_without_chat_fallback(native, case):
    server, _, client = native
    if case == "conflict":
        server.conflict = True
    elif case == "join-timeout":
        server.join_timeout = True
    elif case == "settings-user":
        server.settings_uid = "other-user"
    elif case == "mode":
        server.mode = "unknown"
    elif case == "own-session":
        server.alter_join_id = "existing-bot-desktop"
    else:
        server.peer_ids = ["unsafe\npeer"]
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert internal_frames(server) == [] and not client._sessions
    if case == "conflict":
        joins = [request for request in server.requests if request.method == "POST"]
        assert len(joins) == 1 and json.loads(joins[0].content) == {"force": False}
    assert_no_chat_writes(server)


@pytest.mark.parametrize("server_url", ["http://signal.example", "ws://signal.example", "wss://user:pass@signal.example",
                                        "wss://signal.example/path?secret=1", "wss://signal.example/#fragment"])
def test_external_server_must_be_secure_and_never_receives_basic_credentials(native, server_url):
    server, socket, client = native
    server.mode, server.settings_server = "external", server_url
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert not socket.calls and not any(request.method != "GET" for request in server.requests)


def test_wrong_hpb_identity_or_room_never_gets_typing_frame_and_closes(native):
    server, socket, client = native
    server.mode = "external"
    socket.incoming[1]["hello"]["userid"] = "other-user"
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert not any(frame["type"] == "message" for frame in socket.sent)
    assert socket.closed and not server.sessions


@pytest.mark.parametrize("mode", ["internal", "external", "conversation_cluster"])
def test_readonly_check_whitelists_report_and_never_joins_or_exposes_auth_details(native, mode):
    server, socket, client = native
    server.mode = mode
    assert client.check() == {"status": "ok", "mode": mode, "typing_supported": True,
                              "privacy_allowed": True, "read_only": True}
    assert all(request.method == "GET" for request in server.requests)
    settings = [request for request in server.requests if request.url.path.endswith("/signaling/settings")]
    assert len(settings) == 1 and settings[0].url.params["token"] == ROOM
    assert not socket.calls and not server.sessions
    server.privacy = 1
    report = client.check()
    assert report["status"] == "unavailable" and not report["typing_supported"] and not report["privacy_allowed"]


def test_readonly_check_requests_a_private_room_when_none_can_be_validated(native):
    server, socket, client = native
    server.conversations[ROOM]["type"] = 2
    report = client.check()
    assert report == {"status": "unavailable", "mode": "unknown", "typing_supported": False,
                      "privacy_allowed": True, "read_only": True, "needs_private_room": True}
    assert all(request.method == "GET" for request in server.requests)
    assert not any(request.url.path.endswith("/signaling/settings") for request in server.requests)
    assert not socket.calls


def test_no_session_stop_is_noop_and_close_permanently_prevents_future_starts(native):
    server, _, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=False)
    assert not server.requests
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    client.close()
    count = len(server.requests)
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    client.close()
    assert len(server.requests) == count and not server.sessions


def test_capacity_is_bounded_before_an_additional_session_or_join(native):
    server, _, client = native
    client._sessions = {f"room{i}": object() for i in range(MAX_ROOMS)}
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert not server.requests
    client._sessions = {}


class InlineThread:
    def __init__(self, *, target, args=(), **kwargs):
        self.target, self.args = target, args
        assert kwargs.get("daemon") is True

    def start(self):
        self.target(*self.args)

    def join(self, **kwargs):
        pass


def test_internal_longpoll_uses_owned_cookie_and_discards_notifications_before_cleanup(native, monkeypatch):
    server, _, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    session = client._sessions[ROOM]
    original = session._request
    observations = []
    def request(method, path, **kwargs):
        result = original(method, path, **kwargs)
        if method == "GET" and path.endswith("/signaling/" + ROOM):
            observations.append(kwargs)
            session.stopped.set()
            return [{"type": "message", "data": "untrusted private payload, never a reply"}], {}, 200
        return result
    monkeypatch.setattr(session, "_request", request)
    monkeypatch.setattr("app.services.nextcloud.typing.threading.Thread", InlineThread)
    INTERNAL_POLL(client, session, ROOM)
    assert observations == [{"timeout": 35}]
    polls = [request for request in server.requests if request.method == "GET"
             and request.url.path.endswith("/signaling/" + ROOM)]
    joins = [request for request in server.requests if request.method == "POST"
             and request.url.path.endswith("/participants/active")]
    assert polls[0].headers["cookie"] == joins[0].headers["cookie"]
    assert not client._sessions and not server.sessions and session.http.is_closed
    assert_no_chat_writes(server)


def test_expired_native_session_leaves_without_another_longpoll_or_chat_message(native, monkeypatch):
    server, _, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    session = client._sessions[ROOM]
    session.refreshed = time.monotonic() - 21
    monkeypatch.setattr("app.services.nextcloud.typing.threading.Thread", InlineThread)
    INTERNAL_POLL(client, session, ROOM)
    assert not client._sessions and not server.sessions
    assert not any(request.method == "GET" and request.url.path.endswith("/signaling/" + ROOM)
                   for request in server.requests)
    assert [frame["type"] for frame in internal_frames(server)] == [
        "startedTyping", "startedTyping", "stoppedTyping", "stoppedTyping",
    ]
    assert_no_chat_writes(server)


@pytest.mark.parametrize("kind", ["bye", "error"])
def test_hpb_receiver_session_end_closes_only_own_signaling_without_chat(native, monkeypatch, kind):
    server, socket, client = native
    server.mode = "external"
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    session = client._sessions[ROOM]
    socket.incoming.append({"type": kind, kind: {}})
    monkeypatch.setattr("app.services.nextcloud.typing.threading.Thread", InlineThread)
    SOCKET_POLL(client, session, ROOM)
    assert socket.closed and not client._sessions and not server.sessions
    assert socket.sent[-2:] == [{"type": "room", "room": {"roomid": ""}}, {"type": "bye"}]
    assert_no_chat_writes(server)


def test_close_flag_during_roster_read_prevents_late_started_typing(native, monkeypatch):
    server, _, client = native
    assert client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    session = client._sessions[ROOM]
    before = list(internal_frames(server))
    original = session._private_room
    def close_during_read(*args):
        result = original(*args)
        client._closed = True  # Concurrent close sets this before waiting for the network lock.
        return result
    monkeypatch.setattr(session, "_private_room", close_during_read)
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert internal_frames(server) == before and not server.sessions


def test_close_timeout_includes_wait_for_inflight_lock_and_never_allows_restart(native, monkeypatch):
    _, _, client = native
    entered, release = threading.Event(), threading.Event()
    def hold_network_lock():
        with client._lock:
            entered.set()
            release.wait(timeout=2)
    holder = threading.Thread(target=hold_network_lock, daemon=True)
    holder.start()
    assert entered.wait(timeout=1)
    monkeypatch.setattr("app.services.nextcloud.typing.CLOSE_TIMEOUT", 0.02)
    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started
    release.set()
    holder.join(timeout=1)
    assert elapsed < 0.5 and client._closed
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)


def test_blocked_hpb_write_times_out_aborts_own_socket_and_cannot_send_late_typing(native, monkeypatch):
    server, socket, client = native
    server.mode = "external"
    aborted, writer_finished = threading.Event(), threading.Event()
    class RawSocket:
        def shutdown(self, how):
            aborted.set()

        def close(self):
            aborted.set()
    socket.socket = RawSocket()
    original = socket.send
    def blocked(raw):
        frame = json.loads(raw)
        if frame.get("type") == "message":
            assert aborted.wait(timeout=1)
            writer_finished.set()
            raise OSError("owned socket interrupted")
        original(raw)
    monkeypatch.setattr(socket, "send", blocked)
    monkeypatch.setattr("app.services.nextcloud.typing.WS_SEND_TIMEOUT", 0.02)
    started = time.monotonic()
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert time.monotonic() - started < 0.8
    assert aborted.is_set() and writer_finished.wait(timeout=1)
    assert not any(frame.get("type") == "message" for frame in socket.sent)
    assert socket.closed and not client._sessions and not server.sessions
    assert_no_chat_writes(server)


def test_closed_flag_is_checked_inside_deferred_writer_before_started_typing(native, monkeypatch):
    server, socket, client = native
    server.mode = "external"
    class CloseBeforeWrite(InlineThread):
        def start(self):
            if len(socket.sent) == 2:
                client._closed = True
            super().start()
    monkeypatch.setattr("app.services.nextcloud.typing.threading.Thread", CloseBeforeWrite)
    assert not client.set_typing(ROOM, expected_employee=EMPLOYEE, active=True)
    assert not any(frame.get("type") == "message" for frame in socket.sent)
    assert socket.closed and not client._sessions and not server.sessions
