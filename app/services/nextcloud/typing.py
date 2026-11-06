"""Private, ephemeral Talk typing via the official internal/HPB signaling APIs.

No chat messages, call starts, privacy changes, stored tickets or resumable sessions.
https://nextcloud-talk.readthedocs.io/en/latest/signaling-messages/#typing
https://github.com/nextcloud/spreed/blob/v23.0.1/src/utils/signaling.js
"""

import json
import socket as network_socket
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.services.nextcloud.account import MAX_RESPONSE_BYTES, AccountClient, AccountError

MAX_ROOMS = 16
MAX_PEER_SESSIONS = 64
CLOSE_TIMEOUT = 5
WS_SEND_TIMEOUT = 3
SIGNAL_PATH = "/ocs/v2.php/apps/spreed/api/v3/signaling"
ROOM_PATH = "/ocs/v2.php/apps/spreed/api/v4/room/"


def _session_id(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 512 or value == "0"
            or any(ord(char) < 33 or ord(char) > 126 for char in value)):
        raise AccountError("invalid_signaling_session")
    return value


def _websocket_url(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise AccountError("invalid_signaling_server")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"https", "wss"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or "\\" in value
            or any(ord(char) < 33 for char in value)):
        raise AccountError("invalid_signaling_server")
    return urlunsplit(("wss", parsed.netloc, parsed.path.rstrip("/") + "/spreed", "", ""))


def _profile(data):
    try:
        spreed = data["capabilities"]["spreed"]
        features = spreed["features"]
        privacy = spreed["config"]["chat"]["typing-privacy"]
    except (TypeError, KeyError):
        raise AccountError("typing_capabilities_unavailable") from None
    if (not isinstance(features, list) or len(features) > 1000
            or any(not isinstance(item, str) for item in features)):
        raise AccountError("typing_capabilities_unavailable")
    return "signaling-v3" in features, type(privacy) is int and privacy == 0


class _Session(AccountClient):
    """A new private cookie jar, never a browser's or a different room's session."""

    def __init__(self, settings, transport, employee):
        super().__init__(settings, transport=transport)
        self.http = httpx.Client(auth=(self.username, self.password), verify=True, follow_redirects=False,
                                 timeout=3, transport=transport,
                                 headers={"OCS-APIRequest": "true", "Accept": "application/json"})
        self.employee, self.session_id, self.mode = employee, None, None
        self.socket = None
        self.socket_failed = False
        self.can_start = lambda: True
        self.stopped = threading.Event()
        self.refreshed = time.monotonic()

    def _request(self, method, path, *, ocs=True, acceptable=(200, 201), **kwargs):
        with self.http.stream(method, self.base + path, **kwargs) as response:
            if response.status_code not in acceptable:
                raise AccountError("typing_http_unavailable")
            raw = bytearray()
            for block in response.iter_bytes():
                raw.extend(block)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise AccountError("typing_response_too_large")
            status, headers = response.status_code, dict(response.headers)
        if not ocs:
            return None, headers, status
        try:
            envelope = json.loads(raw)["ocs"]
            if envelope["meta"]["statuscode"] not in {100, 200, 201}:
                raise AccountError("typing_ocs_unavailable")
            return envelope["data"], headers, status
        except (KeyError, ValueError, TypeError):
            raise AccountError("typing_ocs_unavailable") from None

    def receive(self, expected):
        deadline = time.monotonic() + 3
        for _ in range(32):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            value = self.socket.recv(timeout=remaining)
            if not isinstance(value, str) or len(value) > 1_000_000:
                raise AccountError("typing_invalid_frame")
            data = json.loads(value)
            if not isinstance(data, dict) or data.get("type") == "error":
                raise AccountError("typing_signaling_unavailable")
            if data.get("type") == expected:
                return data[expected]
        raise AccountError("typing_signaling_timeout")

    def frame(self, value):
        if self.socket_failed:
            raise AccountError("typing_write_unavailable")
        raw = json.dumps(value, separators=(",", ":"))
        started = value.get("type") == "message" and value.get("message", {}).get("data", {}).get("type") == "startedTyping"
        done, errors = threading.Event(), []
        def write():
            try:
                if started and (self.stopped.is_set() or not self.can_start()):
                    raise AccountError("typing_closed")
                self.socket.send(raw)
            except Exception:
                errors.append(True)
            finally:
                done.set()
        threading.Thread(target=write, name="talk-typing-write", daemon=True).start()
        if not done.wait(timeout=WS_SEND_TIMEOUT):
            self.socket_failed = True
            self.stopped.set()
            # websockets.sync.send() otherwise has no regular write deadline.
            # Shutdown this owned socket to interrupt sendall and prevent a late
            # queued startedTyping frame after an ambiguous timeout or close.
            connection = getattr(self.socket, "socket", None)
            if connection is not None:
                try:
                    connection.shutdown(network_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    connection.close()
                except OSError:
                    pass
            else:
                threading.Thread(target=self.socket.close, daemon=True).start()
            raise AccountError("typing_write_timeout")
        if errors:
            raise AccountError("typing_write_unavailable")


class NativeTyping:
    def __init__(self, settings, account_client=None, *, websocket_connect=None):
        self.settings = settings
        self.account = account_client or AccountClient(settings)
        self.websocket_connect = websocket_connect
        self._sessions = {}
        self._lock = threading.RLock()
        self._closed = False

    def identity(self):
        return self.account.identity()

    def check(self):
        """Read-only report: never exposes identifiers, servers, cookies or tickets."""
        result = {"status": "unavailable", "mode": "unknown", "typing_supported": False,
                  "privacy_allowed": False, "read_only": True}
        try:
            data, _, _ = self.account._request("GET", "/ocs/v2.php/cloud/capabilities")
            supported, public = _profile(data)
            result["privacy_allowed"] = public
            tokens = self.settings.nextcloud_allowed_rooms
            if not tokens:
                tokens = [room.get("token") for room in self.account.rooms() if room.get("type") == 1]
            checked = None
            for token in tokens[:MAX_ROOMS]:
                try:
                    checked = self.account.validate_room(token)
                    break
                except Exception:
                    continue
            if checked is None:
                return {**result, "needs_private_room": True}
            settings, _, _ = self.account._request("GET", SIGNAL_PATH + "/settings",
                                                  params={"token": checked["token"]})
            if not isinstance(settings, dict) or settings.get("userId") != checked["service_id"]:
                raise AccountError("typing_identity_changed")
            mode = settings.get("signalingMode") if isinstance(settings, dict) else None
            result["mode"] = mode if mode in {"internal", "external", "conversation_cluster"} else "unknown"
            if mode in {"external", "conversation_cluster"}:
                _websocket_url(settings.get("server"))
                from websockets.sync.client import connect  # noqa: F401
            result["typing_supported"] = supported and public and result["mode"] != "unknown"
            result["status"] = "ok" if result["typing_supported"] else "unavailable"
        except Exception:
            pass
        return result

    def _privacy(self, session):
        data, _, _ = session._request("GET", "/ocs/v2.php/cloud/capabilities")
        supported, public = _profile(data)
        if not supported or not public:
            raise AccountError("typing_not_public")

    def _open_socket(self, session, settings, token):
        if self.websocket_connect is None:
            from websockets.sync.client import connect
        else:
            connect = self.websocket_connect
        base = urlsplit(session.base)
        session.socket = connect(_websocket_url(settings.get("server")), open_timeout=3, close_timeout=1,
                                 max_size=1_000_000, max_queue=16, ping_interval=15, ping_timeout=5,
                                 proxy=None, origin=base.scheme + "://" + base.netloc)
        try:
            welcome = session.receive("welcome")
        except TimeoutError:
            # Older HPBs wait for hello1.0 without sending welcome first.
            welcome = {}
        features = welcome.get("features", []) if isinstance(welcome, dict) else []
        params = settings.get("helloAuthParams")
        if not isinstance(features, list) or not isinstance(params, dict):
            raise AccountError("typing_invalid_auth")
        version = "2.0" if "hello-v2" in features and "2.0" in params else "1.0"
        auth = params.get(version)
        if not isinstance(auth, dict):
            raise AccountError("typing_invalid_auth")
        if version == "2.0":
            ticket = auth.get("token")
            auth = {"token": ticket}
        else:
            ticket = auth.get("ticket")
            if auth.get("userid") != session.identity():
                raise AccountError("typing_identity_changed")
            auth = {"userid": session.identity(), "ticket": ticket}
        if not isinstance(ticket, str) or not 1 <= len(ticket) <= 16384:
            raise AccountError("typing_invalid_auth")
        session.frame({"type": "hello", "hello": {"version": version, "auth": {
            "url": session.base + SIGNAL_PATH + "/backend", "params": auth,
        }}})
        hello = session.receive("hello")
        if not isinstance(hello, dict) or hello.get("userid") != session.identity():
            raise AccountError("typing_identity_changed")
        _session_id(hello.get("sessionid"))
        session.frame({"type": "room", "room": {"roomid": token, "sessionid": session.session_id}})
        joined = session.receive("room")
        if not isinstance(joined, dict) or joined.get("roomid") != token:
            raise AccountError("typing_room_changed")

    def _open(self, token, employee):
        if len(self._sessions) >= MAX_ROOMS:
            raise AccountError("typing_capacity_reached")
        session = _Session(self.settings, getattr(self.account, "transport", None), employee)
        session.can_start = lambda: not self._closed
        self._sessions[token] = session  # Ensure any partially joined session can be cleaned.
        checked, _ = session._private_room(token, employee)
        if checked["service_id"] != self.identity():
            raise AccountError("typing_identity_changed")
        self._privacy(session)
        settings, _, _ = session._request("GET", SIGNAL_PATH + "/settings", params={"token": token})
        if not isinstance(settings, dict) or settings.get("userId") != checked["service_id"]:
            raise AccountError("typing_identity_changed")
        session.mode = settings.get("signalingMode")
        if session.mode not in {"internal", "external", "conversation_cluster"}:
            raise AccountError("typing_mode_unavailable")
        if session.mode != "internal":
            _websocket_url(settings.get("server"))
        joined, _, _ = session._request("POST", ROOM_PATH + session._token(token) + "/participants/active",
                                        json={"force": False})
        if not isinstance(joined, dict) or joined.get("token") != token:
            raise AccountError("typing_invalid_join")
        session.session_id = _session_id(joined.get("sessionId"))
        current, _, _ = session._request("GET", ROOM_PATH + session._token(token))
        if not isinstance(current, dict) or current.get("sessionId") != session.session_id:
            raise AccountError("typing_session_changed")
        if session.mode != "internal":
            self._open_socket(session, settings, token)
        return session

    def _emit(self, session, token, active):
        if active and self._closed:
            raise AccountError("typing_closed")
        self._privacy(session)
        checked, people = session._private_room(token, session.employee)
        service = next(person for person in people if person["actorId"] == checked["service_id"])
        own_ids = service.get("sessionIds")
        if not isinstance(own_ids, list) or session.session_id not in own_ids:
            raise AccountError("typing_session_changed")
        if active and self._closed:
            raise AccountError("typing_closed")
        kind = "startedTyping" if active else "stoppedTyping"
        if session.mode != "internal":
            # Room relay avoids confusing HPB session IDs with PHP roster IDs.
            # Exactly two local users have just been revalidated above.
            session.frame({"type": "message", "message": {
                "recipient": {"type": "room"}, "data": {"type": kind},
            }})
        else:
            peer = next(person for person in people if person["actorId"] == session.employee)
            peers = peer.get("sessionIds")
            if not isinstance(peers, list) or len(peers) > MAX_PEER_SESSIONS:
                raise AccountError("typing_invalid_peers")
            messages = [{"ev": "message", "fn": json.dumps({"to": _session_id(peer), "type": kind}),
                         "sessionId": session.session_id} for peer in dict.fromkeys(peers)]
            if messages:
                session._request("POST", SIGNAL_PATH + "/" + session._token(token),
                                 data={"messages": json.dumps(messages)})
        session.refreshed = time.monotonic()

    def _start_internal_poll(self, session, token):
        def drain():
            try:
                while not session.stopped.is_set() and time.monotonic() - session.refreshed <= 20:
                    session._private_room(token, session.employee)
                    # Official long-poll also refreshes this owned session's lastPing.
                    # Ignore all incoming content; it never enters support memory.
                    session._request("GET", SIGNAL_PATH + "/" + session._token(token), timeout=35)
                    session.stopped.wait(0.1)
            except Exception:
                pass
            finally:
                self._expire(session, token)
        threading.Thread(target=drain, name="talk-native-typing", daemon=True).start()

    def _start_socket_poll(self, session, token):
        def drain():
            try:
                while not session.stopped.is_set() and time.monotonic() - session.refreshed <= 20:
                    try:
                        raw = session.socket.recv(timeout=1)
                    except TimeoutError:
                        continue
                    if not isinstance(raw, str) or len(raw) > 1_000_000:
                        break
                    data = json.loads(raw)
                    if not isinstance(data, dict) or data.get("type") in {"bye", "error"}:
                        break
            except Exception:
                pass
            finally:
                self._expire(session, token)
        threading.Thread(target=drain, name="talk-native-typing-ws", daemon=True).start()

    def _expire(self, session, token):
        with self._lock:
            if self._sessions.get(token) is session:
                self._sessions.pop(token, None)
                self._finish(session, token)

    def _finish(self, session, token, *, broadcast=True):
        if broadcast and session.session_id:
            try:
                self._emit(session, token, False)
            except Exception:
                pass
        session.stopped.set()
        if session.socket is not None:
            try:
                session.frame({"type": "room", "room": {"roomid": ""}})
                session.frame({"type": "bye"})
            except Exception:
                pass
            try:
                session.socket.close()
            except Exception:
                pass
        if session.session_id:
            try:
                current, _, _ = session._request("GET", ROOM_PATH + session._token(token))
                if isinstance(current, dict) and current.get("sessionId") == session.session_id:
                    session._request("DELETE", ROOM_PATH + session._token(token) + "/participants/active")
            except Exception:
                pass
        session.http.close()

    def set_typing(self, room, *, expected_employee, active):
        """Start/refresh or stop only our transient personal-room typing session."""
        with self._lock:
            session = self._sessions.get(room)
            try:
                if self._closed:
                    return False
                self.account._token(room)
                if type(active) is not bool or not isinstance(expected_employee, str) or not expected_employee:
                    raise AccountError("typing_invalid_request")
                if session is not None and session.employee != expected_employee:
                    raise AccountError("typing_employee_changed")
                if not active:
                    if session is not None:
                        self._sessions.pop(room, None)
                        self._finish(session, room)
                    return True
                if session is not None and session.stopped.is_set():
                    raise AccountError("typing_session_stopped")
                session = session or self._open(room, expected_employee)
                self._emit(session, room, True)
                if session.mode == "internal" and not getattr(session, "polling", False):
                    session.polling = True
                    self._start_internal_poll(session, room)
                elif session.mode != "internal" and not getattr(session, "polling", False):
                    session.polling = True
                    self._start_socket_poll(session, room)
                return True
            except Exception:
                session = self._sessions.pop(room, None)
                if session is not None:
                    self._finish(session, room, broadcast=False)
                return False

    def close(self):
        self._closed = True  # Stop a concurrent in-flight start before it broadcasts.
        deadline = time.monotonic() + CLOSE_TIMEOUT
        def finish(room, session):
            try:
                self._finish(session, room)
            except Exception:
                pass
        def cleanup():
            # Waiting for an in-flight network operation's lock is also optional.
            # The closed flag already forbids it from starting a late indicator.
            with self._lock:
                sessions, self._sessions = self._sessions, {}
            threads = [threading.Thread(target=finish, args=(room, session), daemon=True)
                       for room, session in sessions.items()]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        worker = threading.Thread(target=cleanup, name="talk-native-typing-close", daemon=True)
        worker.start()
        worker.join(timeout=max(0, deadline - time.monotonic()))
