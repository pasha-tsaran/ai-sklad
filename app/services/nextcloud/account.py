"""Authenticated personal Talk transport; never acts in a group or a remote room.

Conversation and participant API v4 (Nextcloud 22+), chat API v1.
https://nextcloud-talk.readthedocs.io/en/stable/conversation/
https://nextcloud-talk.readthedocs.io/en/stable/participant/
https://nextcloud-talk.readthedocs.io/en/stable/chat/
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlsplit

import httpx

MAX_RESPONSE_BYTES = 4_000_000
MAX_IMAGE_BYTES = 8_000_000
MAX_SEQUENCE = 2**63 - 1
_TOKEN = re.compile(r"[a-z0-9]{4,30}")


class AccountError(ValueError):
    """Safe machine code only; response bodies and credentials never reach logs."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def sequence(value, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).isdigit():
        raise AccountError("invalid_message_sequence")
    result = int(value)
    if not (0 if allow_zero else 1) <= result <= MAX_SEQUENCE:
        raise AccountError("invalid_message_sequence")
    return result


class AccountClient:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport
        self.base = settings.nextcloud_base_url.rstrip("/")
        parsed = urlsplit(self.base)
        if (parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise AccountError("invalid_server_url")
        self.username = settings.nextcloud_username
        self.password = settings.nextcloud_app_password.get_secret_value()
        if not self.username or not self.password:
            raise AccountError("account_credentials_missing")
        self._uid = None
        self._employees = {}
        self.server_time = None

    def _request(self, method, path, *, ocs=True, acceptable=(200, 201), **kwargs):
        # A fresh client avoids accidental cookie/session state between conversations.
        with httpx.Client(auth=(self.username, self.password), verify=True, follow_redirects=False,
                          timeout=self.settings.integration_timeout_seconds, transport=self.transport,
                          headers={"OCS-APIRequest": "true", "Accept": "application/json"}) as client:
            with client.stream(method, self.base + path, **kwargs) as response:
                if response.status_code not in acceptable:
                    if response.status_code in {401, 403, 404}:
                        raise AccountError("access_denied")
                    raise AccountError("nextcloud_http_error")
                data = bytearray()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise AccountError("response_too_large")
                headers, status = dict(response.headers), response.status_code
        if not ocs or status == 304:
            return None, headers, status
        try:
            envelope = json.loads(data)["ocs"]
            if envelope["meta"]["statuscode"] not in {100, 200, 201}:
                raise AccountError("nextcloud_ocs_error")
            return envelope["data"], headers, status
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, AccountError):
                raise
            raise AccountError("invalid_nextcloud_response") from None

    def identity(self):
        if self._uid is None:
            data, headers, _ = self._request("GET", "/ocs/v2.php/cloud/user")
            uid = data.get("id") if isinstance(data, dict) else None
            if not isinstance(uid, str) or not uid or len(uid) > 255:
                raise AccountError("invalid_account_identity")
            self._uid = uid
            try:
                stamp = parsedate_to_datetime(headers.get("date", ""))
                self.server_time = stamp.astimezone(timezone.utc).replace(microsecond=0)
            except (TypeError, ValueError, AttributeError):
                self.server_time = datetime.now(timezone.utc).replace(microsecond=0)
        return self._uid

    def _token(self, token):
        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            raise AccountError("invalid_room_token")
        allowed = self.settings.nextcloud_allowed_rooms
        if allowed and token not in allowed:
            raise AccountError("room_not_allowed")
        return quote(token, safe="")

    def rooms(self):
        data, _, _ = self._request("GET", "/ocs/v2.php/apps/spreed/api/v4/room",
                                   params={"noStatusUpdate": 1, "includeStatus": "false"})
        if not isinstance(data, list) or len(data) > 10_000 or any(not isinstance(room, dict) for room in data):
            raise AccountError("invalid_room_list")
        return data

    def validate_room(self, token, expected_employee=None):
        encoded = self._token(token)
        uid = self.identity()
        path = "/ocs/v2.php/apps/spreed/api/v4/room/" + encoded
        room, _, _ = self._request("GET", path)
        if (not isinstance(room, dict) or room.get("token") != token or type(room.get("type")) is not int
                or room["type"] != 1 or room.get("remoteServer") or room.get("remoteToken")
                or room.get("isFederated") or room.get("isFederatedConversation")
                or room.get("objectType") in {"federated", "remote", "note-to-self"}
                or room.get("readOnly") not in (None, 0)):
            raise AccountError("room_not_private")
        people, _, _ = self._request("GET", path + "/participants", params={"includeStatus": "false"})
        if not isinstance(people, list) or len(people) != 2:
            raise AccountError("room_not_private")
        ids = []
        for person in people:
            if (not isinstance(person, dict) or person.get("actorType") != "users"
                    or not isinstance(person.get("actorId"), str) or not person["actorId"]
                    or len(person["actorId"]) > 255 or type(person.get("participantType")) is not int
                    or person["participantType"] not in {1, 2, 3}
                    or person.get("remoteServer") or person.get("remoteId")):
                raise AccountError("room_not_private")
            ids.append(person["actorId"])
        if len(set(ids)) != 2 or uid not in ids:
            raise AccountError("room_not_private")
        employee = next(actor for actor in ids if actor != uid)
        expected_employee = expected_employee if expected_employee is not None else self._employees.get(token)
        if expected_employee is not None and employee != expected_employee:
            raise AccountError("room_employee_changed")
        self._employees[token] = employee
        return {"token": token, "type": 1, "employee_id": employee, "service_id": uid}

    def messages(self, token, cursor, limit=None):
        encoded = self._token(token)
        cursor = sequence(cursor, allow_zero=True)
        limit = self.settings.nextcloud_batch_size if limit is None else limit
        if type(limit) is not int or not 1 <= limit <= 200:
            raise AccountError("invalid_batch_size")
        data, headers, status = self._request(
            "GET", "/ocs/v2.php/apps/spreed/api/v1/chat/" + encoded, acceptable=(200, 304),
            params={"lookIntoFuture": 1, "lastKnownMessageId": cursor, "limit": limit, "timeout": 0,
                    "includeLastKnown": 0, "setReadMarker": 0, "noStatusUpdate": 1,
                    "markNotificationsAsRead": 0},
        )
        data = [] if status == 304 else data
        if not isinstance(data, list) or len(data) > limit:
            raise AccountError("invalid_message_page")
        previous = cursor
        for message in data:
            if not isinstance(message, dict) or message.get("token") != token:
                raise AccountError("invalid_message_page")
            current = sequence(message.get("id"))
            if current <= previous:
                raise AccountError("unordered_message_page")
            timestamp = message.get("timestamp")
            if type(timestamp) is not int or not 0 <= timestamp <= 253_402_300_799:
                raise AccountError("invalid_message_timestamp")
            previous = current
        next_cursor = sequence(headers.get("x-chat-last-given", previous), allow_zero=True)
        if next_cursor < previous:
            raise AccountError("invalid_message_cursor")
        return {"messages": data, "cursor": next_cursor}

    def send(self, room, content, message_id, *, expected_employee=None):
        self.validate_room(room, expected_employee=expected_employee)
        if not isinstance(content, str) or not content.strip() or len(content) > 32_000:
            raise AccountError("invalid_outgoing_message")
        data, _, _ = self._request("POST", "/ocs/v2.php/apps/spreed/api/v1/chat/" + self._token(room),
                                   json={"message": content,
                                         "referenceId": hashlib.sha256(message_id.encode()).hexdigest()})
        if not isinstance(data, dict):
            raise AccountError("invalid_send_result")
        return str(sequence(data.get("id")))

    def send_image(self, room, content, message_id, attachment, *, expected_employee=None):
        checked = self.validate_room(room, expected_employee=expected_employee)
        raw = attachment.data
        if (not isinstance(raw, bytes) or not raw or len(raw) > MAX_IMAGE_BYTES
                or attachment.content_type != "image/png"
                or hashlib.sha256(raw).hexdigest() != attachment.sha256):
            raise AccountError("invalid_illustration_snapshot")
        return self._share_attachment(room, content, message_id, attachment, checked, ".png")

    def send_file(self, room, content, message_id, attachment, *, expected_employee=None):
        """Share only a server-generated immutable mail draft in the same private DM."""
        checked = self.validate_room(room, expected_employee=expected_employee)
        raw = attachment.data
        if (not isinstance(raw, bytes) or not raw or len(raw) > 2_000_000
                or attachment.content_type != "message/rfc822" or not attachment.filename.endswith(".eml")
                or hashlib.sha256(raw).hexdigest() != attachment.sha256):
            raise AccountError("invalid_mail_draft_snapshot")
        return self._share_attachment(room, content, message_id, attachment, checked, ".eml")

    def _share_attachment(self, room, content, message_id, attachment, checked, extension):
        raw = attachment.data
        folder = "AI-Support-" + hashlib.sha256(room.encode()).hexdigest()[:24]
        filename = ("support-request-" if extension == ".eml" else "") + hashlib.sha256(message_id.encode()).hexdigest() + extension
        dav = "/remote.php/dav/files/" + quote(checked["service_id"], safe="") + "/" + folder
        self._request("MKCOL", dav, ocs=False, acceptable=(201, 405))
        self._request("PUT", dav + "/" + filename, ocs=False, acceptable=(201, 204), content=raw,
                      headers={"Content-Type": attachment.content_type})
        # Sharing reveals the file. Recheck after upload, immediately before that POST.
        self.validate_room(room, expected_employee=checked["employee_id"])
        data, _, _ = self._request("POST", "/ocs/v2.php/apps/files_sharing/api/v1/shares",
                                   data={"shareType": "10", "shareWith": room,
                                         "path": "/" + folder + "/" + filename,
                                         "referenceId": hashlib.sha256(message_id.encode()).hexdigest(),
                                         "talkMetaData": json.dumps({"caption": content}, ensure_ascii=False)})
        if not isinstance(data, dict):
            raise AccountError("invalid_share_result")
        return str(sequence(data.get("id")))
