"""One-use hostname capture, bound to the requesting private Talk incident."""

import copy
import hashlib
import re
import secrets
from datetime import timedelta, timezone
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from sqlalchemy import select

from app.models import Conversation, DeviceCollection, Incident, TalkAccount, TalkInbox, TalkRoom, now
from app.services.chat.channel_profile import channel_profile, remember_computer
from app.services.chat.policies import FORM_ACTION, TERMINAL_STATES

TTL_SECONDS = 300
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
_COMPUTER = re.compile(r"[A-Za-z0-9_.-]{1,63}")


def reject(status=400):
    raise HTTPException(status, "Device collection unavailable", headers={
        "Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
    })


def public_base(settings):
    value = getattr(settings, "device_collection_base_url", "")
    if settings.nextcloud_backend != "account" or not isinstance(value, str) or not value:
        reject(404)
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        reject(404)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or "\\" in value
            or any(ord(char) < 33 or ord(char) == 127 for char in value)):
        reject(404)
    return value.rstrip("/")


def token_digest(token):
    if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
        reject()
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _binding(session, settings, room, incident, *, service_uid=None, requester_uid=None):
    account = session.get(TalkAccount, room.account_id) if room is not None else None
    if (account is None or not account.enabled or not room.enabled or incident is None
            or room.incident_id != incident.id or account.server_url != settings.nextcloud_base_url.rstrip("/")
            or not room.employee_id or room.employee_id == account.username
            or (service_uid is not None and service_uid != account.username)
            or (requester_uid is not None and requester_uid != room.employee_id)):
        reject()
    conversation = session.get(Conversation, incident.conversation_id)
    principal = str(uuid5(NAMESPACE_URL, account.server_url + "/" + account.username + "/"
                          + room.room_token + "/" + room.employee_id))
    if (conversation is None or conversation.channel != "nextcloud_talk"
            or conversation.room_token != room.room_token or conversation.user_id != principal):
        reject()
    return account


def issue_collection(session, settings, room, incident):
    base = public_base(settings)
    account = _binding(session, settings, room, incident)
    if incident.pending_action != FORM_ACTION or incident.state in TERMINAL_STATES:
        reject()
    token = secrets.token_urlsafe(32)
    issued = now()
    row = DeviceCollection(account_id=account.id, room_id=room.id, incident_id=incident.id,
                           service_uid=account.username, requester_uid=room.employee_id,
                           token_hash=token_digest(token), created_at=issued,
                           expires_at=issued + timedelta(seconds=TTL_SECONDS))
    session.add(row)
    session.flush()
    return row.id, base + "/device/collect#token=" + token


def _lookup(session, settings, token, *, lock=False):
    public_base(settings)
    statement = select(DeviceCollection).where(DeviceCollection.token_hash == token_digest(token))
    row = session.scalar(statement.with_for_update() if lock else statement)
    if row is None or _utc(row.expires_at) <= now():
        reject()
    room_query = select(TalkRoom).where(TalkRoom.id == row.room_id)
    room = session.scalar(room_query.with_for_update() if lock else room_query)
    incident_query = select(Incident).where(Incident.id == row.incident_id)
    incident = session.scalar(incident_query.with_for_update() if lock else incident_query)
    account = _binding(session, settings, room, incident, service_uid=row.service_uid, requester_uid=row.requester_uid)
    if account.id != row.account_id:
        reject()
    if row.consumed_at is None:
        if not isinstance(room.flow, dict):
            reject()
        escalation = room.flow.get("escalation", {})
        if (incident.pending_action != FORM_ACTION or incident.state in TERMINAL_STATES
                or not isinstance(escalation, dict) or escalation.get("stage") != "DEVICE"
                or escalation.get("collection_id") != row.id
                or not isinstance(escalation.get("intake_fields", {}), dict)):
            reject()
        # A hostname/cancel already received in Talk must win the transition.
        # Otherwise that earlier message could be mistaken for the next topic.
        waiting = session.scalar(select(TalkInbox.id).where(
            TalkInbox.room_id == room.id, TalkInbox.sender_id == row.requester_uid,
            TalkInbox.status.in_(["PENDING", "RUNNING"]),
        ).limit(1))
        if waiting is not None:
            reject()
    return row, room, incident


def collection_status(session, settings, token):
    row, _, _ = _lookup(session, settings, token)
    return {"status": "complete" if row.consumed_at is not None else "pending"}


def capture_computer(session, settings, token, computer_name):
    if (not isinstance(computer_name, str) or _COMPUTER.fullmatch(computer_name) is None
            or not any(char.isalnum() for char in computer_name)):
        reject()
    row, room, incident = _lookup(session, settings, token, lock=True)
    if row.consumed_at is not None:
        if row.computer_name != computer_name:
            reject(409)
        return {"status": "complete"}
    # Import the pure presenter locally: it neither calls a model nor sends Talk.
    from app.services.chat.talk_dialogue import _profile_form_request
    from app.services.chat.service import messenger
    flow = copy.deepcopy(room.flow)
    escalation = flow["escalation"]
    fields = dict(escalation.get("intake_fields", {}))
    fields["computer_name"] = computer_name
    flow["escalation"] = {**escalation, "stage": "FORM", "intake_fields": fields,
                          "intake_requested_field": None}
    if not remember_computer(room, incident.id, computer_name):
        reject()
    room.profile = {**room.profile, "device_source": "windows_device_collector"}
    turn = _profile_form_request(flow, {"channel_profile": channel_profile(room, incident.id)})
    room.flow = turn.flow
    row.computer_name, row.consumed_at = computer_name, now()
    incident.version += 1
    messenger.send_blocks(session, incident.id, turn.blocks)
    return {"status": "complete"}
