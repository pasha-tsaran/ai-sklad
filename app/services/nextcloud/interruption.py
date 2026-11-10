"""Pause only unstarted private replies; resume their immutable unsent rows.

The profile marker is server-owned, bounded and scoped to the room, account and
incident. A possibly accepted external POST (SENDING/UNKNOWN) is never replayed.
"""

import copy
import re

from sqlalchemy import select

from app.models import Delivery, Incident, Message, SolutionAttempt
from app.models.talk import TalkInbox

ACTIVE = "_active_response"
PAUSED = "_paused_responses"
TERMINAL_DELIVERIES = ("SENT", "PAUSED", "CANCELLED")
MAX_PAUSED_RESPONSES = 3
MAX_RESPONSE_ITEMS = 256


def is_resume_request(message):
    text = " ".join(re.sub(r"[^\w\s]", " ", str(message).casefold()).split())
    text = re.sub(r"^(?:пожалуйста\s+)", "", text)
    text = re.sub(r"\s+пожалуйста$", "", text)
    return bool(re.fullmatch(
        r"(?:продолжай|продолжи|продолжить)(?:\s+(?:предыдущий\s+ответ|предыдущую\s+инструкцию|"
        r"с\s+места\s+остановки))?|(?:закончи|закончить|допиши|допечатай)\s+(?:предыдущий\s+ответ|"
        r"предыдущую\s+инструкцию)|вернись\s+к\s+предыдущему\s+ответу", text,
    ))


def _rows(session, room, snapshot, statuses):
    ids = snapshot.get("delivery_ids", [])
    if (not isinstance(ids, list) or not ids or len(ids) > MAX_RESPONSE_ITEMS
            or any(not isinstance(value, str) for value in ids)
            or snapshot.get("incident_id") != room.incident_id):
        return []
    return session.scalars(select(Delivery).join(Message).where(
        Delivery.id.in_(ids), Delivery.account_id == room.account_id,
        Delivery.room_token == room.room_token, Delivery.status.in_(statuses),
        Message.incident_id == room.incident_id, Message.role == "assistant",
    ).order_by(Message.sequence)).all()


def record_response(session, room, incident, inbox, first_sequence):
    """Record this committed turn's exact output, without copying photo bytes."""
    ids = session.scalars(select(Delivery.id).join(Message).where(
        Delivery.account_id == room.account_id, Delivery.room_token == room.room_token,
        Message.incident_id == incident.id, Message.sequence > first_sequence,
        Message.role == "assistant", Delivery.status == "PENDING",
    ).order_by(Message.sequence).limit(MAX_RESPONSE_ITEMS + 1)).all()
    profile = copy.deepcopy(room.profile or {})
    profile.pop(ACTIVE, None)
    if ids and len(ids) <= MAX_RESPONSE_ITEMS:
        attempt = session.scalar(select(SolutionAttempt).where(
            SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "WAITING_FEEDBACK",
        ))
        profile[ACTIVE] = {
            "id": ids[0], "delivery_ids": list(ids), "incident_id": incident.id,
            "remote_sequence": inbox.remote_sequence,
            "flow": {key: value for key, value in copy.deepcopy(room.flow or {}).items() if key != "_processing"},
            "state": incident.state, "pending": incident.pending_action, "result": incident.result,
            "attempt_id": attempt.id if attempt else None,
            "resumable": not (room.flow or {}).get("escalation") and incident.state not in {"RESOLVED", "ESCALATED"},
        }
    room.profile = profile


def _legacy_response(session, room):
    """Recognize only an unambiguous 0.3.9 current approved-guide outbox."""
    flow = room.flow or {}
    source = flow.get("source")
    incident = session.get(Incident, room.incident_id) if room.incident_id else None
    if (not isinstance(source, dict) or flow.get("escalation") or incident is None
            or incident.state != "WAITING_FEEDBACK" or incident.pending_action not in {None, "PROCESSING"}):
        return None
    attempt = session.scalar(select(SolutionAttempt).where(
        SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "WAITING_FEEDBACK",
    ))
    if attempt is None or attempt.evidence.get("talk_source") != source:
        return None
    user = session.scalar(select(Message).where(
        Message.incident_id == incident.id, Message.role == "user",
    ).order_by(Message.sequence.desc()).limit(1))
    if user is None:
        return None
    previous = session.scalar(select(TalkInbox).where(
        TalkInbox.room_id == room.id, TalkInbox.status == "DONE", TalkInbox.sender_id == room.employee_id,
        TalkInbox.content == user.content,
    ).order_by(TalkInbox.remote_sequence.desc()).limit(1))
    if previous is None:
        return None
    rows = session.scalars(select(Delivery).join(Message).where(
        Delivery.account_id == room.account_id, Delivery.room_token == room.room_token,
        Delivery.status == "PENDING", Message.incident_id == incident.id, Message.role == "assistant",
        Message.sequence > user.sequence,
    ).order_by(Message.sequence).limit(MAX_RESPONSE_ITEMS + 1)).all()
    if not rows or len(rows) > MAX_RESPONSE_ITEMS or any(row.provenance != source for row in rows):
        return None
    return {"id": rows[0].id, "delivery_ids": [row.id for row in rows], "incident_id": incident.id,
            "remote_sequence": previous.remote_sequence,
            "flow": {key: value for key, value in copy.deepcopy(flow).items() if key != "_processing"}, "resumable": True,
            "state": incident.state, "pending": None, "result": incident.result, "attempt_id": attempt.id}


def pause_due_to_input(session, room, inbox=None):
    """Called before claiming output or a newer input, inside its transaction."""
    profile = copy.deepcopy(room.profile or {})
    active = profile.get(ACTIVE)
    if not isinstance(active, dict):
        active = _legacy_response(session, room)
        if active is None:
            return False
    sequence = active.get("remote_sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool):
        return False
    if inbox is None:
        inbox = session.scalar(select(TalkInbox).where(
            TalkInbox.room_id == room.id, TalkInbox.sender_id == room.employee_id,
            TalkInbox.remote_sequence > sequence, TalkInbox.status.in_(["PENDING", "RUNNING"]),
        ).order_by(TalkInbox.remote_sequence).limit(1))
    if inbox is None or inbox.sender_id != room.employee_id or inbox.remote_sequence <= sequence:
        return False
    rows = _rows(session, room, active, ["PENDING"])
    if not rows:
        return False
    paused = [item for item in profile.get(PAUSED, []) if isinstance(item, dict)]
    snapshot = {**active, "delivery_ids": [row.id for row in rows]}
    for row in rows:
        row.status = "PAUSED" if active.get("resumable", True) else "CANCELLED"
        session.get(Message, row.message_id).delivery_status = row.status
    if active.get("resumable", True):
        paused.append(snapshot)
    while len(paused) > MAX_PAUSED_RESPONSES:
        expired = paused.pop(0)
        for row in _rows(session, room, expired, ["PAUSED"]):
            row.status = "CANCELLED"
            session.get(Message, row.message_id).delivery_status = "CANCELLED"
    profile.pop(ACTIVE, None)
    profile[PAUSED] = paused
    room.profile = profile
    return True


def latest_paused(session, room):
    for snapshot in reversed((room.profile or {}).get(PAUSED, [])):
        if isinstance(snapshot, dict) and _rows(session, room, snapshot, ["PAUSED"]):
            return copy.deepcopy(snapshot)
    return None


def has_unfinished_guide(session, room, attempt):
    """Explicit feedback may resolve early; an unseen guide cannot seed a cache."""
    source = attempt.evidence.get("talk_source")
    if not source:
        return False
    return any(row.provenance == source for row in session.scalars(select(Delivery).join(Message).where(
        Delivery.account_id == room.account_id, Delivery.room_token == room.room_token,
        Delivery.status.not_in(["SENT", "CANCELLED"]), Message.incident_id == attempt.incident_id,
        Delivery.provenance.is_not(None),
    )))


def resume_response(session, room, incident, inbox, snapshot):
    """Restore only a verified snapshot still owned by this current private DM."""
    current = latest_paused(session, room)
    if current is None or current.get("id") != snapshot.get("id"):
        return False
    rows = _rows(session, room, current, ["PAUSED"])
    if not rows or incident.id != current.get("incident_id"):
        return False
    old_attempt = session.get(SolutionAttempt, current.get("attempt_id")) if current.get("attempt_id") else None
    if old_attempt is not None and (old_attempt.incident_id != incident.id
                                    or old_attempt.status not in {"UNASSESSED", "WAITING_FEEDBACK"}):
        return False
    if old_attempt is not None:
        for other in session.scalars(select(SolutionAttempt).where(
            SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "WAITING_FEEDBACK",
            SolutionAttempt.id != old_attempt.id,
        )):
            other.status = "UNASSESSED"
        session.flush()
        old_attempt.status = "WAITING_FEEDBACK"
    for row in rows:
        row.status = "PENDING"
        session.get(Message, row.message_id).delivery_status = "QUEUED"
    room.flow = copy.deepcopy(current.get("flow") or {})
    incident.state = current.get("state", "DIAGNOSING")
    incident.pending_action, incident.result = current.get("pending"), current.get("result")
    profile = copy.deepcopy(room.profile or {})
    profile[PAUSED] = [item for item in profile.get(PAUSED, []) if item.get("id") != current["id"]]
    profile[ACTIVE] = {**current, "remote_sequence": inbox.remote_sequence}
    room.profile = profile
    return True
