"""Optional native Talk typing. Never writes, edits or deletes a chat message."""

import time
from uuid import uuid4

from sqlalchemy import select

from app.core.logging import event
from app.models import TalkAccount, TalkInbox, TalkRoom

DELAY_SECONDS = 1
REFRESH_SECONDS = 3
ACTIVE_STATES = {"ACTIVE", "STARTING", "REFRESHING", "STOPPING"}


def start_progress(inbox):
    previous = inbox.progress if isinstance(inbox.progress, dict) else {}
    if previous.get("mode") == "native" and previous.get("state") in ACTIVE_STATES:
        return
    inbox.progress = {"mode": "native", "state": "WAITING", "started_at": time.time()}


def recover_progress(factory, settings, service_id):
    """An old process's signaling sessions expire; chat placeholders stay untouched."""
    with factory() as session, session.begin():
        rows = session.scalars(select(TalkInbox).join(TalkRoom).join(TalkAccount).where(
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"), TalkAccount.username == service_id,
            TalkInbox.progress["state"].as_string().in_(
                ["WAITING", "SENDING", "SHOWN", "UNKNOWN", "CLEANING", *ACTIVE_STATES],
            ),
        )).all()
        for row in rows:
            prior = row.progress or {}
            if prior.get("mode") == "native" and row.status in {"RUNNING", "PENDING"}:
                row.progress = {"mode": "native", "state": "WAITING", "started_at": time.time()}
            else:
                row.progress = {"mode": "native", "state": "STOPPED"}


def _current(session, inbox_id, settings, service_id, employee_id, room_token):
    return session.scalar(select(TalkInbox.id).join(TalkRoom).join(TalkAccount).where(
        TalkInbox.id == inbox_id, TalkInbox.status == "RUNNING", TalkInbox.sender_id == employee_id,
        TalkRoom.employee_id == employee_id, TalkRoom.room_token == room_token, TalkRoom.enabled.is_(True),
        TalkAccount.enabled.is_(True), TalkAccount.username == service_id,
        TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
    )) is not None


def tick_progress(factory, settings, client, *, clock=None):
    """Refresh one native indicator or stop it; errors cannot affect support delivery."""
    if settings.nextcloud_backend != "account":
        return False
    selected = None
    try:
        service_id = client.identity()
        stamp = (clock or time.time)()
        with factory() as session, session.begin():
            rows = session.execute(select(TalkInbox, TalkRoom, TalkAccount).join(
                TalkRoom, TalkInbox.room_id == TalkRoom.id,
            ).join(TalkAccount, TalkRoom.account_id == TalkAccount.id).where(
                TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"), TalkAccount.username == service_id,
                TalkInbox.progress["state"].as_string().in_(["WAITING", "ACTIVE"]),
            ).order_by(TalkInbox.created_at, TalkInbox.id).with_for_update(skip_locked=True)).all()
            for row, room, account in rows:
                state = dict(row.progress or {})
                if state.get("mode") != "native":
                    row.progress = {"mode": "native", "state": "STOPPED"}
                    return True  # Retire 0.3.3 placeholders without touching chat history.
                valid = (row.status == "RUNNING" and room.enabled and account.enabled
                         and row.sender_id == room.employee_id and room.employee_id != service_id)
                if state["state"] == "WAITING":
                    if not valid:
                        row.progress = {"mode": "native", "state": "STOPPED"}
                        return True
                    started = state.get("started_at")
                    if type(started) not in {int, float} or stamp - started < DELAY_SECONDS:
                        continue
                elif valid:
                    refreshed = state.get("refreshed_at", 0)
                    if type(refreshed) in {int, float} and stamp - refreshed < REFRESH_SECONDS:
                        continue
                operation = str(uuid4())
                row.progress = {**state, "state": "STARTING" if valid else "STOPPING", "operation": operation}
                selected = (row.id, room.room_token, room.employee_id, operation, valid, state)
                break
        if selected is None:
            return False
        inbox_id, room_token, employee_id, operation, active, state = selected
        if active:
            with factory() as session:
                active = _current(session, inbox_id, settings, service_id, employee_id, room_token)
        success = client.set_typing(room_token, expected_employee=employee_id, active=active)
        if active:
            # Setup can finish after the model or after a local administrator disables the room.
            with factory() as session:
                still_running = _current(session, inbox_id, settings, service_id, employee_id, room_token)
            if not still_running:
                client.set_typing(room_token, expected_employee=employee_id, active=False)
                active = False
        result = {"mode": "native", "state": "ACTIVE" if active and success else "UNAVAILABLE" if active else "STOPPED",
                  "started_at": state.get("started_at", stamp), "refreshed_at": stamp}
        with factory() as session, session.begin():
            row = session.get(TalkInbox, inbox_id)
            if (row.progress or {}).get("operation") == operation:
                row.progress = result
        return True
    except Exception as error:
        # A failed optional start must not remain claimed forever or poison the inbox.
        if selected is not None:
            try:
                with factory() as session, session.begin():
                    row = session.get(TalkInbox, selected[0])
                    if (row.progress or {}).get("operation") == selected[3]:
                        row.progress = {"mode": "native", "state": "UNAVAILABLE"}
                client.set_typing(selected[1], expected_employee=selected[2], active=False)
            except Exception:
                pass
        try:
            event("talk.typing_unavailable", error_code=type(error).__name__)
        except Exception:
            pass
        return False
