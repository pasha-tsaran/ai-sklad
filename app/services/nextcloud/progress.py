"""Best-effort temporary chat progress, outside support memory and delivery queues."""

import time

from sqlalchemy import select

from app.core.logging import event
from app.models import TalkAccount, TalkInbox, TalkRoom
from app.services.nextcloud.account import AccountClient, PROGRESS_MESSAGE

DELAY_SECONDS = 2
TERMINAL = {"DONE", "FAILED", "IGNORED"}


def start_progress(inbox):
    previous = inbox.progress or {}
    if previous.get("state") in {"SENDING", "SHOWN", "UNKNOWN", "CLEANING"}:
        return
    inbox.progress = {"state": "WAITING", "started_at": time.time()}


def recover_progress(factory, settings, service_id):
    with factory() as session, session.begin():
        rows = session.scalars(select(TalkInbox).join(TalkRoom).join(TalkAccount).where(
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"), TalkAccount.username == service_id,
            TalkInbox.progress["state"].as_string().in_(["SENDING", "CLEANING"]),
        )).all()
        for row in rows:
            state = dict(row.progress)
            state["state"] = "UNKNOWN" if state["state"] == "SENDING" else "SHOWN"
            row.progress = state


def tick_progress(factory, settings, client=None, *, clock=None):
    """Show one delayed marker or clean one finished marker. Failures never block replies."""
    if settings.nextcloud_backend != "account":
        return False
    client = client or AccountClient(settings)
    try:
        service_id = client.identity()
        current_time = (clock or time.time)()
        selected = None
        with factory() as session, session.begin():
            rows = session.scalars(select(TalkInbox).join(TalkRoom).join(TalkAccount).where(
                TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
                TalkAccount.username == service_id, TalkAccount.enabled.is_(True), TalkRoom.enabled.is_(True),
                TalkInbox.progress["state"].as_string().in_(["WAITING", "SHOWN", "UNKNOWN"]),
            ).order_by(TalkInbox.created_at, TalkInbox.id).with_for_update(skip_locked=True)).all()
            for row in rows:
                state = dict(row.progress)
                if state["state"] == "WAITING" and row.status in TERMINAL:
                    row.progress = {**state, "state": "CLEANED"}
                    return True  # Fast answers get no extra chat message.
                if state["state"] == "UNKNOWN" and row.status in TERMINAL:
                    row.progress = {**state, "state": "ABANDONED"}
                    return True  # Never blindly repost a timed-out status POST.
                started = state.get("started_at")
                show = (state["state"] == "WAITING" and row.status == "RUNNING"
                        and type(started) in {int, float} and current_time - started >= DELAY_SECONDS)
                clean = state["state"] == "SHOWN" and row.status in TERMINAL
                if not (show or clean):
                    continue
                room = session.get(TalkRoom, row.room_id)
                if row.sender_id != room.employee_id or room.employee_id == service_id:
                    row.progress = {**state, "state": "ABANDONED"}
                    return True
                row.progress = {**state, "state": "SENDING" if show else "CLEANING"}
                selected = (row.id, room.room_token, room.employee_id, state, show)
                break
        if selected is None:
            return False
        inbox_id, room_token, employee_id, state, show = selected
        if show:
            if not client.supports_progress_cleanup():
                result = {**state, "state": "UNSUPPORTED"}
            else:
                # The model can finish while the capabilities check is in flight.
                with factory() as session:
                    still_running = session.scalar(select(TalkInbox.id).join(TalkRoom).join(TalkAccount).where(
                        TalkInbox.id == inbox_id, TalkInbox.status == "RUNNING",
                        TalkInbox.sender_id == employee_id, TalkRoom.employee_id == employee_id,
                        TalkRoom.room_token == room_token, TalkRoom.enabled.is_(True),
                        TalkAccount.enabled.is_(True), TalkAccount.username == service_id,
                        TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
                    )) is not None
                if not still_running:
                    result = {**state, "state": "CLEANED"}
                else:
                    try:
                        remote = client.send(room_token, PROGRESS_MESSAGE, "progress:" + inbox_id,
                                             expected_employee=employee_id)
                        result = {**state, "state": "SHOWN", "remote_id": remote}
                    except Exception:
                        result = {**state, "state": "UNKNOWN"}
        else:
            remote = state.get("remote_id")
            deleted = bool(remote and client.delete_message(room_token, remote, expected_employee=employee_id))
            edited = False if deleted or not remote else client.edit_message(
                room_token, remote, "Обработка завершена.", expected_employee=employee_id,
            )
            result = {**state, "state": "CLEANED" if deleted or edited else "CLEANUP_FAILED"}
        with factory() as session, session.begin():
            row = session.get(TalkInbox, inbox_id)
            if (row.progress or {}).get("state") in {"SENDING", "CLEANING"}:
                row.progress = result
        return True
    except Exception as error:
        try:
            event("talk.progress_unavailable", error_code=type(error).__name__)
        except Exception:
            pass
        return False
