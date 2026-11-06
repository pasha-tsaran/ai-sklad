"""Durably queue new employee text from validated local personal conversations.

The activation boundary is persisted before discovery. New rooms read forward
from cursor zero, including pages of old history, instead of jumping to the last
message (which would discard the first question in a newly discovered DM).
History before the activation second is intentionally ignored, read markers are
not changed, and cursor advancement commits atomically with durable inbox rows.
"""

from datetime import datetime, timezone

from sqlalchemy import select

from app.models.talk import TalkAccount, TalkInbox, TalkRoom
from app.services.nextcloud.account import AccountClient, AccountError, sequence

MAX_INBOUND_CHARS = 8000


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _text(message, employee, service, activated):
    sender = message.get("actorId")
    if (message.get("actorType") != "users" or sender != employee or sender == service
            or message.get("messageType") != "comment" or message.get("systemMessage")
            or message["timestamp"] < int(_utc(activated).timestamp())):
        return None
    parameters = message.get("messageParameters") or {}
    values = parameters.values() if isinstance(parameters, dict) else parameters if isinstance(parameters, list) else []
    if any(isinstance(value, dict) and value.get("type") in {"file", "media", "attachment"} for value in values):
        return None  # No downloading employee files, avatars or remote rich-object URLs.
    content = message.get("message")
    return content if isinstance(content, str) and content.strip() and len(content) <= MAX_INBOUND_CHARS else None


def _room_error(factory, room_id, exc):
    with factory() as session, session.begin():
        room = session.get(TalkRoom, room_id)
        if room:
            room.enabled = False
            room.status = "BLOCKED" if isinstance(exc, AccountError) and exc.code in {
                "room_not_private", "room_employee_changed", "room_not_allowed", "access_denied"
            } else "UNAVAILABLE"
            room.error_code = exc.code if isinstance(exc, AccountError) else type(exc).__name__[:100]


def poll_once(factory, settings, client=None):
    """Discover private DMs and persist at most one bounded page per room, no replies."""
    if settings.nextcloud_backend != "account":
        return False
    client = client or AccountClient(settings)
    service = client.identity()
    base = settings.nextcloud_base_url.rstrip("/")
    with factory() as session, session.begin():
        stale_accounts = session.scalars(select(TalkAccount).where(
            TalkAccount.server_url == base, TalkAccount.username != service, TalkAccount.enabled.is_(True),
        )).all()
        for stale in stale_accounts:
            stale.enabled = False
            for old_room in session.scalars(select(TalkRoom).where(TalkRoom.account_id == stale.id)):
                old_room.enabled, old_room.status, old_room.error_code = False, "ACCOUNT_CHANGED", "account_uid_changed"
        account = session.scalar(select(TalkAccount).where(
            TalkAccount.server_url == base, TalkAccount.username == service,
        ))
        if account is None:
            stamp = getattr(client, "server_time", None) or datetime.now(timezone.utc)
            account = TalkAccount(server_url=base, username=service,
                                  activated_at=_utc(stamp).replace(microsecond=0))
            session.add(account)
            session.flush()
        account_id, activated, enabled = account.id, account.activated_at, account.enabled
    if not enabled:
        return False
    # A full list is required: modifiedSince cannot report removals/deletions.
    discovered = client.rooms()
    if not isinstance(discovered, list) or len(discovered) > 10_000:
        raise AccountError("invalid_room_list")
    present = {room.get("token") for room in discovered if isinstance(room, dict)
               and isinstance(room.get("token"), str)}
    with factory() as session, session.begin():
        known = {room.room_token: (room.id, room.employee_id, room.cursor, room.status)
                 for room in session.scalars(select(TalkRoom).where(TalkRoom.account_id == account_id))}
        for token, (room_id, _, _, status) in known.items():
            if token not in present and status != "DISABLED":
                room = session.get(TalkRoom, room_id)
                room.enabled, room.status, room.error_code = False, "NOT_PRESENT", "room_missing"
    accepted = 0
    seen = set()
    for candidate in discovered:
        if not isinstance(candidate, dict):
            continue
        token = candidate.get("token")
        if not isinstance(token, str) or token in seen:
            continue
        seen.add(token)
        existing = known.get(token)
        if existing and existing[3] == "DISABLED":
            continue
        if type(candidate.get("type")) is not int or candidate["type"] != 1:
            if existing:
                _room_error(factory, existing[0], AccountError("room_not_private"))
            continue
        # The independently fetched room/roster, never this discovery summary, is authoritative.
        try:
            validated = client.validate_room(token, expected_employee=existing[1] if existing else None)
            if (validated.get("type") != 1 or validated.get("service_id") != service
                    or validated.get("token") != token):
                raise AccountError("room_not_private")
        except Exception as exc:
            if existing:
                _room_error(factory, existing[0], exc)
            continue
        if existing is None:
            with factory() as session, session.begin():
                room = session.scalar(select(TalkRoom).where(
                    TalkRoom.account_id == account_id, TalkRoom.room_token == token,
                ))
                if room is None:
                    room = TalkRoom(account_id=account_id, room_token=token,
                                    employee_id=validated["employee_id"])
                    session.add(room)
                    session.flush()
                existing = (room.id, room.employee_id, room.cursor, room.status)
        room_id, employee, cursor, _ = existing
        try:
            page = client.messages(token, cursor, limit=settings.nextcloud_batch_size)
            messages, next_cursor = page["messages"], sequence(page["cursor"], allow_zero=True)
            if not isinstance(messages, list) or len(messages) > settings.nextcloud_batch_size or next_cursor < cursor:
                raise AccountError("invalid_message_page")
            previous = cursor
            for message in messages:
                if not isinstance(message, dict) or message.get("token") != token:
                    raise AccountError("invalid_message_page")
                value = sequence(message.get("id"))
                if value <= previous or value > next_cursor:
                    raise AccountError("unordered_message_page")
                stamp = message.get("timestamp")
                if type(stamp) is not int or not 0 <= stamp <= 253_402_300_799:
                    raise AccountError("invalid_message_timestamp")
                previous = value
            page_accepted = 0
            with factory() as session, session.begin():
                room = session.scalar(select(TalkRoom).where(TalkRoom.id == room_id).with_for_update())
                if room.cursor != cursor or room.employee_id != employee:
                    continue  # Another poll committed first; never advance over unseen messages.
                for message in messages:
                    remote_sequence = sequence(message["id"])
                    content = _text(message, employee, service, activated)
                    # Ignore old/unsupported content without retaining old private transcripts.
                    sender = message.get("actorId")
                    sender = sender if isinstance(sender, str) and len(sender) <= 255 else ""
                    if session.scalar(select(TalkInbox.id).where(
                        TalkInbox.room_id == room_id, TalkInbox.remote_message_id == str(remote_sequence),
                    )) is not None:
                        continue
                    session.add(TalkInbox(
                        room_id=room_id, remote_message_id=str(remote_sequence),
                        remote_sequence=remote_sequence, sender_id=sender, content=content or "",
                        occurred_at=datetime.fromtimestamp(message["timestamp"], timezone.utc),
                        status="PENDING" if content is not None else "IGNORED",
                    ))
                    page_accepted += content is not None
                room.cursor, room.enabled, room.status, room.error_code = next_cursor, True, "ACTIVE", None
            accepted += page_accepted
        except Exception as exc:
            _room_error(factory, room_id, exc)
    return accepted > 0
