"""Personal Talk account runner: polling, dialogue and ordered delivery stay independent."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select, text, update

from app.core.config import get_settings
from app.core.logging import configure_logging, event
from app.db import get_engine, session_factory
from app.models import Delivery, Message, TalkAccount, TalkInbox, TalkRoom
from app.services.chat.talk_dialogue import process_one
from app.services.llm.mock import MockLLM
from app.services.llm.openwebui import OpenWebUILLM
from app.services.nextcloud.account import AccountClient, AccountError
from app.services.nextcloud.delivery import deliver_one
from app.services.nextcloud.poller import poll_once

LOCK_ID = 73412063  # Separate from the main worker and read-only PostgreSQL lock test.


def error_code(error):
    return error.code if isinstance(error, AccountError) else type(error).__name__


def check_connection(settings, client=None):
    """Read-only account/private-room check. No messages, read markers or files are written."""
    if settings.nextcloud_backend != "account":
        raise ValueError("TalkAccountModeRequired")
    client = client or AccountClient(settings)
    client.identity()
    valid = 0
    for room in client.rooms():
        if not isinstance(room, dict) or room.get("type") != 1:
            continue
        try:
            client.validate_room(room.get("token"))
            valid += 1
        except Exception:
            continue
    return {"status": "ok", "account": "authenticated", "private_only": True,
            "private_conversations": valid, "messages_sent": 0, "files_uploaded": 0}


def current_account(factory, settings, service_id):
    with factory() as session:
        return session.scalar(select(TalkAccount.id).where(
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
            TalkAccount.username == service_id, TalkAccount.enabled.is_(True),
        ))


def recover(factory, settings, service_id):
    """Replay read-only generation, but never replay an uncertain external POST."""
    with factory() as session, session.begin():
        room_ids = select(TalkRoom.id).join(TalkAccount).where(
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
            TalkAccount.username == service_id,
        )
        session.execute(update(TalkInbox).where(
            TalkInbox.room_id.in_(room_ids), TalkInbox.status == "RUNNING", TalkInbox.tries < 3,
        ).values(status="PENDING"))
        session.execute(update(TalkInbox).where(
            TalkInbox.room_id.in_(room_ids), TalkInbox.status == "RUNNING", TalkInbox.tries >= 3,
        ).values(status="FAILED", error_code="InterruptedGeneration"))
        accounts = select(TalkAccount.id).where(
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"), TalkAccount.username == service_id,
        )
        session.execute(update(Delivery).where(
            Delivery.account_id.in_(accounts), Delivery.status == "SENDING",
        ).values(status="UNKNOWN"))
        uncertain = select(Delivery.message_id).where(Delivery.account_id.in_(accounts), Delivery.status == "UNKNOWN")
        session.execute(update(Message).where(Message.id.in_(uncertain)).values(delivery_status="UNKNOWN"))


def run(settings, *, once=False):
    if settings.nextcloud_backend != "account":
        raise ValueError("TalkAccountModeRequired")
    engine = get_engine()
    factory = session_factory(engine)
    llm = MockLLM() if settings.llm_backend == "mock" else OpenWebUILLM(settings)
    delivery_client = AccountClient(settings)
    service_id = delivery_client.identity()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        if not guard.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": LOCK_ID}):
            raise RuntimeError("TalkAccountWorkerAlreadyRunning")
        recover(factory, settings, service_id)
        event("talk.started", mode="account", private_only=True, llm=llm.name)
        if once:
            poll_once(factory, settings)
            account_id = current_account(factory, settings, service_id)
            if account_id:
                process_one(factory, settings, llm, account_id=account_id)
            for _ in range(16):
                if not deliver_one(factory, settings, delivery_client):
                    break
            return
        # Slow model calls cannot pause room discovery or the preceding text/photo deliveries.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="talk") as pool:
            polling = processing = None
            next_poll = 0.0
            while True:
                guard.execute(text("SELECT 1"))  # Exit if the singleton lock's session is lost.
                if polling is not None and polling.done():
                    try:
                        polling.result()
                    except Exception as exc:
                        event("talk.poll_error", error_code=error_code(exc))
                    polling = None
                    next_poll = time.monotonic() + settings.nextcloud_poll_interval_seconds
                if processing is not None and processing.done():
                    try:
                        processing.result()
                    except Exception as exc:
                        event("talk.process_error", error_code=error_code(exc))
                    processing = None
                if polling is None and time.monotonic() >= next_poll:
                    polling = pool.submit(poll_once, factory, settings)
                if processing is None:
                    account_id = current_account(factory, settings, service_id)
                    if account_id:
                        processing = pool.submit(process_one, factory, settings, llm, account_id=account_id)
                deliver_one(factory, settings, delivery_client)
                time.sleep(0.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Personal Nextcloud Talk account connector")
    parser.add_argument("--check", action="store_true", help="Check account and private rooms without sending")
    parser.add_argument("--check-knowledge", action="store_true", help="Read approved knowledge and photos without sending")
    parser.add_argument("--once", action="store_true", help="Run one real poll/process/delivery cycle")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        settings = get_settings()
        if args.check_knowledge:
            from app.services.rag.talk import TalkKnowledge

            result = TalkKnowledge(settings).check()
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "ok" else 1
        elif args.check:
            print(json.dumps(check_connection(settings), ensure_ascii=False))
        else:
            run(settings, once=args.once)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Never serialize exception bodies: HTTP/validation errors can contain credentials or personal data.
        print(json.dumps({"status": "error", "error_code": error_code(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
