"""Personal Talk account runner: polling, dialogue and ordered delivery stay independent."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

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
from app.services.rag.catalog_worker import maintain_knowledge
from app.services.nextcloud.progress import recover_progress, tick_progress
from app.services.nextcloud.typing import NativeTyping

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
        recover_progress(factory, settings, service_id)
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
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="talk") as pool, closing(NativeTyping(settings)) as progress_client:
            polling = processing = progressing = indexing = None
            next_poll = 0.0
            next_index = 0.0
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
                if progressing is not None and progressing.done():
                    try:
                        progressing.result()
                    except Exception:
                        pass  # Optional status updates never stop support delivery.
                    progressing = None
                if indexing is not None and indexing.done():
                    try:
                        result = indexing.result()
                        event("talk.knowledge_index", complete=result.get("complete") is True,
                              ready=result.get("indexed_count", result.get("ready")), status=result.get("status"),
                              expired_retrievals=result.get("expired_retrievals"))
                    except Exception as exc:
                        event("talk.knowledge_index_error", error_code=error_code(exc))
                    indexing = None
                    next_index = time.monotonic() + 300
                if polling is None and time.monotonic() >= next_poll:
                    polling = pool.submit(poll_once, factory, settings)
                if processing is None:
                    account_id = current_account(factory, settings, service_id)
                    if account_id:
                        processing = pool.submit(process_one, factory, settings, llm, account_id=account_id)
                if progressing is None:
                    progressing = pool.submit(tick_progress, factory, settings, progress_client)
                if indexing is None and time.monotonic() >= next_index:
                    indexing = pool.submit(maintain_knowledge, factory, settings)
                deliver_one(factory, settings, delivery_client)
                time.sleep(0.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Personal Nextcloud Talk account connector")
    parser.add_argument("--check", action="store_true", help="Check account and private rooms without sending")
    parser.add_argument("--check-knowledge", action="store_true", help="Read approved knowledge and photos without sending")
    parser.add_argument("--check-profile", action="store_true", help="Validate shared settings without sending or printing the prompt")
    parser.add_argument("--check-model", action="store_true", help="Run a small read-only model completion")
    parser.add_argument("--check-question", metavar="QUESTION", help="Try a fresh agent question without a Talk chat or delivery")
    parser.add_argument("--check-typing", action="store_true", help="Read native typing prerequisites without joining or sending")
    parser.add_argument("--once", action="store_true", help="Run one real poll/process/delivery cycle")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        settings = get_settings()
        if args.check_question is not None:
            from app.services.chat.agent_dialogue import run_agent
            from app.services.chat.turn_budget import turn_budget
            from app.services.llm.profile import profile_settings
            from app.services.rag.catalog import KnowledgeCatalog
            from app.services.rag.talk import TalkKnowledge

            if not args.check_question.strip() or len(args.check_question) > 6000:
                parser.error("--check-question requires 1 to 6000 characters")
            snapshot = profile_settings(settings)
            knowledge = TalkKnowledge(snapshot, catalog=KnowledgeCatalog(snapshot, session_factory()))
            started = time.monotonic()
            with turn_budget(snapshot.talk_turn_budget_seconds):
                turn = run_agent(args.check_question, {}, {}, OpenWebUILLM(snapshot), knowledge, snapshot)
            sources = [doc.document["source"]["filename"] for doc in turn.agent_documents]
            outcome = "reply_with_source" if sources else ("dialogue" if turn.action == "AGENT_REPLY" else "no_instruction")
            print(json.dumps({"outcome": outcome, "sources": sources, "images": sum(img is not None for _, img in turn.blocks),
                              "messages_sent": 0, "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False))
            return 0 if outcome != "no_instruction" else 1
        if args.check_model:
            from app.services.llm.profile import profile_settings

            snapshot = profile_settings(settings)
            started = time.monotonic()
            probe = {"model": snapshot.openwebui_model, "num_ctx": snapshot.openwebui_num_ctx, "stage": "task_relation"}
            try:
                relation = OpenWebUILLM(snapshot).classify_task_relation(
                    "Где контакты техподдержки?", {"goal": "Настроить принтер"}, [])
                probe.update(status="ok", relation=relation)
            except Exception as exc:
                probe.update(status="error", error_code=error_code(exc))
            print(json.dumps({**probe, "seconds": round(time.monotonic() - started, 2)}))
            return 0 if probe["status"] == "ok" else 1
        if args.check_profile:
            from app.services.llm.profile import profile_settings

            snapshot = profile_settings(settings)
            result = {"status": "ok" if settings.openwebui_sync_profile else "disabled",
                      "profile": settings.openwebui_profile_id,
                      "model": snapshot.openwebui_model,
                      "num_ctx": snapshot.openwebui_num_ctx,
                      "knowledge_ids": snapshot.openwebui_knowledge_ids,
                      "revision": snapshot.openwebui_profile_revision}
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result["status"] == "ok" else 1
        elif args.check_typing:
            with closing(NativeTyping(settings)) as typing:
                result = typing.check()
            print(json.dumps(result, ensure_ascii=False))
            return 0 if result.get("status") == "ok" else 1
        elif args.check_knowledge:
            from app.services.rag.talk import TalkKnowledge
            from app.services.llm.profile import profile_settings

            result = TalkKnowledge(profile_settings(settings)).check()
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
