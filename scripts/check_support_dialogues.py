"""Exercise real Qwen dialogues, optionally through an actual private Talk room.

The controller mode keeps its conversation in a temporary SQLite database. Talk
mode sends employee messages to the configured bot and observes the running
worker. Its employee app password is read interactively or from process env.
Reports contain case IDs, source names, timings and counts.
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import SecretStr
from sqlalchemy import create_engine, select, update

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db import session_factory
from app.models import Base, Delivery, Incident, Message, MessageAttachment, TalkAccount, TalkInbox, TalkRoom, now
from app.services.chat.talk_dialogue import UNAVAILABLE, process_one
from app.services.llm.openwebui import OpenWebUILLM
from app.services.nextcloud.account import AccountClient
from app.services.nextcloud.inbound_media import parse_inbound_media
from app.services.nextcloud.interruption import is_resume_request, latest_paused
from app.services.rag.catalog import KnowledgeCatalog
from app.services.rag.talk import TalkKnowledge


CASES = [
    {"id": "first-mail-login", "turns": [
        {"message": "Другая задача. Мне выдали рабочую почту. Хочу открыть её в Outlook на этом компьютере, "
                    "раньше здесь не заходил.", "source": "31_", "confirm": True},
        {"message": "Да, правильно", "source": "31_", "photos": True},
    ]},
    {"id": "scanner-clarification-and-photo", "turns": [
        {"message": "Другая задача. Настройка сканера", "question": True},
        {"message": "Сканер штрихкодов. Нужно уменьшить громкость.", "source": "37_", "confirm": True},
        {"message": "Да, правильно", "source": "37_", "photos": True},
        {"message": "Не получилось. Сканер всё равно громко пищит.", "retain_goal": True},
    ]},
    {"id": "unknown-answer-keeps-the-task", "turns": [
        {"message": "Другая задача. Нужно настроить принтер.", "question": True},
        {"message": "Не знаю", "unknown": True},
        {"message": "Хочу печатать этикетки.", "retain_goal": True},
        {"message": "С переносного терминала на одном мобильном принтере TSC по ви фи.",
         "source": "24-1_", "confirm": True},
        {"message": "Да, правильно", "source": "24-1_", "photos": True},
    ]},
    {"id": "two-printers-correction", "turns": [
        {"message": "Другая задача. Нужно подключить ТСД к одному принтеру TSC по Wi-Fi.",
         "source": "24-1_", "confirm": True},
        {"message": "Да, правильно", "source": "24-1_", "photos": True},
        {"message": "Уточняю: у меня два принтера TSC, оба подключаются по ви фи. Блютуз не используется.",
         "source": "24-4_", "confirm": True},
        {"message": "Да, правильно", "source": "24-4_", "photos": True},
    ]},
    {"id": "topic-switch-during-output", "turns": [
        {"message": "Другая задача. Хочу впервые открыть рабочую почту в Outlook на компьютере.",
         "source": "31_", "confirm": True},
        {"message": "Да, правильно", "source": "31_", "wait_delivery": False, "interrupt": True},
        {"message": "Как уменьшить громкость сканера штрихкодов?",
         "source": "37_", "confirm": True, "pause_previous": True},
        {"message": "Да, правильно", "source": "37_", "photos": True, "keep_paused": True},
        {"message": "Продолжи предыдущий ответ", "source": "31_", "resume_previous": True},
    ]},
]


class EvaluationFailure(ValueError):
    pass


def check(condition, code):
    if not condition:
        raise EvaluationFailure(code)


def source_names(flow):
    confirmation = flow.get("understanding_confirmation") or {}
    active = confirmation.get("guide_flow") or flow
    plan = active.get("instruction_plan") or {}
    sources = [part.get("source", {}) for part in plan.get("parts", [])]
    sources += [active.get("source", {}), confirmation.get("source", {})]
    return list(dict.fromkeys(source["filename"] for source in sources if source.get("filename")))


def resumed_delivery_ids(factory, room_id, message):
    if not is_resume_request(message):
        return []
    with factory() as session:
        room = session.get(TalkRoom, room_id)
        paused = latest_paused(session, room) if room else None
        return paused.get("delivery_ids", []) if paused else []


def delivery_states(factory, room_id, delivery_ids):
    with factory() as session:
        room = session.get(TalkRoom, room_id)
        check(room is not None, "EvaluationRoomMissing")
        rows = session.execute(select(Delivery, Message.sequence, Message.content).join(Message).where(
            Delivery.id.in_(delivery_ids), Delivery.account_id == room.account_id,
            Delivery.room_token == room.room_token, Message.incident_id == room.incident_id,
        ).order_by(Message.sequence)).all()
        return [{"id": row.id, "message_id": row.message_id, "remote_id": row.remote_id,
                 "status": row.status, "tries": row.tries, "provenance": copy.deepcopy(row.provenance),
                 "sequence": sequence, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
                for row, sequence, content in rows]


def snapshot(factory, room_id, previous_ids, *, resumed_ids=()):
    with factory() as session:
        room = session.get(TalkRoom, room_id)
        check(room is not None, "EvaluationRoomMissing")
        incident = session.get(Incident, room.incident_id) if room.incident_id else None
        rows = session.scalars(select(Message).where(
            Message.incident_id == room.incident_id, Message.role == "assistant",
        ).order_by(Message.sequence)).all()
        resumed_messages = set(session.scalars(select(Delivery.message_id).where(
            Delivery.id.in_(resumed_ids), Delivery.account_id == room.account_id,
            Delivery.room_token == room.room_token,
        )))
        fresh = [row for row in rows if row.id not in previous_ids or row.id in resumed_messages]
        identifiers = [row.id for row in fresh]
        pictures = session.scalars(select(MessageAttachment).where(
            MessageAttachment.message_id.in_(identifiers), MessageAttachment.content_type == "image/png",
        )).all()
        deliveries = session.scalars(select(Delivery).where(Delivery.message_id.in_(identifiers))).all()
        return {"flow": copy.deepcopy(room.flow or {}), "state": incident.state if incident else None,
                "texts": [row.content for row in fresh], "message_ids": identifiers,
                "sources": source_names(room.flow or {}),
                "photos": [{"sha256": image.sha256, "message_id": image.message_id} for image in pictures],
                "deliveries": [{"id": row.id, "message_id": row.message_id, "remote_id": row.remote_id,
                                "status": row.status, "provenance": copy.deepcopy(row.provenance)}
                               for row in deliveries]}


def room_message_ids(factory, room_id):
    with factory() as session:
        room = session.get(TalkRoom, room_id)
        return set(session.scalars(select(Message.id).where(Message.incident_id == room.incident_id)))


class ControllerRunner:
    mode = "real_qwen_controller"

    def __init__(self, settings):
        self.directory = tempfile.TemporaryDirectory(prefix="support-dialogue-evaluation-")
        self.engine = create_engine("sqlite:///" + str(Path(self.directory.name) / "dialogues.db"))
        Base.metadata.create_all(self.engine)
        self.factory = session_factory(self.engine)
        self.settings = settings.model_copy(update={"nextcloud_backend": "mock", "talk_confirm_understanding": True})
        self.model = OpenWebUILLM(self.settings)
        self.sequence = 0
        self.complete_previous = True
        with self.factory() as session, session.begin():
            account = TalkAccount(server_url=settings.nextcloud_base_url.rstrip("/"), username="evaluation-bot",
                                  activated_at=now() - timedelta(minutes=1))
            session.add(account)
            session.flush()
            room = TalkRoom(account_id=account.id, room_token="evaluation-private", employee_id="evaluation-employee")
            session.add(room)
            session.flush()
            self.account_id, self.room_id = account.id, room.id

    def prepare(self):
        catalog = KnowledgeCatalog(self.settings, self.factory)
        knowledge = TalkKnowledge(self.settings, catalog=catalog)
        report = catalog.sync(knowledge, refresh=True)
        check(report["complete"], "EvaluationIndexIncomplete")
        return {key: report[key] for key in ("source_count", "indexed_count", "complete")}

    def turn(self, message, *, wait_delivery=True):
        previous = room_message_ids(self.factory, self.room_id)
        resumed_ids = resumed_delivery_ids(self.factory, self.room_id, message)
        self.sequence += 1
        with self.factory() as session, session.begin():
            if self.complete_previous:
                session.execute(update(Delivery).where(Delivery.status == "PENDING").values(status="SENT"))
            session.add(TalkInbox(room_id=self.room_id, remote_message_id=str(self.sequence),
                                  remote_sequence=self.sequence, sender_id="evaluation-employee", content=message,
                                  occurred_at=now()))
        check(process_one(self.factory, self.settings, self.model, account_id=self.account_id), "NoControllerTurn")
        self.complete_previous = wait_delivery
        if wait_delivery:
            with self.factory() as session, session.begin():
                session.execute(update(Delivery).where(Delivery.status == "PENDING").values(status="SENT"))
        return snapshot(self.factory, self.room_id, previous, resumed_ids=resumed_ids)

    def delivery_states(self, delivery_ids):
        return delivery_states(self.factory, self.room_id, delivery_ids)

    def close(self):
        self.engine.dispose()
        self.directory.cleanup()


class TalkRunner:
    mode = "real_qwen_and_talk"

    def __init__(self, settings, username, password, room_token=None, timeout=180):
        self.settings, self.timeout = settings, timeout
        self.factory = session_factory()
        self.bot = AccountClient(settings)
        self.bot_id = self.bot.identity()
        employee_settings = settings.model_copy(update={"nextcloud_username": username,
                                                       "nextcloud_app_password": SecretStr(password)})
        self.employee = AccountClient(employee_settings)
        self.employee_id = self.employee.identity()
        check(self.employee_id != self.bot_id, "EmployeeAccountRequired")
        tokens = [room_token] if room_token else [room.get("token") for room in self.bot.rooms()
                                                 if room.get("type") == 1]
        matches = []
        for token in tokens:
            try:
                self.bot.validate_room(token, expected_employee=self.employee_id)
                self.employee.validate_room(token, expected_employee=self.bot_id)
                matches.append(token)
            except Exception:
                continue
        check(len(matches) == 1, "UniquePrivateTestRoomRequired")
        self.token = matches[0]
        with self.factory() as session:
            room = session.scalar(select(TalkRoom).join(TalkAccount).where(
                TalkRoom.room_token == self.token, TalkRoom.employee_id == self.employee_id,
                TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
                TalkAccount.username == self.bot_id, TalkAccount.enabled.is_(True),
            ))
            check(room is not None, "RunningWorkerRoomRequired")
            self.room_id = room.id

    def turn(self, message, *, wait_delivery=True):
        previous = room_message_ids(self.factory, self.room_id)
        resumed_ids = resumed_delivery_ids(self.factory, self.room_id, message)
        remote_id = self.employee.send(self.token, message, "evaluation-" + uuid4().hex,
                                       expected_employee=self.bot_id)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            with self.factory() as session:
                inbox = session.scalar(select(TalkInbox).where(
                    TalkInbox.room_id == self.room_id, TalkInbox.remote_message_id == remote_id,
                ))
                done = inbox is not None and inbox.status == "DONE"
                check(inbox is None or inbox.status not in {"FAILED", "IGNORED"}, "TalkInputFailed")
            if done:
                result = snapshot(self.factory, self.room_id, previous, resumed_ids=resumed_ids)
                check(all(row["status"] not in {"FAILED", "UNKNOWN"} for row in result["deliveries"]),
                      "TalkDeliveryFailed")
                if not wait_delivery or all(row["status"] in {"SENT", "CANCELLED", "PAUSED"}
                                            for row in result["deliveries"]):
                    if wait_delivery:
                        check(all(row["status"] == "SENT" for row in result["deliveries"]), "ReplyNotDelivered")
                        self.verify_photos(result)
                    return result
            time.sleep(1)
        raise EvaluationFailure("TalkTurnTimedOut")

    def verify_photos(self, result):
        expected = {row["message_id"]: row["sha256"] for row in result["photos"]}
        photo_deliveries = [row["message_id"] for row in result["deliveries"] if row["message_id"] in expected]
        check(len(photo_deliveries) == len(set(photo_deliveries)), "DuplicatePhotoDelivery")
        check(set(photo_deliveries) == set(expected), "PhotoDeliveryMissing")
        for delivery in result["deliveries"]:
            digest = expected.get(delivery["message_id"])
            if digest is None:
                continue
            check(delivery["status"] == "SENT" and delivery["remote_id"], "PhotoNotDelivered")
            page = self.employee.messages(self.token, int(delivery["remote_id"]) - 1, limit=1)
            check(len(page["messages"]) == 1, "PhotoMessageMissing")
            message = page["messages"][0]
            check(str(message["id"]) == delivery["remote_id"] and message.get("actorId") == self.bot_id,
                  "PhotoMessageMismatch")
            media = parse_inbound_media(message)
            check(len(media.get("files", [])) == 1, "PhotoAttachmentMissing")
            photo = self.employee.download_inbound_image(self.token, delivery["remote_id"],
                                                       media["files"][0], expected_employee=self.bot_id)
            check(hashlib.sha256(photo.data).hexdigest() == digest, "PhotoPixelsChanged")

    def delivery_states(self, delivery_ids):
        return delivery_states(self.factory, self.room_id, delivery_ids)

    def close(self):
        pass


def validate_turn(result, expected):
    check(not any(UNAVAILABLE in text for text in result["texts"]), "ServiceErrorReply")
    if "message_ids" in result:
        delivered_ids = [row["message_id"] for row in result["deliveries"]]
        check(len(delivered_ids) == len(set(delivered_ids)), "DuplicateOutputDelivery")
        check(set(delivered_ids) == set(result["message_ids"]), "OutputDeliveryMissing")
    if expected.get("source"):
        check(any(name.startswith(expected["source"]) for name in result["sources"]), "WrongInstruction")
        technical = [row for row in result["deliveries"] if row.get("provenance")]
        check(all(row["provenance"].get("filename", "").startswith(expected["source"])
                  for row in technical), "WrongDeliveredInstruction")
        if expected.get("photos") or expected.get("resume_previous") or result["photos"]:
            check(bool(technical), "InstructionProvenanceMissing")
            by_message = {row["message_id"]: row for row in result["deliveries"]}
            check(all(by_message.get(photo["message_id"], {}).get("provenance")
                      for photo in result["photos"]), "PhotoProvenanceMissing")
    if expected.get("confirm"):
        check(bool(result["flow"].get("understanding_confirmation")), "UnderstandingConfirmationMissing")
    if expected.get("question"):
        check(bool(result["flow"].get("clarification_question")), "ClarificationMissing")
    if expected.get("photos"):
        check(bool(result["photos"]), "OriginalPhotographsMissing")
    if expected.get("retain_goal"):
        check(bool(result["flow"].get("task_state", {}).get("goals")), "TaskContextLost")
    if expected.get("unknown"):
        check(bool(result["flow"].get("task_state", {}).get("unknown_answers")), "UnknownAnswerLost")


def validate_interruption(runner, result, expected, interrupted):
    if expected.get("interrupt"):
        rows = runner.delivery_states([row["id"] for row in result["deliveries"]])
        pending = [row["id"] for row in rows if row["status"] == "PENDING"]
        check(bool(pending), "InterruptionNotObserved")
        plan = result["flow"].get("instruction_plan", {})
        check(type(plan.get("cursor")) is int and bool(plan.get("digest")), "InterruptionPlanMissing")
        return {"original": {row["id"]: row for row in rows}, "pending": pending,
                "paused": [], "plan": copy.deepcopy(plan)}
    if not any(expected.get(key) for key in ("pause_previous", "keep_paused", "resume_previous")):
        return interrupted
    check(bool(interrupted), "InterruptedResponseMissing")
    rows = runner.delivery_states(list(interrupted["original"]))
    current = {row["id"]: row for row in rows}
    check(set(current) == set(interrupted["original"]), "InterruptedDeliveryMissing")
    for delivery_id, original in interrupted["original"].items():
        row = current[delivery_id]
        check(row["message_id"] == original["message_id"] and row["provenance"] == original["provenance"]
              and row["sequence"] == original["sequence"]
              and row["content_sha256"] == original["content_sha256"], "InterruptedOutputChanged")
        completed = interrupted.get("sent_before_resume", {}).get(delivery_id)
        if completed or original["status"] == "SENT":
            completed = completed or original
            check(row["status"] == "SENT" and row["remote_id"] == completed["remote_id"]
                  and row["tries"] == completed["tries"], "SentOutputReplayed")
    if expected.get("pause_previous"):
        interrupted["paused"] = [delivery_id for delivery_id in interrupted["pending"]
                                 if current[delivery_id]["status"] == "PAUSED"]
        check(bool(interrupted["paused"]), "InterruptionNotObserved")
        check(all(row["status"] != "PENDING" for row in rows), "PreviousOutputStillQueued")
        interrupted["sent_before_resume"] = {row["id"]: row for row in rows if row["status"] == "SENT"}
        return interrupted
    paused = [current[delivery_id] for delivery_id in interrupted["paused"]]
    check(bool(paused), "InterruptedResponseMissing")
    if expected.get("keep_paused"):
        check(all(row["status"] == "PAUSED" and row["remote_id"] is None for row in paused), "PreviousOutputResumedEarly")
        return interrupted
    check(all(row["status"] == "SENT" for row in paused), "PreviousOutputNotResumed")
    check(set(interrupted["paused"]).issubset(row["id"] for row in result["deliveries"]), "ResumedOutputNotObserved")
    restored = result["flow"].get("instruction_plan", {})
    check(restored.get("digest") == interrupted["plan"]["digest"]
          and restored.get("cursor") == interrupted["plan"]["cursor"], "ResumePositionChanged")
    return interrupted


def run_cases(runner, cases, *, progress=None):
    rows = []
    for case in cases:
        interrupted = None
        for index, turn in enumerate(case["turns"], 1):
            started = time.monotonic()
            row = {"case": case["id"], "turn": index}
            result = None
            try:
                result = runner.turn(turn["message"], wait_delivery=turn.get("wait_delivery", True))
                validate_turn(result, turn)
                interrupted = validate_interruption(runner, result, turn, interrupted)
                photo_hashes = [photo["sha256"] for photo in result["photos"]]
                continuation_count = 0
                while (turn.get("photos") or turn.get("resume_previous")) and result["flow"].get("has_more"):
                    check(continuation_count < 100, "ContinuationLimitExceeded")
                    previous_cursor = result["flow"].get("instruction_plan", {}).get("cursor")
                    result = runner.turn("Продолжить")
                    validate_turn(result, {"source": turn.get("source")})
                    current_cursor = result["flow"].get("instruction_plan", {}).get("cursor")
                    check(type(previous_cursor) is int and type(current_cursor) is int
                          and current_cursor > previous_cursor, "InstructionProgressStalled")
                    photo_hashes.extend(photo["sha256"] for photo in result["photos"])
                    continuation_count += 1
                row.update(status="ok", sources=result["sources"], photos=len(photo_hashes),
                           continuations=continuation_count)
                if turn.get("pause_previous") or turn.get("resume_previous"):
                    row["paused_deliveries"] = len(interrupted["paused"])
            except Exception as error:
                code = str(error) if isinstance(error, EvaluationFailure) else type(error).__name__
                row.update(status="error", error_code=code if re.fullmatch(r"[A-Za-z0-9_]{1,100}", code)
                           else "EvaluationError")
                if result is not None:
                    state = result["flow"].get("task_state", {})
                    row["observed"] = {
                        "sources": result["sources"], "photos": len(result["photos"]),
                        "goal_count": len(state.get("goals", {})),
                        "fact_keys": sorted({fact["key"] for fact in state.get("facts", [])}),
                        "has_clarification": bool(result["flow"].get("clarification_question")),
                        "has_confirmation": bool(result["flow"].get("understanding_confirmation")),
                        "delivered_sources": sorted({row["provenance"]["filename"]
                            for row in result["deliveries"] if (row.get("provenance") or {}).get("filename")}),
                    }
            row["seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            if progress:
                progress(row)
            if row["status"] == "error":
                break
    durations = sorted(row["seconds"] for row in rows)
    return {"mode": runner.mode, "status": "ok" if all(row["status"] == "ok" for row in rows) else "error",
            "turns": rows, "p50_seconds": durations[len(durations) // 2] if durations else None,
            "p95_seconds": durations[min(len(durations) - 1, int(len(durations) * .95))] if durations else None}


def check_catalogue(settings, *, progress=None):
    knowledge = TalkKnowledge(settings)
    with knowledge._client() as client:
        files = knowledge._files(client)
    rows = []
    for file_id, item in files.items():
        started = time.monotonic()
        row = {"file_id": file_id, "source": item["filename"]}
        try:
            approved = knowledge.read(file_id)
            title = item["filename"].rsplit(".", 1)[0].replace("_", " ")
            result = knowledge.search_with_llm("Нужна инструкция «" + title + "».", context={
                "requested_goals": {"g1": "Выполнить процедуру «" + title + "»"},
                "instruction_planning": True, "source_clarification_required": True,
            })
            selected = [part.approved for part in getattr(result, "plan", ())]
            if result.approved is not None:
                selected.append(result.approved)
            check(any(source.document["source"]["file_id"] == file_id for source in selected), "CatalogueSourceNotSelected")
            row.update(status="ok", images=len(approved.illustrations), sha256=approved.document["source"]["sha256"])
        except Exception as error:
            row.update(status="error", error_code=str(error) if isinstance(error, EvaluationFailure)
                       else type(error).__name__)
        row["seconds"] = round(time.monotonic() - started, 3)
        rows.append(row)
        if progress:
            progress(row)
    return {"status": "ok" if all(row["status"] == "ok" for row in rows) else "error", "files": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talk", action="store_true", help="Send test questions in an actual private employee/bot room")
    parser.add_argument("--employee-user")
    parser.add_argument("--room")
    parser.add_argument("--case", action="append", choices=[case["id"] for case in CASES])
    parser.add_argument("--all-instructions", action="store_true", help="Check original parsing and model selection for every current source")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    configure_logging()
    runner = None
    try:
        settings = get_settings()
        check(settings.llm_backend == "openwebui", "RealQwenRequired")
        if args.talk:
            username = args.employee_user or input("Логин тестового сотрудника в Nextcloud: ").strip()
            check(bool(username), "EmployeeUsernameRequired")
            password = os.environ.get("TALK_TEST_APP_PASSWORD") or getpass.getpass("Пароль приложения тестового сотрудника: ")
            runner = TalkRunner(settings, username, password, args.room)
        else:
            runner = ControllerRunner(settings)
            print(json.dumps({"stage": "prepare_index", "status": "started"}), flush=True)
            index = runner.prepare()
            print(json.dumps({"stage": "prepare_index", "status": "ok", **index}), flush=True)
        cases = [case for case in CASES if not args.case or case["id"] in args.case]
        report = run_cases(runner, cases, progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
        from app.main import app

        report.update(app_version=app.version, model=settings.openwebui_model,
                      verification_thinking=settings.openwebui_verification_think)
        if args.all_instructions:
            report["catalogue"] = check_catalogue(
                settings, progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True),
            )
            if report["catalogue"]["status"] != "ok":
                report["status"] = "error"
        if args.report:
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report["status"] == "ok" else 1
    except Exception as error:
        code = str(error) if isinstance(error, EvaluationFailure) else type(error).__name__
        print(json.dumps({"status": "error", "error_code": code}), flush=True)
        return 1
    finally:
        if runner is not None:
            runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
