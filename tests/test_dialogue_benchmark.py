"""The live benchmark reports actual stages and never prints source or secret text."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.models import Delivery, KnowledgeIndex, Message, MessageAttachment, TalkRoom
from app.services.llm.mock import MockLLM
from scripts.check_support_dialogues import (
    ControllerRunner, EvaluationFailure, TalkRunner, room_message_ids, run_cases, snapshot, source_names,
    validate_interruption, validate_turn,
)
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider


def result(**values):
    return {"flow": {}, "state": "DIAGNOSING", "texts": [], "sources": [], "photos": [], "deliveries": [], **values}


def test_source_names_follow_the_plan_waiting_for_confirmation():
    parts = [{"source": {"filename": "Первая.docx"}}, {"source": {"filename": "Вторая.docx"}}]
    flow = {"understanding_confirmation": {"guide_flow": {"instruction_plan": {"parts": parts}},
                                          "source": {"filename": "Первая.docx"}}}
    assert source_names(flow) == ["Первая.docx", "Вторая.docx"]
    assert source_names({"source": {"filename": "Прежняя.docx"}}) == ["Прежняя.docx"]


@pytest.mark.parametrize("expected,code", [
    ({"source": "31_"}, "WrongInstruction"),
    ({"photos": True}, "OriginalPhotographsMissing"),
    ({"confirm": True}, "UnderstandingConfirmationMissing"),
    ({"question": True}, "ClarificationMissing"),
    ({"retain_goal": True}, "TaskContextLost"),
])
def test_benchmark_rejects_missing_required_outcomes(expected, code):
    with pytest.raises(EvaluationFailure, match=code):
        validate_turn(result(), expected)


def test_benchmark_stops_the_failed_case_and_keeps_report_free_of_private_text():
    class Runner:
        mode = "real_qwen_controller"

        def __init__(self):
            self.calls = []

        def turn(self, text, **options):
            self.calls.append(text)
            if text == "first":
                raise RuntimeError("private employee text and API password")
            return result(texts=["private source content"], sources=["Проверенная.docx"])

    runner = Runner()
    report = run_cases(runner, [{"id": "failure", "turns": [{"message": "first"}, {"message": "skip"}]},
                                {"id": "next-case", "turns": [{"message": "second"}]}])
    assert runner.calls == ["first", "second"]
    assert report["status"] == "error" and report["turns"][0]["error_code"] == "RuntimeError"
    assert report["turns"][1]["status"] == "ok"
    assert "private" not in json.dumps(report)


def test_failed_dialogue_reports_safe_task_and_source_metadata():
    state = result(sources=["01_Previous.docx"], texts=["private reply"], flow={
        "understanding_confirmation": {"summary": "private summary"},
        "task_state": {"goals": {"g1": {"text": "private task"}},
                       "facts": [{"key": "application", "value": "private value"}]},
    }, deliveries=[{"provenance": None}, {"provenance": {"filename": "01_Previous.docx"}}])
    runner = SimpleNamespace(mode="real_qwen_controller", turn=lambda *args, **kwargs: state)
    report = run_cases(runner, [{"id": "missing-question", "turns": [{"message": "question", "question": True}]}])
    assert report["turns"][0]["observed"] == {
        "sources": ["01_Previous.docx"], "photos": 0, "goal_count": 1,
        "fact_keys": ["application"], "has_clarification": False, "has_confirmation": True,
        "delivered_sources": ["01_Previous.docx"],
    }
    assert "private" not in json.dumps(report)


def test_benchmark_completes_every_selected_batch_before_the_next_scenario():
    class Runner:
        mode = "real_qwen_controller"

        def __init__(self):
            self.calls = []

        def turn(self, text, **options):
            self.calls.append(text)
            more = len(self.calls) == 1
            return result(flow={"has_more": more, "instruction_plan": {"cursor": len(self.calls)}},
                          sources=["31_Example.docx"],
                          photos=[{"message_id": str(len(self.calls)), "sha256": str(len(self.calls))}],
                          deliveries=[{"message_id": str(len(self.calls)), "provenance": {"filename": "31_Example.docx"}}])

    runner = Runner()
    report = run_cases(runner, [{"id": "full-guide", "turns": [{"message": "question", "photos": True,
                                                                  "source": "31_"}]}])
    assert runner.calls == ["question", "Продолжить"]
    assert report["status"] == "ok" and report["turns"][0]["photos"] == 2
    assert report["turns"][0]["continuations"] == 1


def test_benchmark_rejects_restarted_or_stalled_step_progress():
    runner = SimpleNamespace(mode="real_qwen_controller", turn=lambda *args, **kwargs: result(
        flow={"has_more": True, "instruction_plan": {"cursor": 3}}, photos=[{"sha256": "image"}],
    ))
    report = run_cases(runner, [{"id": "stalled-guide", "turns": [{"message": "question", "photos": True}]}])
    assert report["status"] == "error" and report["turns"][0]["error_code"] == "InstructionProgressStalled"


def test_controller_benchmark_uses_an_isolated_database_and_the_production_entry_point(monkeypatch):
    monkeypatch.setattr("scripts.check_support_dialogues.OpenWebUILLM", lambda settings: MockLLM())
    settings = Settings(_env_file=None, database_url="postgresql://production-is-never-opened",
                        nextcloud_base_url="https://talk.example.invalid")
    runner = ControllerRunner(settings)
    directory = runner.directory.name
    try:
        state = runner.turn("Привет")
        assert state["texts"] and runner.sequence == 1
        assert str(runner.engine.url).startswith("sqlite:///")
        assert directory in str(runner.engine.url)
    finally:
        runner.close()


def test_controller_prepares_the_whole_isolated_index_before_dialogues(monkeypatch):
    rows = [(f"00000000-0000-4000-8000-{number:012}", f"Материал {number}.docx",
             instruction([f"Выполните действие для нового приложения номер {number}."]))
            for number in range(1, 5)]
    knowledge, _ = scanner_provider(rows)

    def configured(settings, *, catalog):
        knowledge.set_catalog(catalog)
        return knowledge

    monkeypatch.setattr("scripts.check_support_dialogues.OpenWebUILLM", lambda settings: MockLLM())
    monkeypatch.setattr("scripts.check_support_dialogues.TalkKnowledge", configured)
    runner = ControllerRunner(knowledge.settings)
    try:
        assert runner.prepare() == {"source_count": 4, "indexed_count": 4, "complete": True}
        with runner.factory() as session:
            assert len(session.scalars(select(KnowledgeIndex)).all()) == 4
    finally:
        runner.close()


def test_talk_photo_check_reads_actual_remote_attachment_and_compares_original_bytes():
    data = b"exact original image bytes"
    calls = []
    message = {"id": 42, "actorId": "bot", "actorType": "users", "messageType": "comment", "message": "{file}",
               "messageParameters": {"file": {"type": "file", "id": "7", "mimetype": "image/png", "size": len(data)}}}

    def download(room, remote_id, file_id, **options):
        calls.append((room, remote_id, file_id, options))
        return SimpleNamespace(data=data)

    runner = object.__new__(TalkRunner)
    runner.token, runner.bot_id = "private", "bot"
    runner.employee = SimpleNamespace(messages=lambda *args, **kwargs: {"messages": [message]},
                                      download_inbound_image=download)
    state = result(photos=[{"message_id": "local", "sha256": hashlib.sha256(data).hexdigest()}],
                   deliveries=[{"message_id": "local", "remote_id": "42", "status": "SENT"}])
    runner.verify_photos(state)
    assert calls == [("private", "42", 7, {"expected_employee": "bot"})]
    state["photos"][0]["sha256"] = "0" * 64
    with pytest.raises(EvaluationFailure, match="PhotoPixelsChanged"):
        runner.verify_photos(state)


def test_talk_photo_check_rejects_an_unsent_attachment():
    runner = object.__new__(TalkRunner)
    with pytest.raises(EvaluationFailure, match="PhotoNotDelivered"):
        runner.verify_photos(result(photos=[{"message_id": "image", "sha256": "a" * 64}],
                                    deliveries=[{"message_id": "image", "remote_id": None, "status": "PAUSED"}]))


def test_talk_photo_check_requires_a_delivery_for_every_original_attachment():
    runner = object.__new__(TalkRunner)
    with pytest.raises(EvaluationFailure, match="PhotoDeliveryMissing"):
        runner.verify_photos(result(photos=[{"message_id": "image", "sha256": "a" * 64}]))


def test_talk_photo_check_rejects_duplicate_attachment_deliveries():
    runner = object.__new__(TalkRunner)
    delivery = {"message_id": "image", "remote_id": "42", "status": "SENT"}
    with pytest.raises(EvaluationFailure, match="DuplicatePhotoDelivery"):
        runner.verify_photos(result(photos=[{"message_id": "image", "sha256": "a" * 64}],
                                    deliveries=[delivery, delivery]))


def test_current_plan_source_cannot_hide_blocks_from_a_different_instruction():
    state = result(sources=["31_Correct.docx"], photos=[{"sha256": "original"}],
                   deliveries=[{"provenance": {"filename": "37_Previous.docx"}}])
    with pytest.raises(EvaluationFailure, match="WrongDeliveredInstruction"):
        validate_turn(state, {"source": "31_", "photos": True})


def test_photo_reply_requires_actual_instruction_block_provenance():
    state = result(sources=["31_Correct.docx"], photos=[{"sha256": "original"}],
                   deliveries=[{"provenance": None}])
    with pytest.raises(EvaluationFailure, match="InstructionProvenanceMissing"):
        validate_turn(state, {"source": "31_", "photos": True})


def test_text_provenance_does_not_replace_missing_photo_provenance():
    state = result(sources=["31_Correct.docx"], photos=[{"message_id": "photo", "sha256": "original"}],
                   deliveries=[{"message_id": "text", "provenance": {"filename": "31_Correct.docx"}},
                               {"message_id": "photo", "provenance": None}])
    with pytest.raises(EvaluationFailure, match="PhotoProvenanceMissing"):
        validate_turn(state, {"source": "31_", "photos": True})


def test_continuation_photos_keep_their_instruction_provenance():
    state = result(sources=["31_Correct.docx"], photos=[{"message_id": "photo", "sha256": "original"}],
                   deliveries=[{"message_id": "photo", "provenance": None}])
    with pytest.raises(EvaluationFailure, match="InstructionProvenanceMissing"):
        validate_turn(state, {"source": "31_"})


def test_every_output_message_requires_exactly_one_delivery():
    with pytest.raises(EvaluationFailure, match="OutputDeliveryMissing"):
        validate_turn(result(message_ids=["text", "photo"], deliveries=[{"message_id": "photo"}]), {})
    with pytest.raises(EvaluationFailure, match="DuplicateOutputDelivery"):
        validate_turn(result(message_ids=["text"], deliveries=[{"message_id": "text"}] * 2), {})


def test_snapshot_includes_only_the_tracked_resumed_original_rows(monkeypatch):
    monkeypatch.setattr("scripts.check_support_dialogues.OpenWebUILLM", lambda settings: MockLLM())
    runner = ControllerRunner(Settings(_env_file=None, nextcloud_base_url="https://talk.example.invalid"))
    try:
        runner.turn("Привет")
        with runner.factory() as session, session.begin():
            room = session.get(TalkRoom, runner.room_id)
            message = session.scalar(select(Message).where(
                Message.incident_id == room.incident_id,
            ).order_by(Message.sequence.desc()).limit(1))
            resumed = Message(incident_id=room.incident_id, sequence=message.sequence + 1,
                              role="assistant", content="original photo", delivery_status="SENT")
            session.add(resumed)
            session.flush()
            session.add(MessageAttachment(message_id=resumed.id, filename="source.png", caption="Original",
                                          content_type="image/png", sha256="a" * 64, data=b"original"))
            delivery = Delivery(message_id=resumed.id, room_token=room.room_token, account_id=room.account_id,
                                status="SENT", remote_id="42", provenance={"filename": "31_Original.docx"})
            session.add(delivery)
            session.flush()
            delivery_id, message_id = delivery.id, resumed.id
        previous = room_message_ids(runner.factory, runner.room_id)
        assert snapshot(runner.factory, runner.room_id, previous)["photos"] == []
        state = snapshot(runner.factory, runner.room_id, previous, resumed_ids=[delivery_id])
        assert state["photos"] == [{"sha256": "a" * 64, "message_id": message_id}]
        assert [row["id"] for row in state["deliveries"]] == [delivery_id]
    finally:
        runner.close()


def interruption_example():
    original = [
        {"id": "sent", "message_id": "m1", "status": "SENT", "remote_id": "41", "sequence": 1,
         "tries": 1, "content_sha256": "original-text-1",
         "provenance": {"filename": "31_Original.docx"}},
        {"id": "pending", "message_id": "m2", "status": "PENDING", "remote_id": None, "sequence": 2,
         "tries": 0, "content_sha256": "original-text-2",
         "provenance": {"filename": "31_Original.docx"}},
    ]
    runner = SimpleNamespace(delivery_states=lambda identifiers: copy.deepcopy(original))
    state = result(deliveries=copy.deepcopy(original), flow={"instruction_plan": {"cursor": 3, "digest": "original"}})
    interrupted = validate_interruption(runner, state, {"interrupt": True}, None)
    return runner, state, original, interrupted


def test_interruption_requires_observed_unfinished_and_paused_output():
    runner, state, rows, interrupted = interruption_example()
    rows[1].update(status="SENT", remote_id="42")
    with pytest.raises(EvaluationFailure, match="InterruptionNotObserved"):
        validate_interruption(runner, state, {"interrupt": True}, None)
    with pytest.raises(EvaluationFailure, match="InterruptionNotObserved"):
        validate_interruption(runner, state, {"pause_previous": True}, interrupted)


def test_interruption_and_resume_reuse_exact_rows_and_preserve_position():
    runner, state, rows, interrupted = interruption_example()
    rows[1]["status"] = "PAUSED"
    validate_interruption(runner, state, {"pause_previous": True}, interrupted)
    validate_interruption(runner, state, {"keep_paused": True}, interrupted)
    rows[1].update(status="SENT", remote_id="42")
    state["deliveries"] = copy.deepcopy(rows)
    assert validate_interruption(runner, state, {"resume_previous": True}, interrupted) == interrupted


@pytest.mark.parametrize("change,code", [
    ("restarted", "ResumedOutputNotObserved"),
    ("position", "ResumePositionChanged"),
    ("source", "InterruptedOutputChanged"),
    ("content", "InterruptedOutputChanged"),
    ("replayed", "SentOutputReplayed"),
    ("retried", "SentOutputReplayed"),
])
def test_resume_rejects_restarted_changed_or_replayed_output(change, code):
    runner, state, rows, interrupted = interruption_example()
    rows[1]["status"] = "PAUSED"
    validate_interruption(runner, state, {"pause_previous": True}, interrupted)
    rows[1].update(status="SENT", remote_id="42")
    state["deliveries"] = copy.deepcopy(rows)
    if change == "restarted":
        state["deliveries"][1]["id"] = "new-delivery"
    if change == "position":
        state["flow"]["instruction_plan"]["cursor"] = 0
    if change == "source":
        rows[1]["provenance"] = {"filename": "37_Wrong.docx"}
    if change == "content":
        rows[1]["content_sha256"] = "replaced-text"
    if change == "replayed":
        rows[0]["remote_id"] = "99"
    if change == "retried":
        rows[0]["tries"] += 1
    with pytest.raises(EvaluationFailure, match=code):
        validate_interruption(runner, state, {"resume_previous": True}, interrupted)


def test_old_output_stays_paused_until_explicit_resume():
    runner, state, rows, interrupted = interruption_example()
    rows[1]["status"] = "PAUSED"
    validate_interruption(runner, state, {"pause_previous": True}, interrupted)
    rows[1].update(status="SENT", remote_id="42")
    with pytest.raises(EvaluationFailure, match="PreviousOutputResumedEarly"):
        validate_interruption(runner, state, {"keep_paused": True}, interrupted)


def test_resume_does_not_replay_a_delivery_completed_while_the_new_input_arrived():
    runner, state, rows, interrupted = interruption_example()
    extra = {**rows[1], "id": "last", "message_id": "m3", "sequence": 3}
    rows.append(extra)
    state["deliveries"] = copy.deepcopy(rows)
    interrupted = validate_interruption(runner, state, {"interrupt": True}, None)
    rows[1].update(status="SENT", remote_id="42", tries=1)
    rows[2]["status"] = "PAUSED"
    validate_interruption(runner, state, {"pause_previous": True}, interrupted)
    rows[1]["tries"] = 2
    rows[2].update(status="SENT", remote_id="43")
    state["deliveries"] = copy.deepcopy(rows)
    with pytest.raises(EvaluationFailure, match="SentOutputReplayed"):
        validate_interruption(runner, state, {"resume_previous": True}, interrupted)
