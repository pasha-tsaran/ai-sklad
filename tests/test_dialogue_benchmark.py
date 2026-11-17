"""The live benchmark reports actual stages and never prints source or secret text."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.llm.mock import MockLLM
from scripts.check_support_dialogues import (
    ControllerRunner, EvaluationFailure, TalkRunner, run_cases, source_names, validate_turn,
)


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
