"""Dialogue regressions: clarification is part of an attempt, not another solution."""

import json

import pytest
from sqlalchemy import select, update

from app.core.config import get_settings
from app.models import Incident, Job, KnowledgeDocument, SolutionAttempt
from app.services.llm.mock import MockLLM
from app.services.rag.openwebui import OpenWebUIKnowledge
from tests.test_illustrations import document_transport, docx
from tests.test_integrations import post_talk, setup_talk
from tests.test_integrations import remote_settings
from tests.test_workflow import start, view


def clarify(env, incident_id, message="Где находится эта кнопка?", key=None):
    response = env.post(
        "/api/chat", {"incident_id": incident_id, "message": message}, key
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_followup_keeps_current_attempt_and_feedback_contract(env):
    incident_id = start(env)
    env.drain()
    before = view(env, incident_id)
    attempt = before["attempts"][0]

    clarify(env, incident_id)
    assert env.post(
        f"/api/incidents/{incident_id}/feedback",
        {"attempt_id": attempt["id"], "helped": True},
    ).status_code == 409  # Do not race a pending clarification response.
    env.drain()
    after = view(env, incident_id)

    assert after["state"] == "WAITING_FEEDBACK"
    assert after["pending_action"] is None
    assert after["failed_attempts"] == 0
    assert [row["id"] for row in after["attempts"]] == [attempt["id"]]
    assert after["attempts"][0]["status"] == "WAITING_FEEDBACK"
    assert after["messages"][-1]["role"] == "assistant"
    assert len(after["messages"]) > len(before["messages"])
    # Feedback must still address the original instruction after clarification.
    response = env.post(
        f"/api/incidents/{incident_id}/feedback",
        {"attempt_id": attempt["id"], "helped": True},
    )
    assert response.status_code == 202, response.text
    assert view(env, incident_id)["state"] == "RESOLVED"


def test_followup_retry_and_replayed_job_cannot_duplicate_reply(env):
    incident_id = start(env)
    env.drain()
    first = clarify(env, incident_id, key="clarification-retry")
    repeated = clarify(env, incident_id, key="clarification-retry")
    assert repeated == first
    env.drain()
    before = view(env, incident_id)
    with env.factory() as session, session.begin():
        job = session.get(Job, first["operation_id"])
        assert job.status == "DONE"
        job.status = "PENDING"  # Simulate crash after business commit.
    env.drain()
    after = view(env, incident_id)
    assert after["messages"] == before["messages"]
    assert len(after["attempts"]) == 1
    assert after["failed_attempts"] == 0


def test_failed_feedback_after_followup_counts_only_once(env):
    incident_id = start(env)
    env.drain()
    attempt_id = view(env, incident_id)["attempts"][0]["id"]
    clarify(env, incident_id)
    env.drain()
    payload = {"attempt_id": attempt_id, "helped": False, "comment": "Кнопка неактивна"}
    endpoint = f"/api/incidents/{incident_id}/feedback"
    assert env.post(endpoint, payload).status_code == 202
    assert env.post(endpoint, payload).status_code == 202
    env.drain()
    after = view(env, incident_id)
    assert after["failed_attempts"] == 1
    assert len(after["attempts"]) == 2
    assert after["attempts"][0]["status"] == "FAILED"


def test_revoked_instruction_cannot_be_reused_for_followup(env):
    incident_id = start(env)
    env.drain()
    before = view(env, incident_id)
    approved_steps = before["attempts"][0]["evidence"]["steps"]
    with env.factory() as session, session.begin():
        session.execute(update(KnowledgeDocument).values(approved=False))
    clarify(env, incident_id)
    env.drain()
    after = view(env, incident_id)
    new_assistant = [
        row for row in after["messages"][len(before["messages"]):]
        if row["role"] == "assistant"
    ]
    assert new_assistant
    assert all(step not in row["content"] for step in approved_steps for row in new_assistant)
    assert all(row["attachments"] == [] for row in new_assistant)
    assert after["failed_attempts"] == 0
    assert len(after["attempts"]) == 1


def test_requesting_human_after_followup_preserves_unassessed_attempt(env):
    incident_id = start(env)
    env.drain()
    clarify(env, incident_id)
    env.drain()
    assert env.post("/api/chat", {"incident_id": incident_id, "message": "оператор"}).status_code == 202
    after = view(env, incident_id)
    assert after["attempts"][0]["status"] == "UNASSESSED"
    assert after["failed_attempts"] == 0
    assert after["pending_action"] == "COLLECT_ESCALATION_DETAILS"


def test_talk_accepts_followup_without_feedback_command(env, monkeypatch):
    setup_talk(monkeypatch)
    initial = post_talk(env, "Принтер не печатает").json()
    env.drain()
    with env.factory() as session:
        original = session.scalar(select(SolutionAttempt).where(
            SolutionAttempt.incident_id == initial["incident_id"]
        )).id
    result = post_talk(env, "Где находится эта кнопка?", message_id="followup")
    assert result.status_code == 200, result.text
    assert result.json().get("accepted") is not False
    env.drain()
    with env.factory() as session:
        incident = session.get(Incident, initial["incident_id"])
        attempts = session.scalars(select(SolutionAttempt).where(
            SolutionAttempt.incident_id == incident.id
        )).all()
        assert incident.state == "WAITING_FEEDBACK"
        assert incident.failed_attempts == 0
        assert [row.id for row in attempts] == [original]


def test_followup_no_answer_is_explicit_and_keeps_attempt_open(env):
    incident_id = start(env)
    env.drain()
    before = view(env, incident_id)
    clarify(env, incident_id, "Как перепрошить контроллер по UART?")
    env.drain()
    after = view(env, incident_id)
    content = after["messages"][-1]["content"].lower()
    assert "баз" in content and "нет" in content
    assert "uart" not in content
    assert after["messages"][-1]["attachments"] == []
    assert after["attempts"] == before["attempts"]
    assert after["state"] == "WAITING_FEEDBACK"
    assert after["failed_attempts"] == 0


@pytest.mark.parametrize(
    "bad_fields",
    [
        {"step_indices": [999]},
        {"step_indices": [-1]},
        {"step_indices": [True]},
        {"illustration_indices": [999]},
        {"source_chunk_ids": ["invented-source"]},
        {"technical_answer": "DELETE ALL DATA"},
    ],
)
def test_followup_model_cannot_invent_sources_actions_or_image_indices(env, bad_fields):
    class InventedFollowup(MockLLM):
        def followup(self, question, context, approved_source):
            return {
                "result": "EXCERPT",
                "source_chunk_ids": [approved_source["chunk_id"]],
                "step_indices": [0],
                "illustration_indices": [],
                **bad_fields,
            }

    incident_id = start(env)
    env.drain()
    before = view(env, incident_id)
    clarify(env, incident_id)
    env.drain(InventedFollowup())
    after = view(env, incident_id)
    answer = after["messages"][-1]
    assert "DELETE" not in answer["content"]
    assert "invented-source" not in answer["content"]
    assert answer["attachments"] == []
    assert after["state"] == "WAITING_FEEDBACK"
    assert after["attempts"] == before["attempts"]
    assert after["failed_attempts"] == 0


def test_followup_uses_saved_original_photo_and_caption(env, monkeypatch):
    get_settings().knowledge_backend = "openwebui"
    provider = OpenWebUIKnowledge(remote_settings(), document_transport(docx()))
    monkeypatch.setattr("app.workers.handlers.OpenWebUIKnowledge", lambda _: provider)

    class SelectPhoto(MockLLM):
        seen = None

        def followup(self, question, context, approved_source):
            self.seen = (question, context, approved_source)
            return {
                "result": "EXCERPT",
                "source_chunk_ids": [approved_source["chunk_id"]],
                "step_indices": [0],
                "illustration_indices": [0],
            }

    incident_id = start(env, "PDF с изображениями не печатается")
    env.drain()
    before = view(env, incident_id)
    original = next(photo for row in before["messages"] for photo in row["attachments"])
    original_bytes = env.client.get(original["url"]).content
    model = SelectPhoto()
    clarify(env, incident_id, "Покажите ещё раз кнопку Дополнительно")
    env.drain(model)
    after = view(env, incident_id)
    photos = [
        photo for row in after["messages"][len(before["messages"]):]
        for photo in row["attachments"]
    ]
    assert model.seen is not None
    assert len(photos) == 1
    assert photos[0]["caption"] == original["caption"]
    assert env.client.get(photos[0]["url"]).content == original_bytes
    assert len(after["attempts"]) == 1
    assert after["failed_attempts"] == 0


def test_feedback_comment_reaches_next_analysis_as_context(env):
    class CaptureContext(MockLLM):
        seen = []

        def analyze_context(self, message, context):
            self.seen.append((message, context))
            return super().analyze(message)

    incident_id = start(env)
    model = CaptureContext()
    env.drain(model)
    attempt_id = view(env, incident_id)["attempts"][0]["id"]
    response = env.post(
        f"/api/incidents/{incident_id}/feedback",
        {"attempt_id": attempt_id, "helped": False, "comment": "Кнопка неактивна после перезапуска"},
    )
    assert response.status_code == 202
    env.drain(model)
    assert len(model.seen) == 2
    context = json.dumps(model.seen[-1][1], ensure_ascii=False)
    assert "Кнопка неактивна после перезапуска" in context


def test_source_revoked_during_followup_selection_is_not_delivered(env):
    class RevokeWhileSelecting(MockLLM):
        def followup(self, question, context, approved_source):
            with env.factory() as session, session.begin():
                session.execute(update(KnowledgeDocument).values(approved=False))
            return {
                "result": "EXCERPT",
                "source_chunk_ids": [approved_source["chunk_id"]],
                "step_indices": [0],
                "illustration_indices": [],
            }

    incident_id = start(env)
    env.drain()
    before = view(env, incident_id)
    steps = before["attempts"][0]["evidence"]["steps"]
    clarify(env, incident_id)
    env.drain(RevokeWhileSelecting())
    after = view(env, incident_id)
    new_assistant = [row for row in after["messages"][len(before["messages"]):]
                     if row["role"] == "assistant"]
    assert new_assistant
    assert all(step not in row["content"] for step in steps for row in new_assistant)
    assert after["failed_attempts"] == 0
    assert len(after["attempts"]) == 1


def test_missing_connection_asks_a_bounded_question_without_consuming_attempt(env):
    class NeedConnection(MockLLM):
        def analyze_context(self, message, context):
            return {
                "intent": "REPORT_PROBLEM",
                "problem_type": "printer_not_printing",
                "equipment_model": None,
                "search_query": "принтер не печатает",
                "clarification_key": "CONNECTION",
            }

        def compose(self, candidate):
            raise AssertionError("Do not compose a solution before essential clarification")

    incident_id = start(env)
    env.drain(NeedConnection())
    result = view(env, incident_id)
    assert result["result"] == "NEED_MORE_INFO"
    assert result["pending_action"] is None
    assert result["attempts"] == []
    assert result["failed_attempts"] == 0
    assert "подключ" in result["messages"][-1]["content"].lower()
    assert result["messages"][-1]["attachments"] == []
    response = env.post("/api/chat", {"incident_id": incident_id, "message": "Через USB"})
    assert response.status_code == 202
    env.drain()
    assert len(view(env, incident_id)["attempts"]) == 1


def test_followup_provider_outage_does_not_claim_knowledge_is_missing(env):
    class ProviderDown(MockLLM):
        def followup(self, question, context, approved_source):
            raise TimeoutError("do-not-expose-provider-detail")

    incident_id = start(env)
    env.drain()
    before = view(env, incident_id)
    clarify(env, incident_id)
    env.drain(ProviderDown())
    after = view(env, incident_id)
    answer = after["messages"][-1]["content"].lower()
    assert "недоступ" in answer
    assert "нет ответа" not in answer
    assert "provider-detail" not in answer
    assert after["state"] == "WAITING_FEEDBACK"
    assert after["attempts"] == before["attempts"]
    assert after["failed_attempts"] == 0


def test_confirmed_equipment_is_recalled_in_new_incident_after_worker_restart(env):
    class EquipmentModel(MockLLM):
        context = None

        def analyze_context(self, message, context):
            self.context = context
            result = self.analyze(message)
            if "Zebra ZD420" in message:
                result.equipment_model = "Zebra ZD420"
            return result

    first_model = EquipmentModel()
    first_id = start(env, "Принтер Zebra ZD420 не печатает по USB")
    env.drain(first_model)
    attempt_id = view(env, first_id)["attempts"][0]["id"]
    assert env.post(
        f"/api/incidents/{first_id}/feedback", {"attempt_id": attempt_id, "helped": True}
    ).status_code == 202

    restarted_model = EquipmentModel()  # No in-process model memory is carried over.
    next_id = start(env, "Сканер не считывает штрихкод")
    env.drain(restarted_model)
    assert next_id != first_id
    recalled = restarted_model.context["long_term"]
    assert len(recalled) == 1
    assert recalled[0]["incident_id"] == first_id
    assert recalled[0]["equipment_model"] == "Zebra ZD420"
    assert "steps" not in recalled[0] and "content" not in recalled[0]
