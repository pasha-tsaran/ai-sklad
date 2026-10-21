from datetime import datetime, timezone

from sqlalchemy import event, func, select, update

from app.models import (
    Conversation,
    Escalation,
    Feedback,
    Incident,
    Job,
    KnowledgeDocument,
    Message,
    Solution,
    User,
)
from app.schemas.contracts import AnswerResult
from app.services.llm.mock import MockLLM

DETAILS = {
    "account_holder_name": "Иванов Иван Иванович",
    "account_login": "ivanov",
    "computer_name": "WS-001",
    "subject": "Принтер не печатает",
    "additional_info": "Кабель проверен",
}


def test_message_order_is_independent_of_timestamp_and_uuid(env):
    counter = 0

    def freeze_timestamp(mapper, connection, message):
        nonlocal counter
        counter += 1
        message.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        message.id = f"{1000 - counter:036d}"

    event.listen(Message, "before_insert", freeze_timestamp)
    try:
        incident_id = start(env)
        env.drain()
        rate(env, incident_id, view(env, incident_id)["attempts"][0]["id"])
        env.drain()
        messages = view(env, incident_id)["messages"]
        assert [message["sequence"] for message in messages] == [1, 2, 3, 4]
        assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
        assert env.post("/api/chat", {"incident_id": incident_id, "message": "оператор"}).status_code == 202
        assert env.post(f"/api/incidents/{incident_id}/escalation-details", DETAILS).status_code == 202
        env.drain()
        body = view(env, incident_id)["escalation"]["mock_email"]["body"]
        assert body.index("Пользователь: Принтер не печатает") < body.index("Пользователь: Не помогло")
        assert body.index("Пользователь: Не помогло") < body.index("Пользователь: оператор")
    finally:
        event.remove(Message, "before_insert", freeze_timestamp)


def start(env, message="Принтер не печатает"):
    response = env.post("/api/chat", {"message": message})
    assert response.status_code == 202, response.text
    return response.json()["incident_id"]


def view(env, incident_id):
    response = env.client.get(f"/api/incidents/{incident_id}")
    assert response.status_code == 200, response.text
    return response.json()


def rate(env, incident_id, attempt_id, helped=False):
    response = env.post(
        f"/api/incidents/{incident_id}/feedback", {"attempt_id": attempt_id, "helped": helped}
    )
    assert response.status_code == 202, response.text
    return response


def test_five_failures_form_and_one_mock_email(env):
    incident_id = start(env)
    env.drain()
    solutions = set()
    for ordinal in range(1, 6):
        incident = view(env, incident_id)
        assert incident["state"] == "WAITING_FEEDBACK"
        assert len(incident["attempts"]) == ordinal
        attempt = incident["attempts"][-1]
        assert attempt["solution_id"] not in solutions
        solutions.add(attempt["solution_id"])
        rate(env, incident_id, attempt["id"])
        # A retry with a different idempotency key still cannot count feedback twice.
        rate(env, incident_id, attempt["id"])
        assert view(env, incident_id)["failed_attempts"] == ordinal
        env.drain()
    incident = view(env, incident_id)
    assert incident["pending_action"] == "COLLECT_ESCALATION_DETAILS"
    assert incident["escalation"] is None
    assert len(incident["attempts"]) == 5
    assert "| ФИО" in incident["messages"][-1]["content"]
    assert env.post("/api/chat", {"incident_id": incident_id, "message": "Ещё решение"}).status_code == 409
    endpoint = f"/api/incidents/{incident_id}/escalation-details"
    assert env.post(endpoint, {"subject": "неполная форма"}).status_code == 422
    assert env.post(endpoint, DETAILS).status_code == 202
    assert env.post(endpoint, DETAILS).status_code == 202
    env.drain()
    incident = view(env, incident_id)
    assert incident["state"] == "ESCALATED"
    assert incident["escalation"]["status"] == "SENT_MOCK"
    email = incident["escalation"]["mock_email"]
    assert DETAILS["subject"] in email["subject"]
    for expected in (
        DETAILS["account_holder_name"],
        "WS-001",
        "Пользователь:",
        "Помощник (бот):",
        "Время обращения",
    ):
        assert expected in email["body"]
    with env.factory() as session:
        assert session.scalar(select(func.count()).select_from(Escalation)) == 1
        assert session.scalar(select(func.count()).select_from(Feedback)) == 5
        assert session.scalar(select(func.count()).select_from(Job).where(Job.kind == "SEND_MAIL")) == 1


def test_resolved_updates_stats_and_raises_priority(env):
    incident_id = start(env)
    env.drain()
    first = view(env, incident_id)["attempts"][-1]
    rate(env, incident_id, first["id"])
    env.drain()
    second = view(env, incident_id)["attempts"][-1]
    rate(env, incident_id, second["id"], helped=True)
    assert view(env, incident_id)["state"] == "RESOLVED"
    stats = env.client.get("/api/admin/solutions/stats").json()["solutions"]
    winner = next(row for row in stats if row["solution_id"] == second["solution_id"])
    assert winner["success_rate"] == 1
    assert winner["successes"] == 1
    next_incident = start(env)
    env.drain()
    assert view(env, next_incident)["attempts"][0]["solution_id"] == second["solution_id"]


def test_no_knowledge_never_composes_or_consumes_attempt(env):
    class MustNotCompose(MockLLM):
        def compose(self, candidate):
            raise AssertionError("No generation without evidence")

    incident_id = start(env, "Сломался космический телескоп")
    env.drain(MustNotCompose())
    incident = view(env, incident_id)
    assert incident["result"] == "NO_KNOWLEDGE"
    assert incident["attempts"] == []
    assert incident["failed_attempts"] == 0
    assert incident["pending_action"] == "COLLECT_ESCALATION_DETAILS"


def test_unapproved_knowledge_is_excluded(env):
    with env.factory() as session, session.begin():
        session.execute(update(KnowledgeDocument).values(approved=False))
    incident_id = start(env)
    env.drain()
    assert view(env, incident_id)["result"] == "NO_KNOWLEDGE"


def test_knowledge_revoked_during_generation_is_not_delivered(env):
    class RevokeDuringGeneration(MockLLM):
        def compose(self, candidate):
            with env.factory() as session, session.begin():
                session.execute(update(KnowledgeDocument).values(approved=False))
            return super().compose(candidate)

    incident_id = start(env)
    env.drain(RevokeDuringGeneration())
    incident = view(env, incident_id)
    assert incident["attempts"] == []
    assert incident["result"] == "NO_KNOWLEDGE"


def test_fewer_solutions_escalates_early(env):
    with env.factory() as session, session.begin():
        session.execute(update(Solution).where(Solution.priority > 1).values(active=False))
    incident_id = start(env)
    env.drain()
    rate(env, incident_id, view(env, incident_id)["attempts"][0]["id"])
    env.drain()
    incident = view(env, incident_id)
    assert incident["failed_attempts"] == 1
    assert len(incident["attempts"]) == 1
    assert incident["pending_action"] == "COLLECT_ESCALATION_DETAILS"


def test_clarification_does_not_consume_attempt(env):
    incident_id = start(env, "Помогите")
    env.drain()
    assert view(env, incident_id)["result"] == "NEED_MORE_INFO"
    assert not view(env, incident_id)["attempts"]
    assert (
        env.post("/api/chat", {"incident_id": incident_id, "message": "Принтер не печатает"}).status_code
        == 202
    )
    env.drain()
    assert len(view(env, incident_id)["attempts"]) == 1


def test_idempotency_and_conflicting_feedback(env):
    body = {"message": "Принтер не печатает"}
    first = env.post("/api/chat", body, "same")
    repeated = env.post("/api/chat", body, "same")
    assert first.json() == repeated.json()
    assert env.post("/api/chat", {"message": "другое"}, "same").status_code == 409
    incident_id = first.json()["incident_id"]
    assert env.post("/api/chat", body).status_code == 409  # pending processing
    env.drain()
    attempt_id = view(env, incident_id)["attempts"][0]["id"]
    rate(env, incident_id, attempt_id)
    assert (
        env.post(
            f"/api/incidents/{incident_id}/feedback", {"attempt_id": attempt_id, "helped": True}
        ).status_code
        == 409
    )


def test_model_cannot_invent_steps_or_control_state(env):
    class MaliciousLLM(MockLLM):
        def compose(self, candidate):
            return {
                "result": "ANSWER",
                "solution_id": candidate.solution_id,
                "source_chunk_ids": [candidate.chunk_id],
                "steps": ["DELETE ALL DATA"],
                "send_email": True,
                "failed_attempts": 5,
            }

    incident_id = start(env)
    env.drain(MaliciousLLM())
    incident = view(env, incident_id)
    assert incident["state"] == "WAITING_FEEDBACK"
    assert incident["failed_attempts"] == 0
    assert incident["escalation"] is None
    assert "DELETE" not in incident["messages"][-1]["content"]
    assert incident["attempts"][0]["evidence"]["steps"][0] in incident["messages"][-1]["content"]


def test_invalid_citations_fall_back_to_approved_card(env):
    class WrongSource(MockLLM):
        def compose(self, candidate):
            return AnswerResult(result="ANSWER", solution_id="invented", source_chunk_ids=["invented"])

    incident_id = start(env)
    env.drain(WrongSource())
    assert "invented" not in view(env, incident_id)["messages"][-1]["content"]
    assert view(env, incident_id)["state"] == "WAITING_FEEDBACK"


def test_llm_outage_consumes_no_attempt_and_summary_failure_does_not_block_mail(env):
    class BrokenLLM(MockLLM):
        def analyze(self, message):
            raise TimeoutError("secret-provider-details")

        def summarize(self, transcript):
            raise TimeoutError()

    incident_id = start(env)
    env.drain(BrokenLLM())
    incident = view(env, incident_id)
    assert incident["attempts"] == []
    assert incident["pending_action"] is None
    assert "secret" not in str(incident)
    assert env.post("/api/chat", {"incident_id": incident_id, "message": "оператор"}).status_code == 202
    assert env.post(f"/api/incidents/{incident_id}/escalation-details", DETAILS).status_code == 202
    env.drain(BrokenLLM())
    escalation = view(env, incident_id)["escalation"]
    assert escalation["status"] == "SENT_MOCK"
    assert "Summary недоступно" in escalation["mock_email"]["body"]


def test_replayed_worker_job_cannot_duplicate_attempt(env):
    incident_id = start(env)
    env.drain()
    with env.factory() as session, session.begin():
        job = session.scalar(select(Job).where(Job.incident_id == incident_id))
        job.status = "PENDING"  # Crash after business commit, before marking job DONE.
    env.drain()
    assert len(view(env, incident_id)["attempts"]) == 1
    rate(env, incident_id, view(env, incident_id)["attempts"][0]["id"])
    with env.factory() as session, session.begin():
        job = session.scalar(select(Job).where(Job.incident_id == incident_id).order_by(Job.created_at))
        job.status = "PENDING"  # Stale job must also not process a newer incident version.
    env.drain()
    assert len(view(env, incident_id)["attempts"]) == 2


def test_auth_and_ownership(env):
    assert (
        env.client.get("/api/admin/solutions/stats", headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    with env.factory() as session, session.begin():
        other = User(external_id="other", display_name="Other")
        session.add(other)
        session.flush()
        conversation = Conversation(user_id=other.id)
        session.add(conversation)
        session.flush()
        incident = Incident(conversation_id=conversation.id, subject="Other", description="Other")
        session.add(incident)
        session.flush()
        other_id = incident.id
    assert env.client.get(f"/api/incidents/{other_id}").status_code == 404
    assert (
        env.post("/api/chat", {"incident_id": other_id, "message": "Принтер не печатает"}).status_code == 404
    )


def test_abandoned_attempt_not_counted_as_failure(env):
    incident_id = start(env)
    env.drain()
    assert env.post("/api/chat", {"incident_id": incident_id, "message": "оператор"}).status_code == 202
    incident = view(env, incident_id)
    assert incident["attempts"][0]["status"] == "UNASSESSED"
    assert incident["failed_attempts"] == 0
    stats = env.client.get("/api/admin/solutions/stats").json()["solutions"]
    assert sum(s["assessed"] for s in stats) == 0
