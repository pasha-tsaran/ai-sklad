"""Direct and previously saved support forms always retain a backend reason."""

import pytest
from sqlalchemy import func, select

from app.models import Escalation, Incident, Job
from tests.test_talk_dialogue import Knowledge, NoGeneration, QUESTION, talk as talk
from tests.test_workflow import DETAILS, start


def test_direct_talk_form_completes_after_an_answer_without_visiting_operator_menu(talk):
    talk.settings.mail_backend = "mailto"
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    assert talk.process(knowledge, model)
    assert talk.snapshot()["incident"].escalation_reason is None

    talk.incoming("Оставить заявку", 2)
    assert talk.process(knowledge, model)
    assert talk.snapshot()["incident"].escalation_reason == "USER_REQUEST"
    talk.incoming("ФИО: Тестовый Сотрудник\nЛогин: employee\nКомпьютер: TEST-PC\n"
                  "Тема: Сканер\nОписание: После сканирования звук остался громким.", 3)
    assert talk.process(knowledge, model)

    assert talk.snapshot()["incident"].state == "ESCALATED" and not model.calls
    with talk.env.factory() as session:
        letter = session.scalar(select(Escalation))
        assert letter.reason == "USER_REQUEST" and letter.status == "DRAFT"
        assert session.scalar(select(func.count()).select_from(Job).where(Job.kind == "SEND_MAIL")) == 0


@pytest.mark.parametrize("saved_reason", [None, "NO_KNOWLEDGE"])
def test_api_completes_a_legacy_form_with_no_reason_and_preserves_a_known_reason(env, saved_reason):
    incident_id = start(env, "оператор")
    env.drain()
    with env.factory() as session, session.begin():
        incident = session.get(Incident, incident_id)
        assert incident.pending_action == "COLLECT_ESCALATION_DETAILS"
        incident.escalation_reason = saved_reason

    endpoint = f"/api/incidents/{incident_id}/escalation-details"
    assert env.post(endpoint, DETAILS).status_code == 202
    assert env.post(endpoint, DETAILS).status_code == 202
    with env.factory() as session:
        letter = session.scalar(select(Escalation).where(Escalation.incident_id == incident_id))
        assert letter.reason == (saved_reason or "USER_REQUEST")
        assert letter.status == "QUEUED" and letter.details == DETAILS
        assert session.get(Incident, incident_id).escalation_reason == letter.reason
        assert session.scalar(select(func.count()).select_from(Escalation)) == 1
        assert session.scalar(select(func.count()).select_from(Job).where(Job.kind == "SEND_MAIL")) == 1
