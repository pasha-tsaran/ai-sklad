"""Employees can report progress and problems naturally without changing source advice."""

import pytest
from sqlalchemy import select

from app.models import Escalation, MessageAttachment, SolutionAttempt
from app.schemas.contracts import AnalysisResult
from app.services.chat.talk_dialogue import CLARIFY_QUESTIONS, INTAKE_QUESTIONS
from app.services.llm.mock import MockLLM
from tests.test_talk_dialogue import (
    DOC, OTHER, QUESTION, Knowledge, NoGeneration, _profile, approved, assistants, talk as talk,
)


@pytest.mark.parametrize("reply", [
    "Спасибо! Всё заработало!", "Сканер теперь работает", "Теперь сканирует!",
    "Проблема решена.", "Да, помогло", "Получилось, всё работает!", "Да, всё заработало, спасибо большое!",
    "Вот теперь всё работает, благодарю!",
])
def test_explicit_natural_resolution_closes_only_the_reported_approved_attempt(talk, reply):
    # A model-free acknowledgement avoids an extra generation just to classify
    # success. The attempt and source were already established by the backend.
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    before = talk.snapshot()
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["incident"].state == "RESOLVED" and after["attempts"][0].status == "SUCCEEDED"
    assert after["attempts"][0].id == before["attempts"][0].id and after["incident"].failed_attempts == 0
    assert after["flow"] == {} and len(knowledge.searches) == 1 and not model.calls
    assert [photo.data for _, photo in assistants(after) if photo] == [photo.data for _, photo in assistants(before) if photo]


@pytest.mark.parametrize("reply", ["Теперь принтер печатает", "Теперь печатает!"])
def test_printer_outcome_is_success_for_the_current_printing_problem(talk, reply):
    knowledge, model = Knowledge([approved(1)]), NoGeneration()
    talk.incoming("Принтер не печатает документ PDF")
    talk.process(knowledge, model)
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    assert talk.snapshot()["incident"].state == "RESOLVED"
    assert talk.snapshot()["attempts"][0].status == "SUCCEEDED" and not model.calls


def test_success_about_a_different_device_is_not_success_for_the_current_guide(talk):
    knowledge, model = Knowledge([approved(1)]), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    talk.incoming("Теперь принтер печатает", 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["incident"].state != "RESOLVED"
    assert all(attempt.status != "SUCCEEDED" for attempt in after["attempts"])


def test_new_paper_jam_is_not_misread_as_difficulty_with_the_existing_scanner_step(talk):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    report = "Застрял лист в принтере, что делать?"
    talk.incoming(report, 2)
    talk.process(knowledge, model)
    assert knowledge.searches[-1][0] == report and len(knowledge.searches) == 2
    assert not knowledge.reads and talk.snapshot()["incident"].state != "RESOLVED"


@pytest.mark.parametrize("reply", ["Да", "Получилось", "Да, получилось", "Готово", "Сделал это, что дальше?"])
def test_short_confirmation_advances_steps_but_never_claims_the_whole_problem_is_solved(talk, reply):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["incident"].state == "WAITING_FEEDBACK" and after["attempts"][0].status == "WAITING_FEEDBACK"
    assert not after["flow"]["has_more"] and len(after["attempts"]) == 1
    assert sum(bool(photo) for _, photo in assistants(after)) == 8
    assert len(knowledge.searches) == 1 and not model.calls


@pytest.mark.parametrize("reply", ["Нет, не помогло", "Не сработало", "Проблема осталась", "Сделал, но не помогло"])
def test_natural_failure_offers_a_different_approved_source_without_repair_invention(talk, reply):
    knowledge, model = Knowledge([approved(1), approved(1, OTHER)]), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert len(after["attempts"]) == 2 and after["attempts"][0].status == "FAILED"
    assert after["attempts"][1].status == "WAITING_FEEDBACK" and after["incident"].failed_attempts == 1
    assert after["flow"]["source"]["file_id"] == OTHER
    assert knowledge.searches[-1][1] == (DOC,) and not model.calls


@pytest.mark.parametrize("reply", [
    "Застрял на этом шаге", "Объясни попроще, пожалуйста", "У меня такого пункта нет",
    "Куда здесь нажать?", "Кнопка недоступна", "Не разобралась с этим шагом", "Не нашёл эту панель", "Не получилось",
])
def test_free_current_step_difficulty_repeats_current_original_photo_without_spending_an_attempt(talk, reply):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    before = talk.snapshot()
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    newest = assistants(after)[len(assistants(before)):]
    assert after["flow"] == before["flow"] and after["attempts"][0].id == before["attempts"][0].id
    assert len(after["attempts"]) == 1 and after["incident"].failed_attempts == 0
    photo = next(photo for _, photo in newest if photo)
    assert photo.data == knowledge.documents[0].illustrations[1].data
    assert any(photo.caption in text for text, _ in newest)
    assert len(knowledge.searches) == 1 and not model.calls


def test_printer_wifi_question_after_escalation_choice_resumes_search_instead_of_becoming_ticket_data(talk):
    knowledge, model = Knowledge(), NoGeneration()
    knowledge.missing = True
    talk.incoming("Как подключить принтер к ТСД?")
    talk.process(knowledge, model)
    assert talk.snapshot()["flow"]["escalation"]["stage"] == "CHOICE"
    model.calls.clear()  # The initial genuine miss attempted the existing analysis fallback.
    knowledge.missing = False
    question = "Как подключить мобильный принтер к ТСД по Wi-Fi?"
    talk.incoming(question, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["incident"].pending_action is None and after["incident"].state == "WAITING_FEEDBACK"
    assert len(after["attempts"]) == 1 and "escalation" not in after["flow"]
    assert question in knowledge.searches[-1][0] and sum(bool(photo) for _, photo in assistants(after)) == 6
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
    assert not model.calls


@pytest.mark.parametrize("reply", ["Теперь работает?", "Если заработает, будет хорошо", "Принтер работает, но сканер нет"])
def test_question_hypothesis_or_mixed_report_does_not_record_a_success(talk, reply):
    knowledge, model = Knowledge([approved(1)]), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["incident"].state != "RESOLVED"
    assert all(attempt.status != "SUCCEEDED" for attempt in after["attempts"])


def test_new_question_with_no_answer_exits_choice_without_inventing_a_source_or_mail(talk):
    knowledge, model = Knowledge(), MockLLM()
    knowledge.missing = True
    talk.incoming("Как подключить принтер к ТСД?")
    talk.process(knowledge, model)
    question = "Как подключить мобильный принтер к ТСД по Wi-Fi?"
    talk.incoming(question, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert not after["attempts"] and any(question in query for query, _, _ in knowledge.searches)
    assert "нет ответа" in assistants(after)[-1][0] or "нет подходящей" in assistants(after)[-1][0]
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
        assert session.scalar(select(MessageAttachment)) is None


def test_long_factual_connection_answer_keeps_the_problem_and_does_not_repeat_the_question(talk):
    class ClarifyConnection(MockLLM):
        def __init__(self):
            self.inputs = []

        def analyze_context(self, message, context):
            self.inputs.append(message)
            return AnalysisResult(intent="REPORT_PROBLEM", search_query=message, clarification_key="CONNECTION")

    knowledge, model = Knowledge(), ClarifyConnection()
    knowledge.missing = True
    problem = "Мобильный принтер не печатает с ТСД"
    talk.incoming(problem)
    talk.process(knowledge, model)
    assert assistants(talk.snapshot())[-1] == (CLARIFY_QUESTIONS["CONNECTION"], None)
    knowledge.missing = False
    factual = "Он подключается по Wi-Fi к беспроводной сети на нашем рабочем месте рядом с терминалом"
    talk.incoming(factual, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert knowledge.searches[-1][2] == problem + "\n" + factual
    assert after["flow"]["query"] == problem + "\n" + factual and model.inputs == [problem]
    assert len(after["attempts"]) == 1


def test_actual_form_subject_and_description_are_literal_even_if_they_look_like_questions(talk):
    knowledge, model = Knowledge(), NoGeneration()
    _profile(talk)
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    for sequence, content in enumerate(["оператор", "1", "terminal76-sk"], 1):
        talk.incoming(content, sequence)
        talk.process(knowledge, model)
    assert assistants(talk.snapshot())[-1] == (INTAKE_QUESTIONS["subject"], None)
    subject = "Как подключить мобильный принтер к ТСД?"
    description = "Почему на терминале нет печати? Принтер подключён по Wi-Fi, нужна помощь специалиста."
    for sequence, content in enumerate([subject, description], 4):
        talk.incoming(content, sequence)
        talk.process(knowledge, model)
    with talk.env.factory() as session:
        draft = session.scalar(select(Escalation))
        assert draft.details["subject"] == subject and draft.details["additional_info"] == description
        assert session.scalar(select(SolutionAttempt)) is None
    assert not knowledge.searches and not model.calls


def test_a_complete_unlabelled_letter_still_works_directly_from_the_choice_menu(talk):
    knowledge, model = Knowledge(), NoGeneration()
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    talk.incoming("оператор")
    talk.process(knowledge, model)
    assert talk.snapshot()["flow"]["escalation"]["stage"] == "CHOICE"
    talk.incoming("Павел\ntsaran\nterminal76-sk\nКак подключить принтер к ТСД?\nПринтер не работает по Wi-Fi.", 2)
    talk.process(knowledge, model)
    with talk.env.factory() as session:
        draft = session.scalar(select(Escalation))
        assert draft.details["subject"] == "Как подключить принтер к ТСД?"
        assert draft.details["additional_info"] == "Принтер не работает по Wi-Fi."
        assert session.scalar(select(SolutionAttempt)) is None
    assert not knowledge.searches and not model.calls


def test_retrieval_cache_key_is_saved_only_with_a_new_guide_and_survives_step_clarification(talk):
    knowledge, model = Knowledge(), NoGeneration()
    from app.services.rag.success import cache_key

    key = cache_key(talk.settings, QUESTION, {"incident": {}})
    knowledge.last_cache_key = {"untrusted_delegate_key": True}
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    assert talk.snapshot()["flow"]["retrieval_key"] == key
    knowledge.last_cache_key = {"different": True}
    talk.incoming("Объясни попроще", 2)
    talk.process(knowledge, model)
    assert talk.snapshot()["flow"]["retrieval_key"] == key and not model.calls
