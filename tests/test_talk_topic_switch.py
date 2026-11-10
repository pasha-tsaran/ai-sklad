"""A continued work question is never implicit consent to prepare a letter."""

import copy

import pytest
from sqlalchemy import select

from app.models import Escalation, TalkRoom
from app.services.chat.talk_dialogue import INTAKE_QUESTIONS
from app.schemas.contracts import DialogueResult
from tests.test_talk_dialogue import Knowledge, NoGeneration, _profile, assistants, talk as talk


PROBLEM = "как подключить тсд к принтеру?"


def choice(talk, knowledge, model):
    knowledge.missing = True
    talk.incoming(PROBLEM)
    assert talk.process(knowledge, model)
    assert talk.snapshot()["flow"]["escalation"]["stage"] == "CHOICE"


def no_letter(talk):
    state = talk.snapshot()
    assert "intake_fields" not in state["flow"].get("escalation", {})
    assert not state["flow"].get("escalation", {}).get("profile_prefilled")
    assert INTAKE_QUESTIONS["computer_name"] not in assistants(state)[-1][0]
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
    return state


def test_exact_screenshot_connection_followups_do_not_implicitly_start_a_form(talk):
    knowledge, model = Knowledge(), NoGeneration()
    choice(talk, knowledge, model)
    for sequence, clarification in enumerate(["подключить поблютуз", "А если подключить по wi-fi?"], 2):
        count = len(knowledge.searches)
        talk.incoming(clarification, sequence)
        assert talk.process(knowledge, model)
        no_letter(talk)
        assert len(knowledge.searches) > count
        assert any(clarification in query for query, _, _ in knowledge.searches[count:])
        if "wi-fi" in clarification:
            latest = " ".join(query for query, _, _ in knowledge.searches[count:]).casefold()
            assert "блютуз" not in latest and "bluetooth" not in latest


@pytest.mark.parametrize("reply", ["Да", "Готово", "понятно", "по Bluetooth"])
def test_unlabelled_choice_reply_is_not_consent_to_collect_a_letter(talk, reply):
    knowledge, model = Knowledge(), NoGeneration()
    choice(talk, knowledge, model)
    talk.incoming(reply, 2)
    talk.process(knowledge, model)
    no_letter(talk)


@pytest.mark.parametrize("field,question", [
    ("computer_name", "А если подключить по wi-fi?"),
    ("computer_name", "Как настроить сканер штрихкодов?"),
    ("account_holder_name", "Как подключить принтер к ТСД?"),
    ("subject", "Другой вопрос: как настроить сканер?"),
    ("additional_info", "А если подключить по Wi-Fi?"),
])
def test_technical_topic_switch_recovers_a_legacy_form_without_reusing_letter_data(talk, field, question):
    knowledge, model = Knowledge(), NoGeneration()
    choice(talk, knowledge, model)
    with talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, talk.room_id)
        flow = copy.deepcopy(room.flow)
        flow["escalation"].update({
            "stage": "FORM", "profile_prefilled": True, "intake_requested_field": field,
            "intake_fields": {"account_holder_name": "PRIVATE-PERSON", "account_login": "private-login",
                              "computer_name": "PRIVATE-COMPUTER", "subject": "PRIVATE-LETTER-TOPIC"},
        })
        # Reproduce both genuine partial forms and the accidentally opened
        # pre-fix form from the screenshot; the current question must escape it.
        flow["escalation"]["intake_fields"].pop(field, None)
        room.flow = flow
    knowledge.missing = False
    searches = len(knowledge.searches)
    talk.incoming(question, 2)
    assert talk.process(knowledge, model)
    state = no_letter(talk)
    assert state["incident"].pending_action is None and "escalation" not in state["flow"]
    queries = " ".join(query for query, _, _ in knowledge.searches[searches:])
    if "сканер" in question:
        # Escaping a letter is not consent to pick an arbitrary scanner guide.
        # The new broad request needs a device/goal clarification first.
        assert state["incident"].state == "DIAGNOSING" and state["flow"]["clarification_keys"]
        assert len(knowledge.searches) == searches and not state["attempts"]
        assert not any(picture for _, picture in assistants(state))
        queries += state["flow"]["clarification_query"]
        assert "принтер" not in queries.casefold() and "тсд" not in queries.casefold()
    else:
        assert state["incident"].state == "WAITING_FEEDBACK" and len(knowledge.searches) > searches
        assert sum(bool(picture) for _, picture in assistants(state)) == 6
    assert question in queries and "PRIVATE-" not in queries and "private-login" not in queries


def test_explicit_letter_still_accepts_question_shaped_subject_and_description_verbatim(talk):
    knowledge, model = Knowledge(), NoGeneration()
    _profile(talk)
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    for sequence, text in enumerate(["оператор", "1", "terminal76-sk"], 1):
        talk.incoming(text, sequence)
        assert talk.process(knowledge, model)
    assert assistants(talk.snapshot())[-1][0] == INTAKE_QUESTIONS["subject"]
    subject = "Как подключить принтер к ТСД?"
    description = "Почему нет печати? Принтер подключён по Wi-Fi, нужна помощь специалиста."
    for sequence, text in enumerate([subject, description], 4):
        talk.incoming(text, sequence)
        assert talk.process(knowledge, model)
    with talk.env.factory() as session:
        draft = session.scalar(select(Escalation))
        assert draft.details["subject"] == subject and draft.details["additional_info"] == description
    assert not knowledge.searches and not model.calls


def test_an_ordinary_new_question_leaves_the_form_and_gets_a_normal_reply(talk):
    class Friendly(NoGeneration):
        def dialogue(self, message, context):
            self.calls.append(message)
            return DialogueResult(route="SOCIAL", reply="Коллега пожелал кофе доброго утра. И себе тоже.")

    knowledge, model = Knowledge(), Friendly()
    _profile(talk)
    for sequence, message in enumerate(["оператор", "1"], 1):
        talk.incoming(message, sequence)
        talk.process(knowledge, model)
    question = "Расскажи короткую добрую шутку"
    talk.incoming(question, 3)
    talk.process(knowledge, model)
    state = no_letter(talk)
    assert state["incident"].pending_action is None
    assert assistants(state)[-1][0] == "Коллега пожелал кофе доброго утра. И себе тоже."
    assert model.calls == [question] and not knowledge.searches
