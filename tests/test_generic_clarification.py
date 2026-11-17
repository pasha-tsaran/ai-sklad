"""Source-led clarification works for existing and previously unknown instructions."""

import copy
import hashlib
import json

import httpx
import pytest
from sqlalchemy import select

from app.models import Incident, Message
from app.models.talk import TalkRoom
from app.services.chat.policies import FORM_ACTION
from app.services.chat.talk_dialogue import UNAVAILABLE
from app.services.llm.openwebui import OpenWebUILLM
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_openwebui_instruction_images import instruction, png
from tests.test_talk_dialogue import assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC, OTHER
from tests.test_talk_model_dialogue import model_chat as model_chat


def match(file_id, evidence):
    return {"result": "MATCH", "file_id": file_id, "covered_goals": [
        {"goal_id": "g1", "evidence": evidence, "support_kind": "IMAGE", "image_numbers": [1]},
    ]}


def replies_since(before, after):
    return assistants(after)[len(assistants(before)):]


@pytest.mark.parametrize("error,expected_code", [
    (ValueError("UnknownTaskGoal"), "UnknownTaskGoal"),
    (ValueError("InvalidTaskEvidence"), "InvalidTaskEvidence"),
    (ValueError("IncompleteTaskUnderstanding"), "IncompleteTaskUnderstanding"),
    (ValueError("Untrusted message with employee@example.test"), "ValueError"),
    (RuntimeError("Untrusted message with employee@example.test"), "RuntimeError"),
])
def test_task_failure_reports_only_known_machine_codes(model_chat, monkeypatch, error, expected_code):
    chat = model_chat([])
    chat.talk.settings.talk_confirm_understanding = True
    failures = []
    monkeypatch.setattr("app.services.chat.talk_dialogue.report_failure", failures.append)

    def understand(*args):
        raise error

    chat.model.understand_task = understand
    result = chat.send("Нужно открыть рабочую программу.")
    assert any(text == UNAVAILABLE for text, _ in assistants(result))
    assert [(failure.stage, failure.code) for failure in failures] == [("task_understanding", expected_code)]
    assert result["flow"]["task_state"]["messages"] == ["Нужно открыть рабочую программу."]


CASES = [
    {
        "id": "outlook",
        "opening": "Мне выдали рабочую почту. Хочу впервые открыть её в Outlook на этом компьютере.",
        "goal": "Первое подключение рабочей почты в Outlook",
        "evidence": "впервые открыть её в Outlook",
        "application": "Outlook",
        "question": "Какой тип подключения к почте используется — IMAP, POP3 или Exchange?",
        "field": "connection",
        "filename": "Первое подключение рабочей почты.docx",
        "caption": "При первом подключении рабочего ящика в Outlook выберите Exchange в окне типов учётной записи.",
    },
    {
        "id": "future-application",
        "opening": "Нужно впервые подключить рабочую папку в программе АльфаРегистр на этом компьютере.",
        "goal": "Первое подключение рабочей папки в АльфаРегистр",
        "evidence": "впервые подключить рабочую папку в программе АльфаРегистр",
        "application": "АльфаРегистр",
        "question": "Какой режим хранилища используется — S3, WebDAV или SMB?",
        "field": "other",
        "filename": "Новая инструкция подключения рабочей папки.docx",
        "caption": "Для первого подключения рабочей папки в АльфаРегистр выберите корпоративную папку на начальном экране.",
    },
    {
        "id": "mobile-printer",
        "opening": "Нужно подключить мобильный термопринтер к ТСД по Wi-Fi.",
        "goal": "Подключить мобильный термопринтер к ТСД по Wi-Fi",
        "evidence": "подключить мобильный термопринтер к ТСД по Wi-Fi",
        "application": None,
        "question": "Какой язык печати использует устройство — ZPL или CPCL?",
        "field": "other",
        "filename": "Подключение мобильного принтера к рабочему терминалу.docx",
        "caption": "Для подключения мобильного термопринтера к ТСД по Wi-Fi откройте указанное окно настройки печати.",
    },
]


@pytest.mark.parametrize("case", [
    *CASES,
    {**CASES[0], "id": "outlook-mislabelled-equipment", "field": "equipment_type"},
    {**CASES[1], "id": "future-application-mislabelled-equipment", "field": "equipment_type"},
], ids=lambda case: case["id"])
def test_clear_goal_searches_originals_before_asking_technical_conditions(model_chat, case):
    data = instruction([case["caption"]], image_data=png("blue"))
    chat = model_chat([match(DOC, case["caption"])], rows=[(DOC, case["filename"], data)])
    chat.talk.settings.talk_confirm_understanding = True
    facts = []
    if case["application"]:
        facts = [{"key": "application", "value": case["application"], "evidence": case["application"]},
                 {"key": "scope", "value": "application", "evidence": case["application"]}]

    def understand(message, previous, previous_question):
        return {"facts": facts, "goals": [{"text": case["goal"], "evidence": case["evidence"]}],
                "question": case["question"], "question_field": case["field"]}

    chat.model.understand_task = understand
    first = chat.send(case["opening"])
    assert len(chat.model.calls) == 1
    assert "understanding_confirmation" in first["flow"]
    assert all(case["question"] not in text and UNAVAILABLE not in text for text, _ in assistants(first))
    assert not first["attempts"] and not any(photo for _, photo in assistants(first))
    accepted = chat.send("Да, правильно", 2)
    photos = [photo for _, photo in replies_since(first, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
    assert len(chat.model.calls) == 1


def legacy_question(chat, opening, goal, evidence, question, *, field="connection", application="Outlook"):
    """An upgraded room can still hold a technical question asked by an older version."""
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.understand_task = lambda *args: {
        "question": "Какой результат вы хотите получить?", "question_field": "goal",
    }
    first = chat.send(opening)
    state = {"facts": [{"key": "application", "value": application, "evidence": application}],
             "goals": {"g1": {"text": goal, "evidence": evidence}}, "messages": [opening],
             "question": question, "question_field": field, "new_task": False}
    flow = {"task_state": state, "task_query": goal, "clarification_query": goal,
            "clarification_keys": ["MODEL"], "clarification_question": question,
            "opening_query": opening, "excluded": []}
    with chat.talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, chat.talk.room_id)
        room.flow = copy.deepcopy(flow)
        message = session.scalar(select(Message).where(
            Message.incident_id == room.incident_id, Message.role == "assistant",
        ).order_by(Message.sequence.desc()))
        message.content = question
    return first, state


@pytest.mark.parametrize("pending_form", [False, True])
def test_explicit_new_task_drops_an_old_confirmation_before_a_failed_understanding(model_chat, pending_form):
    case = CASES[0]
    chat = model_chat([match(DOC, case["caption"])], rows=[
        (DOC, case["filename"], instruction([case["caption"]])),
    ])
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.understand_task = lambda *args: {
        "facts": [{"key": "application", "value": "Outlook", "evidence": "Outlook"}],
        "goals": [{"text": case["goal"], "evidence": case["evidence"]}],
    }
    old = chat.send(case["opening"])
    assert old["flow"]["understanding_confirmation"]
    if pending_form:
        with chat.talk.env.factory() as session, session.begin():
            room = session.get(TalkRoom, chat.talk.room_id)
            session.get(Incident, room.incident_id).pending_action = FORM_ACTION
            room.flow = {"escalation": {"stage": "FORM", "resume": {"flow": copy.deepcopy(room.flow)}}}
    opening = "Другая задача. Нужно настроить рабочее устройство."

    def failed_understanding(message, previous, question):
        assert message == opening
        assert previous["goals"] == {} and previous["facts"] == [] and previous["messages"] == []
        assert question == ""
        raise ValueError("IncompleteTaskUnderstanding")

    chat.model.understand_task = failed_understanding
    result = chat.send(opening, 2)
    assert replies_since(old, result) == [(UNAVAILABLE, None)]
    state = result["flow"]["task_state"]
    assert state["facts"] == [] and state["goals"] == {} and state["messages"] == [opening]
    assert state["new_task"] is True
    assert result["flow"]["clarification_query"] == opening
    assert not set(result["flow"]) & {"source", "instruction_plan", "understanding_confirmation", "task_query"}
    assert result["incident"].pending_action is None
    assert len(chat.model.calls) == 1


def test_new_task_selection_uses_only_current_user_history_and_current_step_facts(model_chat):
    caption = "При подключении сканера штрихкодов откройте указанное окно настройки рабочего устройства."
    opening = "Другая задача. Настройка сканера"
    answer = "Сканер штрихкодов. При подключении на четвёртом шаге ошибка «нет связи»."
    question = "Это сканер штрихкодов или сканер документов?"

    def choose(payload):
        context = payload["context"]
        assert payload["requested_goals"] == {"g1": "Подключить сканер штрихкодов"}
        assert context["recent_messages"] == [{"role": "user", "content": opening},
                                               {"role": "user", "content": answer}]
        assert not context.get("technical_facts")
        assert "Outlook" not in json.dumps(context, ensure_ascii=False)
        facts = context["task_state"]["facts"]
        assert any(fact["key"] == "error_text" and fact["value"] == "нет связи" for fact in facts)
        assert any(fact["key"] == "current_step" for fact in facts)
        return match(DOC, caption)

    chat = model_chat([choose], rows=[(DOC, "Подключение рабочего считывателя.docx", instruction([caption]))])
    old, _ = legacy_question(chat, "Хочу открыть почту в Outlook", "Открыть почту в Outlook",
                             "открыть почту в Outlook", "Какой тип почты используется?")
    updates = [{"facts": [{"key": "equipment_type", "value": "сканер", "evidence": "сканера"}],
                "question": question, "question_field": "equipment_type"},
               {"facts": [{"key": "equipment_type", "value": "сканер штрихкодов", "evidence": "Сканер штрихкодов"},
                          {"key": "error_text", "value": "нет связи", "evidence": "нет связи"},
                          {"key": "current_step", "value": "четвёртый шаг", "evidence": "четвёртом шаге"}],
                "goals": [{"text": "Подключить сканер штрихкодов", "evidence": "При подключении"}]}]
    calls = []

    def understand(message, previous, previous_question):
        calls.append((message, copy.deepcopy(previous), previous_question))
        return updates.pop(0)

    chat.model.understand_task = understand
    asked = chat.send(opening, 2)
    assert replies_since(old, asked) == [(question, None)] and not chat.model.calls
    assert not asked["flow"]["task_state"]["goals"]
    assert calls[0][1]["goals"] == {} and calls[0][2] == ""
    result = chat.send(answer, 3)
    assert result["flow"]["understanding_confirmation"]["source"]["file_id"] == DOC
    with chat.talk.env.factory() as session:
        messages = session.scalars(select(Message.content).where(
            Message.incident_id == result["incident"].id, Message.role == "user",
        )).all()
    assert "Хочу открыть почту в Outlook" in messages


@pytest.mark.parametrize("answer", ["не знаю", "Не разбираюсь в этом", "затрудняюсь ответить"])
def test_unknown_technical_answer_recovers_original_goal_without_inventing_protocol(model_chat, answer):
    case = CASES[0]
    data = instruction([case["caption"]], image_data=png("blue"))

    def choose(payload):
        assert payload["requested_goals"] == {"g1": case["goal"]}
        unknown = payload["context"]["unknown_answers"]
        assert len(unknown) == 1
        assert unknown[0]["question"] == case["question"] and unknown[0]["answer"] == answer
        assert not any(item["key"] == "connection" for item in payload["context"]["task_state"]["facts"])
        return match(DOC, case["caption"])

    chat = model_chat([choose], rows=[(DOC, case["filename"], data)])
    before, previous = legacy_question(chat, case["opening"], case["goal"], case["evidence"], case["question"])

    def unexpected_understanding(*args):
        pytest.fail("An unambiguous unknown answer retains the existing goal without another model call")

    chat.model.understand_task = unexpected_understanding
    result = chat.send(answer, 2)
    assert result["flow"]["task_state"]["goals"] == previous["goals"]
    assert answer in result["flow"]["task_state"]["messages"]
    assert result["flow"]["task_state"]["unknown_answers"][0]["field"] == "connection"
    assert result["flow"]["understanding_confirmation"]["source"]["file_id"] == DOC
    assert all(case["question"] not in text and UNAVAILABLE not in text for text, _ in replies_since(before, result))
    accepted = chat.send("да", 3)
    photos = [photo for _, photo in replies_since(result, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
    assert not any(item["key"] == "connection" for item in accepted["flow"]["task_state"]["facts"])


def test_unknown_technical_detail_can_be_replaced_by_a_source_choice_the_employee_can_observe(model_chat):
    opening = "Нужно подключить мобильный термопринтер к программе ТерминалПечать."
    goal = "Подключить мобильный термопринтер к программе ТерминалПечать"
    technical = "Какой протокол обмена поддерживает печатающий модуль?"
    observable = "Сколько принтеров нужно подключить к этому рабочему терминалу — один или два?"
    first_caption = "При подключении одного мобильного принтера в ТерминалПечать выберите строку единственного принтера."
    second_caption = "При подключении двух мобильных принтеров в ТерминалПечать выберите строки обоих устройств."
    first_data = instruction([first_caption], image_data=png("blue"))
    second_data = instruction([second_caption], image_data=png("green"))
    chat = model_chat([
        {"result": "CLARIFY", "question": observable, "clarification_kind": "PHYSICAL_SETUP",
         "clarification_sources": [{"file_id": DOC, "evidence": first_caption},
                                   {"file_id": OTHER, "evidence": second_caption}]},
        match(DOC, first_caption),
    ], rows=[(DOC, "ТерминалПечать — один принтер.docx", first_data),
             (OTHER, "ТерминалПечать — два принтера.docx", second_data)])
    chat.model.supports_source_clarification = True
    _, previous = legacy_question(chat, opening, goal, "подключить мобильный термопринтер к программе ТерминалПечать",
                                  technical, application="ТерминалПечать")
    chat.model.understand_task = lambda *args: pytest.fail("Unknown answer must preserve the task directly")
    asked = chat.send("не знаю", 2)
    assert assistants(asked)[-1] == (observable, None)
    assert asked["flow"]["task_state"]["goals"] == previous["goals"]
    assert not asked["attempts"] and not any(photo for _, photo in assistants(asked))
    chat.model.understand_task = lambda *args: {
        "goals": [{"goal_id": "g1", "text": "Подключить один мобильный принтер к программе ТерминалПечать",
                   "evidence": "один"}],
    }
    pending = chat.send("один", 3)
    assert pending["flow"]["understanding_confirmation"]["source"]["file_id"] == DOC
    accepted = chat.send("да, правильно", 4)
    photos = [photo for _, photo in replies_since(pending, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(first_data)[0].data]
    assert all(photo.data != read_docx_images(second_data)[0].data for photo in photos)


def test_semantic_unknown_answer_preserves_the_task_beyond_the_short_phrase_aliases(model_chat):
    case = CASES[1]
    answer = "С такими словами не знаком"
    data = instruction([case["caption"]], image_data=png("green"))

    def choose(payload):
        assert payload["requested_goals"] == {"g1": case["goal"]}
        assert payload["context"]["unknown_answers"] == [{"question": case["question"], "answer": answer}]
        assert not any(item["key"] == "connection" for item in payload["context"]["task_state"]["facts"])
        return match(DOC, case["caption"])

    chat = model_chat([choose], rows=[(DOC, case["filename"], data)])
    before, previous = legacy_question(chat, case["opening"], case["goal"], case["evidence"],
                                      case["question"], field="other", application=case["application"])
    calls = []

    def understand(message, state, previous_question):
        calls.append(message)
        assert state["goals"] == previous["goals"] and previous_question == case["question"]
        return {"unknown_answer_evidence": answer}

    chat.model.understand_task = understand
    result = chat.send(answer, 2)
    assert calls == [answer]
    assert result["flow"]["task_state"]["goals"] == previous["goals"]
    assert result["flow"]["understanding_confirmation"]["source"]["file_id"] == DOC
    assert all(case["question"] not in text and UNAVAILABLE not in text for text, _ in replies_since(before, result))
    accepted = chat.send("да", 3)
    assert [photo.data for _, photo in replies_since(result, accepted) if photo] == [read_docx_images(data)[0].data]


def test_strict_source_clarification_does_not_repeat_a_technical_question_answered_unknown(model_chat):
    case = CASES[0]
    data = instruction([case["caption"]], image_data=png("blue"))
    chat = model_chat([
        {"result": "CLARIFY", "question": case["question"], "clarification_kind": "VISIBLE_STATE",
         "clarification_sources": [{"file_id": DOC, "evidence": case["caption"]}]},
        match(DOC, case["caption"]),
    ], rows=[(DOC, case["filename"], data)])
    chat.model.supports_source_clarification = True
    before, _ = legacy_question(chat, case["opening"], case["goal"], case["evidence"], case["question"])
    chat.model.understand_task = lambda *args: pytest.fail("Unknown answer keeps the known task")
    result = chat.send("не знаю", 2)
    assert result["flow"]["understanding_confirmation"]["source"]["file_id"] == DOC
    assert all(case["question"] not in text and UNAVAILABLE not in text for text, _ in replies_since(before, result))
    assert len(chat.model.calls) == 2
    assert chat.model.calls[-1]["payload"]["source_question_retry"] is True
    accepted = chat.send("да", 3)
    photos = [photo for _, photo in replies_since(result, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]


def test_an_instruction_uploaded_after_a_successful_dialogue_uses_the_same_question_policy(model_chat):
    first = CASES[0]
    second = CASES[1]
    first_data = instruction([first["caption"]], image_data=png("blue"))
    second_data = instruction([second["caption"]], image_data=png("green"))

    def choose_new_original(payload):
        assert OTHER in payload["catalogue"] and OTHER in payload["candidates"]
        assert payload["catalogue"][OTHER] == second["filename"]
        return match(OTHER, second["caption"])

    chat = model_chat([match(DOC, first["caption"]), choose_new_original],
                      rows=[(DOC, first["filename"], first_data)])
    chat.talk.settings.talk_confirm_understanding = True
    updates = [{
        "facts": [{"key": "application", "value": case["application"], "evidence": case["application"]}],
        "goals": [{"text": case["goal"], "evidence": case["evidence"]}],
        "question": case["question"], "question_field": case["field"],
    } for case in (first, second)]
    chat.model.understand_task = lambda *args: updates.pop(0)
    chat.send(first["opening"])
    chat.send("да", 2)
    resolved = chat.send("Помогло", 3)
    assert resolved["incident"].state == "RESOLVED"
    chat.state["files"].append({"id": OTHER, "filename": second["filename"]})
    chat.state["originals"][OTHER] = second_data
    pending = chat.send(second["opening"], 4)
    assert pending["flow"]["understanding_confirmation"]["source"]["file_id"] == OTHER
    assert all(second["question"] not in text for text, _ in replies_since(resolved, pending))
    accepted = chat.send("Да, правильно", 5)
    photos = [photo for _, photo in replies_since(pending, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(second_data)[0].data]
    assert len(chat.model.calls) == 2


def test_a_missing_coarse_equipment_type_can_still_be_asked_before_search(model_chat):
    question = "Это сканер штрихкодов или сканер документов?"
    chat = model_chat([])
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.understand_task = lambda *args: {
        "facts": [{"key": "equipment_type", "value": "сканер", "evidence": "сканер"}],
        "goals": [{"text": "Подключить сканер", "evidence": "подключить сканер"}],
        "question": question, "question_field": "equipment_type",
    }
    result = chat.send("Нужно подключить сканер")
    assert assistants(result) == [(question, None)]
    assert not chat.model.calls and not result["attempts"]


@pytest.mark.parametrize("resume_previous_task", [False, True])
def test_real_adapter_reference_selection_reaches_durable_original_photo_without_generating_quotes(
    model_chat, resume_previous_task,
):
    opening = "Нужно впервые подключить рабочую папку в программе ПотокДело."
    caption = "Для первого подключения рабочей папки в ПотокДело выберите пункт «Корпоративная папка» в начальном окне."
    data = instruction([caption], image_data=png("green"))
    chat = model_chat([], rows=[(DOC, "Новый запуск ПотокДело.docx", data)])
    chat.talk.settings.talk_confirm_understanding = True
    sequence = 1
    if resume_previous_task:
        legacy_question(
            chat, "Нужно уменьшить громкость считывателя.", "Уменьшить громкость считывателя",
            "уменьшить громкость считывателя", "Какой результат нужен?", application="СтараяПрограмма",
        )
        sequence = 2
    calls = []
    selected = {}

    def handler(request):
        assert request.url.path == "/api/chat/completions"
        body = json.loads(request.content)
        assert body["tools"] == [] and body["tool_ids"] == []
        assert body["features"]["web_search"] is False
        payload = json.loads(body["messages"][-1]["content"])
        calls.append(copy.deepcopy(payload))
        if "message" in payload:
            assert payload["message"] == opening
            response = {
                "new_task": resume_previous_task,
                "facts": [{"key": "application", "value": "ПотокДело", "evidence": "ПотокДело"},
                          {"key": "scope", "value": "application", "evidence": "в программе ПотокДело"}],
                "goals": [{"text": "Первое подключение рабочей папки в ПотокДело",
                           "evidence": "впервые подключить рабочую папку в программе ПотокДело",
                           "goal_id": "g1" if resume_previous_task else None}],
                "question": "Какой протокол доступа использует эта папка — SMB или WebDAV?",
                "question_field": "connection",
            }
        else:
            assert payload["evidence_references"] is True
            assert payload["source_clarification_required"] is True
            assert list(payload["requested_goals"].values()) == ["Первое подключение рабочей папки в ПотокДело"]
            assert "СтараяПрограмма" not in json.dumps(payload, ensure_ascii=False)
            assert "candidate_image_captions" not in payload and "candidate_image_numbers" not in payload
            file_id, registry = next(iter(payload["candidates"].items()))
            evidence_id, proof = next((key, item) for key, item in registry.items()
                                      if item["support_kind"] == "IMAGE" and item["image_numbers"] == [1])
            selected.update(proof)
            goal_id = next(iter(payload["requested_goals"]))
            response = {"result": "MATCH", "file_id": file_id,
                        "covered_goals": [{"goal_id": goal_id, "evidence_id": evidence_id}]}
            assert set(response["covered_goals"][0]) == {"goal_id", "evidence_id"}
            assert caption not in json.dumps(response, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps(response, ensure_ascii=False),
        }, "finish_reason": "stop"}]})

    adapter = OpenWebUILLM(chat.talk.settings, httpx.MockTransport(handler))
    chat.model.understand_task = adapter.understand_task
    chat.model.select_instruction = adapter.select_instruction
    chat.model.supports_source_clarification = True
    chat.model.name = adapter.name
    pending = chat.send(opening, sequence)
    assert len(calls) == 2 and "understanding_confirmation" in pending["flow"]
    assert not pending["attempts"] and not any(photo for _, photo in assistants(pending))
    assert all("SMB" not in text and "WebDAV" not in text and UNAVAILABLE not in text
               for text, _ in assistants(pending))
    accepted = chat.send("Да, правильно", sequence + 1)
    assert len(calls) == 2 and len(accepted["attempts"]) == 1
    attempt = accepted["attempts"][0]
    assert selected["text"] in attempt.evidence["content"]
    assert attempt.evidence["talk_source"]["file_id"] == DOC
    assert attempt.evidence["talk_source"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert attempt.evidence["source"]["file_id"] == DOC
    photos = [photo for _, photo in replies_since(pending, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
