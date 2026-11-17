"""An error and a second short message stay attached to the current instruction."""

import copy

import pytest

from app.services.chat.talk_dialogue import UNAVAILABLE, _model_clarification
from app.services.chat.talk_dialogue import process_one
from app.services.rag.talk import KnowledgeUnavailable
from tests.test_openwebui_instruction_images import instruction, png
from tests.test_talk_dialogue import assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC
from tests.test_talk_model_dialogue import decision
from tests.test_talk_model_dialogue import model_chat as model_chat


GOAL = "Загрузить файл в Nextcloud"
OPENING = "Нужно загрузить файл в Nextcloud."
EVIDENCE = "На четвёртом шаге подключения загрузчика используйте пароль приложения из настроек безопасности."
GENERIC = "Что именно нужно сделать и с каким типом оборудования или программой?"
ERROR_QUESTION = "Какой текст ошибки появляется на четвёртом шаге?"
WINDOW_QUESTION = "В каком окне появляется сообщение «Неверный пароль»?"


@pytest.mark.parametrize("question,report", [
    (WINDOW_QUESTION, "неверный пароль"),
    ('В каком приложении появляется ошибка "Invalid password"?', "Ошибка Invalid password"),
])
def test_clarification_can_quote_a_password_error_already_reported_by_the_employee(question, report):
    assert _model_clarification(question, reported_messages=[report]) == question


@pytest.mark.parametrize("question,reports", [
    (WINDOW_QUESTION, ["Файл не загружается"]),
    (WINDOW_QUESTION, []),
    ("Какой пароль вы вводите?", ["неверный пароль"]),
    ("Пришлите пароль приложения?", ["неверный пароль"]),
    ("Какой пароль вы вводите после сообщения «Неверный пароль»?", ["неверный пароль"]),
    ("В каком окне вы вводите «пароль»?", ["пароль"]),
])
def test_quoted_error_context_never_authorizes_a_credential_request(question, reports):
    with pytest.raises(KnowledgeUnavailable):
        _model_clarification(question, reported_messages=reports)


def new_replies(previous, current):
    return assistants(current)[len(assistants(previous)):]


def upload_chat(model_chat, updates, *, selection_replies=()):
    document = instruction([
        "Откройте программу загрузки файлов Nextcloud.",
        "Выберите файл для загрузки в корпоративную папку.",
        "Проверьте адрес сервера в окне подключения программы.",
        EVIDENCE,
    ], image_data=png("blue"))
    match = {
        "result": "MATCH", "file_id": DOC,
        "covered_goals": [{"goal_id": "g1", "evidence": EVIDENCE,
                           "support_kind": "IMAGE", "image_numbers": [4]}],
    }
    chat = model_chat([match, *selection_replies], rows=[
        (DOC, "Загрузка файла в Nextcloud.docx", document),
    ])
    chat.talk.settings.talk_confirm_understanding = True
    responses = [{
        "facts": [{"key": "application", "value": "Nextcloud", "evidence": "Nextcloud"}],
        "goals": [{"text": GOAL, "evidence": "загрузить файл в Nextcloud"}],
    }, *updates]
    calls = []

    def understand(message, previous, previous_question):
        calls.append({"message": message, "previous": copy.deepcopy(previous),
                      "previous_question": previous_question})
        assert responses, "Unexpected task-understanding call"
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    chat.model.understand_task = understand
    pending = chat.send(OPENING)
    assert not pending["attempts"]
    guide = chat.send("да, правильно", 2)
    assert guide["flow"]["source"]["file_id"] == DOC
    assert len(guide["attempts"]) == 1
    return chat, guide, calls


def test_asking_where_a_password_error_appears_does_not_request_the_password(model_chat):
    chat, guide, _ = upload_chat(model_chat, [
        {"question": ERROR_QUESTION, "question_field": "symptoms"},
        {"question": WINDOW_QUESTION, "question_field": "other"},
    ])
    problem = chat.send("На 4 шаге выдает ошибку", 3)
    assert new_replies(guide, problem) == [(ERROR_QUESTION, None)]
    result = chat.send("неверный пароль", 4)
    assert new_replies(problem, result) == [(WINDOW_QUESTION, None)]
    assert result["flow"]["source"] == guide["flow"]["source"]
    assert result["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]
    assert len(result["attempts"]) == 1 and result["incident"].failed_attempts == 0


def test_error_text_and_supplemental_browser_login_preserve_the_active_upload_source(model_chat):
    browser_question = "В каком приложении возникает ошибка при загрузке файла?"
    chat, guide, calls = upload_chat(model_chat, [
        {"question": ERROR_QUESTION, "question_field": "symptoms"},
        {},
        {"facts": [{"key": "completed", "value": "Вход через браузер выполнен",
                    "evidence": "в браузере зашел нормально"}],
         "question": browser_question, "question_field": "other"},
    ], selection_replies=[decision("CLARIFY", question=WINDOW_QUESTION)])
    asked = chat.send("На 4 шаге выдает ошибку", 3)
    error = chat.send("неверный пароль", 4)
    assert all(GENERIC not in text and UNAVAILABLE not in text for text, _ in new_replies(asked, error))
    supplement = chat.send("причем в браузере зашел нормально", 5)
    assert new_replies(error, supplement) == [(browser_question, None)]
    assert supplement["flow"]["source"] == guide["flow"]["source"]
    state = supplement["flow"]["task_state"]
    assert state["goals"] == guide["flow"]["task_state"]["goals"]
    assert "неверный пароль" in calls[-1]["previous"]["messages"]
    assert any(item["key"] == "completed" and "браузер" in item["value"] for item in state["facts"])
    assert len(supplement["attempts"]) == 1 and supplement["incident"].failed_attempts == 0


def test_a_reminder_about_the_same_task_cannot_restart_generic_equipment_intake(model_chat):
    chat, guide, _ = upload_chat(model_chat, [
        {"question": ERROR_QUESTION, "question_field": "symptoms"},
        {"question": GENERIC, "question_field": "goal"},
    ], selection_replies=[decision("CLARIFY", question=ERROR_QUESTION)])
    asked = chat.send("На 4 шаге выдает ошибку", 3)
    result = chat.send("Я же писал уже тебе что нужно сделать, все то же самое", 4)
    replies = new_replies(asked, result)
    assert replies and all(GENERIC not in text and UNAVAILABLE not in text for text, _ in replies)
    assert result["flow"]["source"] == guide["flow"]["source"]
    assert result["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]
    assert len(result["attempts"]) == 1 and result["incident"].failed_attempts == 0


def test_two_queued_error_messages_keep_the_original_task_and_active_instruction(model_chat):
    browser_question = "В каком приложении возникает ошибка при загрузке файла?"
    chat, guide, calls = upload_chat(model_chat, [
        {"question": ERROR_QUESTION, "question_field": "symptoms"},
        {},
        {"facts": [{"key": "completed", "value": "Вход через браузер выполнен",
                    "evidence": "в браузере зашел нормально"}],
         "question": browser_question, "question_field": "other"},
    ], selection_replies=[decision("CLARIFY", question=WINDOW_QUESTION)])
    chat.send("На 4 шаге выдает ошибку", 3)
    chat.talk.incoming("неверный пароль", 4)
    chat.talk.incoming("причем в браузере зашел нормально", 5, previous_delivered=False)
    for _ in range(2):
        assert process_one(chat.talk.env.factory, chat.talk.settings, chat.model,
                           account_id=chat.talk.account_id)
    result = chat.talk.snapshot()
    assert result["flow"]["source"] == guide["flow"]["source"]
    assert result["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]
    assert calls[-1]["message"] == "причем в браузере зашел нормально"
    assert "неверный пароль" in calls[-1]["previous"]["messages"]
    assert new_replies(guide, result)[-1] == (browser_question, None)
    assert len(result["attempts"]) == 1 and result["incident"].failed_attempts == 0


def test_reminder_with_an_explicit_new_task_still_reaches_task_understanding(model_chat):
    message = "Я уже писал про загрузку, теперь нужно настроить принтер"
    question = "Что нужно изменить в настройках принтера?"
    chat, guide, calls = upload_chat(model_chat, [{
        "new_task": True,
        "facts": [{"key": "equipment_type", "value": "принтер", "evidence": "принтер"}],
        "goals": [],
        "question": question, "question_field": "goal",
    }])
    result = chat.send(message, 3)
    assert calls[-1]["message"] == message
    assert new_replies(guide, result) == [(question, None)]
    assert "source" not in result["flow"]
    assert GOAL not in [item["text"] for item in result["flow"]["task_state"]["goals"].values()]


def test_reminder_with_a_new_step_keeps_the_supplied_condition(model_chat):
    message = "Все то же самое, но ошибка теперь на пятом шаге"
    question = "Какой текст ошибки появляется на пятом шаге?"
    chat, guide, calls = upload_chat(model_chat, [{
        "facts": [{"key": "current_step", "value": "пятый шаг", "evidence": "пятом шаге"}],
        "question": question, "question_field": "symptoms",
    }])
    result = chat.send(message, 3)
    assert calls[-1]["message"] == message
    assert new_replies(guide, result) == [(question, None)]
    assert result["flow"]["source"] == guide["flow"]["source"]
    assert any(item["key"] == "current_step" and item["value"] == "пятый шаг"
               for item in result["flow"]["task_state"]["facts"])


def test_task_understanding_timeout_does_not_erase_an_error_before_the_next_short_message(model_chat):
    browser_question = "В каком приложении возникает ошибка при загрузке файла?"
    chat, guide, calls = upload_chat(model_chat, [
        {"question": ERROR_QUESTION, "question_field": "symptoms"},
        TimeoutError("controlled temporary failure"),
        {"facts": [{"key": "completed", "value": "Вход через браузер выполнен",
                    "evidence": "в браузере зашел нормально"}],
         "question": browser_question, "question_field": "other"},
    ])
    chat.send("На 4 шаге выдает ошибку", 3)
    failed = chat.send("неверный пароль", 4)
    assert failed["flow"]["source"] == guide["flow"]["source"]
    result = chat.send("причем в браузере зашел нормально", 5)
    assert "неверный пароль" in calls[-1]["previous"]["messages"]
    assert new_replies(failed, result) == [(browser_question, None)]
    assert result["flow"]["source"] == guide["flow"]["source"]
    assert result["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]


def test_forbidden_task_question_keeps_validated_error_context_and_uses_a_safe_question(model_chat):
    forbidden = "Какой пароль вы вводите?"
    chat, guide, _ = upload_chat(model_chat, [{
        "facts": [{"key": "error_text", "value": "неверный пароль", "evidence": "неверный пароль"}],
        "question": forbidden, "question_field": "other",
    }])
    result = chat.send("На четвёртом шаге ошибка: неверный пароль", 3)
    replies = new_replies(guide, result)
    assert replies and all(forbidden not in text and UNAVAILABLE not in text for text, _ in replies)
    assert result["flow"]["source"] == guide["flow"]["source"]
    assert any(item["key"] == "error_text" and item["value"] == "неверный пароль"
               for item in result["flow"]["task_state"]["facts"])
    assert len(result["attempts"]) == 1 and result["incident"].failed_attempts == 0


def test_generic_selection_question_repaired_to_the_previous_question_is_retried_once(model_chat):
    contextual = "В каком окне появляется это сообщение об ошибке?"
    supplement_question = "Какой текст показан в заголовке окна с ошибкой?"
    chat, guide, calls = upload_chat(model_chat, [
        {"facts": [{"key": "error_text", "value": "неверный пароль", "evidence": "неверный пароль"}],
         "question": contextual, "question_field": "other"},
        {"facts": [{"key": "application", "value": "программа загрузки Nextcloud",
                    "evidence": "программа загрузки Nextcloud"}]},
        {"facts": [{"key": "completed", "value": "Вход через браузер выполнен",
                    "evidence": "в браузере зашел нормально"}],
         "question": supplement_question, "question_field": "other"},
    ], selection_replies=[
        decision("CLARIFY", question=GENERIC),
        {"result": "NO_KNOWLEDGE", "missing_goal_ids": ["g1"]},
    ])
    asked = chat.send("На четвёртом шаге ошибка: неверный пароль", 3)
    assert new_replies(guide, asked) == [(contextual, None)]
    result = chat.send("Это программа загрузки Nextcloud", 4)
    replies = new_replies(asked, result)
    assert replies and all(contextual not in text and GENERIC not in text and UNAVAILABLE not in text
                           for text, _ in replies)
    assert len(chat.model.calls) == 3
    assert chat.model.calls[-1]["payload"]["allow_clarification"] is False
    assert result["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]
    assert any(item["key"] == "error_text" and item["value"] == "неверный пароль"
               for item in result["flow"]["task_state"]["facts"])
    assert len(result["attempts"]) == 1 and result["incident"].failed_attempts == 0
    assert result["flow"]["escalation"]["stage"] == "CHOICE"
    continued = chat.send("причем в браузере зашел нормально", 5)
    previous = calls[-1]["previous"]
    assert previous["goals"] == guide["flow"]["task_state"]["goals"]
    assert any(item["key"] == "error_text" and item["value"] == "неверный пароль"
               for item in previous["facts"])
    assert "intake_fields" not in previous and "escalation" not in previous
    assert new_replies(result, continued) == [(supplement_question, None)]
    assert continued["flow"]["source"] == guide["flow"]["source"]
    assert continued["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]
    assert "escalation" not in continued["flow"]
    assert continued["incident"].pending_action is None
    assert len(continued["attempts"]) == 1 and continued["incident"].failed_attempts == 0
