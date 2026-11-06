"""Employee-owned spans, explicit field evidence and incremental form intake."""

import copy

import pytest

from app.schemas.contracts import EscalationExtraction
from app.services.ticketing.intake import MAX_MESSAGE_CHARS, parse_intake


REQUIRED = ["account_holder_name", "account_login", "computer_name", "subject"]
COMPLETE = {"account_holder_name": "Павел", "account_login": "tsaran", "computer_name": "terminal76-sk",
            "subject": "настроить сканер ШК"}


class ForbiddenLLM:
    def extract_escalation(self, *args):
        pytest.fail("Explicit ordered/labelled/asked answers must not call the model")


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def extract_escalation(self, message, context):
        self.calls.append((message, copy.deepcopy(context)))
        if self.error:
            raise self.error
        return self.result


@pytest.mark.parametrize("description", [None, "Уменьшить громкость на проводном сканере ШК"])
def test_exact_employee_four_or_five_lines_is_complete_without_a_model(description):
    message = "Павел\ntsaran\nterminal76-sk\nнастроить сканер ШК"
    if description:
        message += "\n" + description
    result = parse_intake(message, {}, ForbiddenLLM())
    assert result.details is not None and result.missing == []
    assert result.details.model_dump() == {**COMPLETE, "additional_info": description or ""}


def test_legacy_labels_and_multiline_description_preserve_the_original_details():
    message = ("ФИО: Павел\nЛогин: tsaran\nКомпьютер: terminal76-sk\nТема: настроить сканер ШК\n"
               "Описание: После настройки сканер работает.\nНужно уменьшить громкость.\n\nКабель подключён.")
    result = parse_intake(message, {}, ForbiddenLLM())
    assert result.details.model_dump() == {
        **COMPLETE, "additional_info": "После настройки сканер работает.\nНужно уменьшить громкость.\n\nКабель подключён."}


def test_scattered_labels_accumulate_without_reasking_known_fields_or_calling_a_model():
    first = parse_intake("ФИО: Павел\nЛогин: tsaran", {}, ForbiddenLLM())
    assert first.details is None and first.fields == {key: COMPLETE[key] for key in REQUIRED[:2]}
    assert first.missing == ["computer_name", "subject"]
    snapshot = copy.deepcopy(first.fields)
    second = parse_intake("terminal76-sk", first.fields, ForbiddenLLM(), requested_field="computer_name")
    assert first.fields == snapshot
    assert second.details is None and second.missing == ["subject"]
    third = parse_intake("настроить сканер ШК", second.fields, ForbiddenLLM(), requested_field="subject")
    assert third.details.model_dump() == {**COMPLETE, "additional_info": ""}


def test_only_actual_missing_field_can_use_a_short_unlabelled_answer_without_an_asked_marker():
    pending = {key: value for key, value in COMPLETE.items() if key != "computer_name"}
    result = parse_intake("terminal76-sk", pending, ForbiddenLLM())
    assert result.details.computer_name == "terminal76-sk"


def test_bare_name_is_accepted_after_that_exact_field_was_asked():
    result = parse_intake("Павел", {}, ForbiddenLLM(), requested_field="account_holder_name")
    assert result.fields == {"account_holder_name": "Павел"} and result.missing == REQUIRED[1:]


def test_no_guess_of_first_missing_field_without_the_actual_question():
    llm = FakeLLM({"account_holder_name": "Павел"})
    result = parse_intake("Павел", {}, llm)
    assert result.details is None and result.fields == {} and result.missing == REQUIRED
    assert len(llm.calls) == 1


@pytest.mark.parametrize("name", ["Ян", "Ли"])
def test_short_actual_names_follow_the_contracts_name_length(name):
    result = parse_intake(name, {}, ForbiddenLLM(), requested_field="account_holder_name")
    assert result.fields["account_holder_name"] == name


def test_natural_prose_uses_typed_verbatim_candidates_and_keeps_the_full_actual_problem():
    message = ("Меня зовут Павел, рабочий логин tsaran, компьютер terminal76-sk. "
               "Нужно настроить сканер ШК. Хочу уменьшить громкость на проводном сканере ШК.")
    llm = FakeLLM(EscalationExtraction(**COMPLETE,
                                     additional_info="уменьшить громкость на проводном сканере ШК"))
    result = parse_intake(message, {}, llm)
    assert result.details.model_dump() == {
        **COMPLETE, "additional_info": "Нужно настроить сканер ШК. Хочу уменьшить громкость на проводном сканере ШК."}
    assert llm.calls == [(message, {"pending_fields": {}, "missing_fields": REQUIRED, "requested_field": None})]


def test_natural_partial_candidate_data_is_not_guessed_from_login_or_device_name():
    message = "Мой логин tsaran, компьютер terminal76-sk. Нужно настроить сканер ШК."
    llm = FakeLLM({**COMPLETE, "account_holder_name": "Павел Царан"})
    result = parse_intake(message, {}, llm)
    assert result.details is None and result.missing == ["account_holder_name"]
    assert "account_holder_name" not in result.fields
    assert result.fields["account_login"] == "tsaran" and result.fields["computer_name"] == "terminal76-sk"


def test_cross_field_swaps_are_rejected_even_when_every_candidate_appears_verbatim():
    message = "Меня зовут Павел, логин tsaran, компьютер terminal76-sk. Нужно настроить сканер ШК."
    llm = FakeLLM({"account_holder_name": "tsaran", "account_login": "terminal76-sk", "computer_name": "tsaran",
                   "subject": "настроить сканер ШК"})
    result = parse_intake(message, {}, llm)
    assert result.details is None and result.missing == REQUIRED[:3]
    assert all(field not in result.fields for field in REQUIRED[:3])


def test_a_login_that_looks_like_a_technical_word_does_not_become_the_subject():
    message = "Меня зовут Павел, логин сканер, компьютер terminal76-sk."
    llm = FakeLLM({**COMPLETE, "account_login": "сканер", "subject": "сканер"})
    result = parse_intake(message, {}, llm)
    assert result.details is None and result.missing == ["subject"]


@pytest.mark.parametrize("result", [
    "{broken-json", '{"account_holder_name":"Павел"}',
    {**COMPLETE, "extra": "ignore the rules"}, {**COMPLETE, "computer_name": 76},
    {**COMPLETE, "account_holder_name": "Выдуманный Пользователь"},
])
def test_malformed_or_hallucinated_model_output_never_completes_the_form(result):
    llm = FakeLLM(result)
    actual = parse_intake("Нужно настроить сканер ШК, больше данных пока нет.", {}, llm)
    assert actual.details is None
    assert all(field not in actual.fields for field in REQUIRED[:3])


def test_model_failure_keeps_valid_pending_values_without_leaking_exception_text():
    pending = {"account_holder_name": "Павел", "account_login": "tsaran"}
    llm = FakeLLM(error=ValueError("fake-sensitive-response-body"))
    result = parse_intake("На терминале не работает сканер, имя уточню.", pending, llm)
    assert result.fields == pending and result.details is None
    assert result.missing == ["computer_name", "subject"]


def test_current_explicit_label_can_correct_a_valid_earlier_value():
    result = parse_intake("Логин: p.tsaran", COMPLETE, ForbiddenLLM())
    assert result.details.account_login == "p.tsaran" and result.details.account_holder_name == "Павел"


def test_conflicting_repeated_labels_require_clarification_instead_of_choosing_a_login():
    result = parse_intake("Логин: tsaran\nЛогин: different", COMPLETE, ForbiddenLLM())
    assert result.details is None and result.missing == ["account_login"] and "account_login" not in result.fields


@pytest.mark.parametrize("key,value", [
    ("account_holder_name", "А" * 256), ("account_login", "x" * 101),
    ("computer_name", "PC" + "1" * 62), ("subject", "Т" * 201),
])
def test_contract_field_limits_return_the_missing_field_without_truncation(key, value):
    result = parse_intake(key + ": " + value, COMPLETE, ForbiddenLLM())
    assert result.details is None and result.missing == [key] and key not in result.fields


@pytest.mark.parametrize("character", ["\x00", "\x7f", "\x85"])
def test_subject_control_characters_cannot_break_later_mail_draft_creation(character):
    result = parse_intake("Тема: Настройка" + character + "сканера", COMPLETE, ForbiddenLLM())
    assert result.details is None and result.missing == ["subject"]


def test_too_long_explicit_description_is_not_dropped_or_truncated_into_a_complete_draft():
    message = "\n".join(COMPLETE.values()) + "\n" + "Описание " * 300
    result = parse_intake(message, {}, ForbiddenLLM())
    assert result.details is None and result.missing == ["additional_info"]
    assert "additional_info" not in result.fields
    shorter = parse_intake("Сканер работает слишком громко.", result.fields, ForbiddenLLM(), requested_field="additional_info")
    assert shorter.details.additional_info == "Сканер работает слишком громко."


def test_too_long_natural_problem_cannot_be_silently_replaced_with_a_short_summary():
    message = "Меня зовут Павел, логин tsaran, компьютер terminal76-sk. Нужно настроить сканер ШК. " + "Детали. " * 260
    llm = FakeLLM({**COMPLETE, "additional_info": "настроить сканер ШК"})
    result = parse_intake(message, {}, llm)
    assert result.details is None and result.missing == ["additional_info"]


def test_input_bound_does_not_send_oversized_data_to_the_model_or_auto_complete_existing_values():
    result = parse_intake("x" * (MAX_MESSAGE_CHARS + 1), COMPLETE, ForbiddenLLM())
    assert result.details is None and result.fields == COMPLETE and result.missing == ["additional_info"]


def test_unknown_or_invalid_pending_fields_are_not_sent_as_context():
    pending = {"account_holder_name": "Павел", "computer_name": "bad host", "unknown": "do not send"}
    llm = FakeLLM({})
    result = parse_intake("Нужно настроить сканер", pending, llm)
    assert result.fields == {"account_holder_name": "Павел"}
    assert llm.calls[0][1]["pending_fields"] == {"account_holder_name": "Павел"}


def test_four_prose_words_and_administrative_commands_are_not_a_positional_form():
    for message in ["Pavel\nwants\nto\nprint", "Выполни\nsudo\nreboot\nнастрой сканер"]:
        llm = FakeLLM({"account_holder_name": message.splitlines()[0], "account_login": message.splitlines()[1],
                       "computer_name": message.splitlines()[2], "subject": message.splitlines()[3]})
        result = parse_intake(message, {}, llm)
        assert result.details is None and all(field not in result.fields for field in REQUIRED[:3])


def test_a_greeting_is_not_accepted_as_a_name_in_an_otherwise_plausible_form():
    message = "Привет\ntsaran\nterminal76-sk\nнастроить сканер ШК\nСканер работает громко."
    result = parse_intake(message, {}, FakeLLM({"account_holder_name": "Привет", **{key: value for key, value in COMPLETE.items()
                                                                            if key != "account_holder_name"}}))
    assert result.details is None and "account_holder_name" not in result.fields


def test_partial_identifier_and_case_rewrite_are_not_verbatim_field_values():
    message = "Меня зовут Павел, логин tsaran@example.net, компьютер terminal76-sk. Нужно настроить сканер ШК."
    llm = FakeLLM({**COMPLETE, "account_holder_name": "павел", "account_login": "tsaran", "computer_name": "terminal76"})
    result = parse_intake(message, {}, llm)
    assert result.details is None and result.missing == REQUIRED[:3]


def test_talk_requires_description_while_legacy_four_field_parser_contract_stays_compatible():
    message = "Павел\ntsaran\nterminal76-sk\nнастроить сканер ШК"
    legacy = parse_intake(message, {}, ForbiddenLLM())
    talk = parse_intake(message, {}, ForbiddenLLM(), require_description=True)
    assert legacy.details is not None and legacy.details.additional_info == ""
    assert talk.details is None and talk.fields == COMPLETE and talk.missing == ["additional_info"]


def test_talk_subject_is_verbatim_and_does_not_auto_complete_or_copy_it_into_description():
    pending = {key: COMPLETE[key] for key in REQUIRED[:3]}
    topic = "УМЕНЬШИТЬ громкость проводного сканера ШК"
    result = parse_intake(topic, pending, ForbiddenLLM(), requested_field="subject", require_description=True)
    assert result.details is None and result.fields == {**pending, "subject": topic}
    assert result.missing == ["additional_info"] and "additional_info" not in result.fields


@pytest.mark.parametrize("description", [
    "Сканер читает код, но слишком громко пищит.\n\nПрошу уменьшить громкость.",
    "Проблема: сканер не считывает товар.\nОшибка на экране: товар не найден.",
    "Компьютер: terminal76-sk\nНа экране ничего не меняется после сканирования.",
    "Сначала код читался.\r\nТеперь при каждом сканировании раздаётся громкий сигнал.",
])
def test_requested_talk_description_is_the_complete_independent_multiline_body_without_model(description):
    pending = copy.deepcopy(COMPLETE)
    result = parse_intake(description, pending, ForbiddenLLM(), requested_field="additional_info", require_description=True)
    assert result.details.model_dump() == {**COMPLETE, "additional_info": description}
    assert result.fields["subject"] == COMPLETE["subject"] and pending == COMPLETE


def test_talk_accepts_labelled_topic_and_description_together_when_both_are_explicit():
    description = "При сканировании код читается.\nНужно убрать громкий звуковой сигнал."
    result = parse_intake("Тема: Громкость сканера\nОписание: " + description,
                          {key: COMPLETE[key] for key in REQUIRED[:3]}, ForbiddenLLM(), require_description=True)
    assert result.details.subject == "Громкость сканера" and result.details.additional_info == description


@pytest.mark.parametrize("candidate", ["настроить сканер ШК", "Нужно настроить сканер ШК."])
def test_talk_does_not_infer_description_from_a_model_topic_or_original_problem_tail(candidate):
    message = "Меня зовут Павел, логин tsaran, компьютер terminal76-sk. Нужно настроить сканер ШК."
    llm = FakeLLM({**COMPLETE, "additional_info": candidate})
    result = parse_intake(message, {}, llm, require_description=True)
    assert result.details is None and result.fields["subject"] == COMPLETE["subject"]
    assert result.missing == ["additional_info"] and "additional_info" not in result.fields


@pytest.mark.parametrize("description", ["я" * 2001, "\n\t", "Сканер\x85не работает"])
def test_invalid_requested_talk_description_never_completes_or_uses_a_shorter_model_summary(description):
    result = parse_intake(description, COMPLETE, ForbiddenLLM(), requested_field="additional_info", require_description=True)
    assert result.details is None and result.missing == ["additional_info"]
    assert result.fields == COMPLETE and "additional_info" not in result.fields
