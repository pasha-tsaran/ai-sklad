"""Ambiguous scanner requests ask one useful question before selecting a guide."""

import pytest

from tests.test_talk_dialogue import (
    QUESTION, Knowledge, NoGeneration, approved, assistants, talk as talk,
)


@pytest.mark.parametrize("message", [
    "Настройка сканера ШК", "Нужно настроить сканер штрихкодов", "Помоги настроить проводной сканер ШК",
])
def test_bare_barcode_setup_asks_goal_without_guessing_address_or_sending_photos(talk, message):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(message)
    talk.process(knowledge, model)
    state = talk.snapshot()
    replies = assistants(state)
    assert len(replies) == 1 and replies[0][1] is None
    assert "что" in replies[0][0].casefold() and "настроить" in replies[0][0].casefold()
    assert "Операции" not in replies[0][0] and "АДРЕС 1.0.53" not in replies[0][0]
    assert not state["attempts"] and not knowledge.searches and not model.calls


@pytest.mark.parametrize("message", [
    "Не работает сканер", "Сканер не работает", "Помоги со сканером", "Сканер выдаёт ошибку", "Сканер мигает красным",
])
def test_unspecified_scanner_asks_device_kind_without_unproven_model_names(talk, message):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(message)
    talk.process(knowledge, model)
    state = talk.snapshot()
    reply = assistants(state)[-1]
    assert "штрихкод" in reply[0].casefold() and "документ" in reply[0].casefold()
    assert reply[1] is None and "Zebra" not in reply[0] and "Honeywell" not in reply[0]
    assert not knowledge.searches and not state["attempts"] and not model.calls


@pytest.mark.parametrize("message", [
    "Сканер уже подключен. Мне надо его настроить", "Он уже подключен. Мне надо его настроить",
])
def test_connected_scanner_correction_asks_goal_instead_of_restarting_previous_guide(talk, message):
    knowledge, model = Knowledge([approved(8)]), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    before = talk.snapshot()
    talk.incoming(message, 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    new = assistants(after)[len(assistants(before)):]
    assert len(new) == 1 and new[0][1] is None
    assert "что" in new[0][0].casefold() and "настроить" in new[0][0].casefold()
    assert "Операции" not in new[0][0] and len(after["attempts"]) == 1
    assert len(knowledge.searches) == 1 and not model.calls
    assert "сканер" in after["flow"]["clarification_query"].casefold()
    assert "1С АДРЕС" not in after["flow"]["clarification_query"]


def test_address_answer_to_goal_question_uses_existing_source_photos_without_repeating_clarification(talk):
    knowledge, model = Knowledge([approved(2)]), NoGeneration()
    talk.incoming("Настройка сканера ШК")
    talk.process(knowledge, model)
    before = talk.snapshot()
    talk.incoming("В 1С АДРЕС, драйвер уже установлен", 2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    new = assistants(after)[len(assistants(before)):]
    assert any(photo for _, photo in new) and len(after["attempts"]) == 1
    assert len(knowledge.searches) == 1
    assert "Настройка сканера ШК" in knowledge.searches[-1][0]
    assert "1С АДРЕС" in knowledge.searches[-1][0] and not model.calls


def test_barcode_answer_to_kind_question_moves_to_goal_instead_of_generic_guide(talk):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming("Не работает сканер")
    talk.process(knowledge, model)
    talk.incoming("Штрихкодов", 2)
    talk.process(knowledge, model)
    state = talk.snapshot()
    reply = assistants(state)[-1]
    assert "что" in reply[0].casefold() and ("происходит" in reply[0].casefold() or "не работает" in reply[0].casefold())
    assert not state["attempts"] and not knowledge.searches and not model.calls


def test_fully_specified_address_setup_keeps_existing_instruction_and_photos(talk):
    knowledge, model = Knowledge([approved(2)]), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    state = talk.snapshot()
    assert len(state["attempts"]) == 1 and len(knowledge.searches) == 1
    assert sum(bool(photo) for _, photo in assistants(state)) == 2 and not model.calls


def test_named_volume_task_is_specific_and_does_not_trigger_generic_scanner_questions(talk):
    knowledge, model = Knowledge([approved(1)]), NoGeneration()
    talk.incoming("У проводного сканера ШК нужно уменьшить громкость звукового сигнала")
    talk.process(knowledge, model)
    assert len(knowledge.searches) == 1 and len(talk.snapshot()["attempts"]) == 1
    assert not model.calls


def test_kind_question_asks_type_without_turning_catalogue_models_into_requirements(talk):
    class CatalogKnowledge(Knowledge):
        def scanner_context(self):
            return {"status": "ok", "sources": [
                {"categories": ["barcode"], "models": ["Zebra DS2208"], "applications": ["1С АДРЕС"]},
                {"categories": ["document"], "models": ["HP ScanJet 200"]},
            ], "models": ["Hallucinated model"]}

    knowledge, model = CatalogKnowledge(), NoGeneration()
    talk.incoming("Не работает сканер")
    talk.process(knowledge, model)
    reply = assistants(talk.snapshot())[-1][0]
    assert "сканер штрихкодов" in reply and "сканер документов" in reply
    assert "Zebra DS2208" not in reply and "модел" not in reply.casefold()
    assert "Hallucinated" not in reply and "HP ScanJet" not in reply
    assert not knowledge.searches and not model.calls


def test_goal_question_includes_settings_without_assuming_a_program(talk):
    class CatalogKnowledge(Knowledge):
        def scanner_context(self):
            return {"status": "ok", "sources": [
                {"categories": ["barcode"], "models": [], "applications": ["1С АДРЕС"]},
            ]}

    knowledge, model = CatalogKnowledge(), NoGeneration()
    talk.incoming("Настройка сканера ШК")
    talk.process(knowledge, model)
    reply = assistants(talk.snapshot())[-1][0]
    assert "1С АДРЕС" not in reply and "громкость" in reply and "подсветку" in reply
    assert not knowledge.searches and not model.calls


def test_inventory_error_does_not_guess_model_or_prevent_clarifying_question(talk):
    class CatalogKnowledge(Knowledge):
        def scanner_context(self):
            raise RuntimeError("Unavailable")

    knowledge, model = CatalogKnowledge(), NoGeneration()
    talk.incoming("Не работает сканер")
    talk.process(knowledge, model)
    reply = assistants(talk.snapshot())[-1][0]
    assert "штрихкод" in reply and "документ" in reply and "Zebra" not in reply
    assert not knowledge.searches and not model.calls


def test_model_from_current_inventory_answers_device_kind_without_asking_it_again(talk):
    class CatalogKnowledge(Knowledge):
        def scanner_context(self):
            return {"status": "ok", "sources": [
                {"categories": ["barcode"], "models": ["Zebra DS2208"]},
            ]}

    knowledge, model = CatalogKnowledge(), NoGeneration()
    talk.incoming("Не работает сканер")
    talk.process(knowledge, model)
    talk.incoming("DS2208", 2)
    talk.process(knowledge, model)
    reply = assistants(talk.snapshot())[-1][0]
    assert "Что происходит при сканировании" in reply
    assert "или сканер документов" not in reply
    assert not knowledge.searches and not model.calls


def test_new_printer_question_does_not_inherit_unanswered_scanner_clarification(talk):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming("Настройка сканера ШК")
    talk.process(knowledge, model)
    question = "Принтер не печатает документ PDF"
    talk.incoming(question, 2)
    talk.process(knowledge, model)
    assert knowledge.searches[-1][0] == question
    assert len(talk.snapshot()["attempts"]) == 1 and not model.calls


def test_plain_type_question_needs_no_equipment_catalogue_request(talk):
    class CatalogKnowledge(Knowledge):
        inventories = 0

        def scanner_context(self):
            self.inventories += 1
            return {"status": "ok", "sources": [
                {"categories": ["barcode"], "models": ["Zebra DS2208"]},
            ]}

    knowledge = CatalogKnowledge()
    talk.incoming("Не работает сканер")
    talk.process(knowledge, NoGeneration())
    assert knowledge.inventories == 0
