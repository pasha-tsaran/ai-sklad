"""Device type and requested task guide retrieval without a model questionnaire."""

from xml.sax.saxutils import escape

import pytest

from app.schemas.contracts import AnalysisResult
from app.services.chat.talk_dialogue import CLARIFY_QUESTIONS
from app.services.llm.mock import MockLLM
from app.services.rag.query import printer_kinds, retrieval_query, source_compatible
from tests.test_openwebui_instruction_images import instruction
from tests.test_talk_dialogue import QUESTION, Knowledge, NoGeneration, approved, assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC, provider


def test_actual_barcode_setup_then_beam_and_volume_uses_instruction37_original_photo(talk):
    caption = "Сканер штрихкодов Zebra DS2208. Штрихкоды настройки громкости и яркости подсветки."
    body = ("Настройка сканера штрихкодов Zebra DS2208. "
            "Громкость и подсветка выбираются соответствующим штрихкодом на листе.")
    source = instruction([caption], extra_paragraph="<w:p><w:r><w:t>" + escape(body) + "</w:t></w:r></w:p>")
    knowledge, requests = provider(filename="ИТ_37_Zebra DS2208 — лист настроечных штрихкодов.docx",
                                   data=source, chunks=[(DOC, body)])
    model = NoGeneration()
    talk.incoming("настроить сканер ШК")
    assert talk.process(knowledge, model)
    first = talk.snapshot()
    assert not first["attempts"] and "модел" not in assistants(first)[-1][0].casefold()
    talk.incoming("настроить луч сканера и громкость", 2)
    assert talk.process(knowledge, model)
    result = talk.snapshot()
    assert len(result["attempts"]) == 1
    assert result["flow"]["source"]["filename"].startswith("ИТ_37")
    assert any(photo for _, photo in assistants(result))
    assert "луч сканера и громкость" in result["flow"]["query"]
    assert not model.calls
    assert any(request.method == "POST" for request in requests["calls"])


@pytest.mark.parametrize("message", ["Не работает сканер", "Настройка сканера", "Сканер выдаёт ошибку"])
def test_scanner_kind_question_offers_types_without_model_request(talk, message):
    knowledge = Knowledge()
    talk.incoming(message)
    talk.process(knowledge, NoGeneration())
    text = assistants(talk.snapshot())[-1][0]
    assert "штрихкод" in text and "документ" in text and "планшетный" in text
    assert "модел" not in text.casefold() and not knowledge.searches


@pytest.mark.parametrize("message", ["Не работает принтер", "Принтер не печатает", "Помоги настроить принтер"])
def test_unknown_printer_kind_asks_one_type_question_before_search(talk, message):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(message)
    talk.process(knowledge, model)
    text = assistants(talk.snapshot())[-1][0]
    assert "А4" in text and "стационарный" in text and "мобильный" in text
    assert text.count("?") == 1 and "модел" not in text.casefold()
    assert not knowledge.searches and not model.calls


def test_thermal_printer_question_distinguishes_stationary_and_mobile(talk):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming("Термопринтер не печатает этикетки")
    talk.process(knowledge, model)
    text = assistants(talk.snapshot())[-1][0]
    assert "стационарный" in text and "мобильный" in text and "А4" not in text
    assert not knowledge.searches and not model.calls


@pytest.mark.parametrize("answer,kind", [("Мобильный", "мобильный термопринтер"),
                                        ("Стационарный", "стационарный термопринтер"),
                                        ("А4", "принтер документов а4")])
def test_short_printer_type_answer_preserves_actual_problem_for_retrieval(talk, answer, kind):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming("Принтер не печатает")
    talk.process(knowledge, model)
    talk.incoming(answer, 2)
    talk.process(knowledge, model)
    assert "Принтер не печатает" in knowledge.searches[-1][0]
    assert kind in knowledge.searches[-1][0].casefold()
    assert not model.calls


@pytest.mark.parametrize("key", ["DEVICE_MODEL", "DEVICE_TYPE"])
def test_legacy_analysis_cannot_ask_model_after_barcode_type_and_goal_are_known(talk, key):
    class Legacy(MockLLM):
        def analyze_context(self, message, context):
            return AnalysisResult(intent="REPORT_PROBLEM", search_query=message, clarification_key=key)

    knowledge = Knowledge()
    knowledge.missing = True
    talk.incoming("У сканера штрихкодов нужно уменьшить громкость")
    talk.process(knowledge, Legacy())
    replies = [text for text, _ in assistants(talk.snapshot())]
    assert not any("модел" in text.casefold() for text in replies)
    assert CLARIFY_QUESTIONS["DEVICE_TYPE"] not in replies


def test_wrong_guide_is_excluded_then_new_task_uses_another_source_without_negative_feedback(talk):
    first, second = approved(1), approved(1, "00000000-0000-4000-8000-000000000077")
    knowledge, model = Knowledge([first]), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    knowledge.documents.append(second)
    talk.incoming("Это не та инструкция, мне нужно уменьшить громкость сканера", 2)
    talk.process(knowledge, model)
    latest = knowledge.searches[-1]
    assert first.document["source"]["file_id"] in latest[1]
    assert "громкость" in latest[0] and "1С АДРЕС" not in latest[0]
    assert talk.snapshot()["flow"]["source"]["file_id"] == second.document["source"]["file_id"]
    assert talk.snapshot()["incident"].failed_attempts == 0
    assert not model.calls


@pytest.mark.parametrize("problem,source", [
    ("Подключить мобильный термопринтер", "Подключение стационарного термопринтера по USB."),
    ("Настроить стационарный термопринтер", "Настройка мобильного термопринтера по Bluetooth."),
    ("Принтер документов А4 не печатает", "Мобильный принтер этикеток не печатает."),
    ("Нужно настроить луч сканера ШК", "Подключение сканера штрихкодов в 1С."),
])
def test_device_type_and_task_guards_reject_an_explicitly_different_procedure(problem, source):
    assert not source_compatible(problem, source, complete=True)


def test_unknown_printer_source_type_can_still_be_semantically_checked():
    assert source_compatible("Стационарный термопринтер нуждается в калибровке", "Калибровка принтера TSC TE310.", complete=True)
    assert printer_kinds("Стационарный термопринтер") == {"thermal_stationary"}
    assert printer_kinds("мобильный термопринтер") == {"thermal_mobile"}
    assert printer_kinds("Принтер для документов А4") == {"document"}
    assert "подсветка" in retrieval_query("Настроить луч сканера ШК")


def test_thermal_printer_still_requires_printer_evidence_and_cannot_become_a_mobile_scanner():
    assert not source_compatible("Подключить мобильный термопринтер", "Подключение мобильного сканера штрихкодов.", complete=True)
    assert not source_compatible("Термопринтер печатает полосами", "Очистка очереди принтера при отсутствии печати.", complete=True)
    assert not source_compatible("Зажевало бумагу в принтере", "Очистка очереди принтера при отсутствии печати.", complete=True)
