"""Complete original reading reaches the canonical planner and current source pixels."""

import hashlib
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest

from app.services.chat.instruction_plan import render_plan, restore_plan, serialize_plan
from app.services.rag.talk import KnowledgeUnavailable
from app.services.llm.instruction_discovery import InstructionNomination
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_instruction_planning import selected_part
from tests.test_instruction_reader import model_reader, wire_evidence, wire_goal
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, OTHER


THIRD = "00000000-0000-4000-8000-000000000088"


def read_context(goals, kinds=None):
    return {"instruction_planning": True, "requested_goals": goals,
            "requested_goal_kinds": kinds or {identity: "LOOKUP" for identity in goals}}


def configure_reader(rows, responder, monkeypatch, order=None):
    knowledge, state = scanner_provider(rows)
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    model, calls = model_reader(responder)
    knowledge.reader = model.read_instruction
    if order is not None:
        monkeypatch.setattr(knowledge, "_instruction_ranked_sources", lambda *args, **kwargs: list(order))
    return knowledge, state, calls


def answer(wire, supported=None, *, status="SUPPORTED", mode=None):
    supported = supported or {}
    goals = {}
    for goal_id in wire["requested_goals"]:
        if goal_id not in supported:
            goals[goal_id] = wire_goal("MISSING")
            continue
        needles = supported[goal_id]
        needles = [needles] if isinstance(needles, str) else needles
        references = list(dict.fromkeys(identity for needle in needles for identity in wire_evidence(wire, needle)))
        assert references
        goals[goal_id] = wire_goal(status, references)
        if status == "SUPPORTED":
            goals[goal_id]["delivery_mode"] = mode or (
                "PROCEDURE" if wire["requested_goal_kinds"].get(goal_id) == "ACTION" else "EXCERPT")
    return {"goals": goals}


def test_reader_receives_original_sections_beyond_the_old_eight_fragment_window(monkeypatch):
    paragraphs = [f"Уникальная исходная справочная строка номер {number}." for number in range(24)]
    target = "Контрольное значение: 170 мм."
    original = "\n".join(["# Полный новый справочник", *paragraphs, target]).encode()
    knowledge, _, calls = configure_reader([(DOC, "Новый справочник.md", original)],
                                           lambda wire, count: answer(wire, {"g1": target}), monkeypatch)
    result = knowledge.search_with_llm("Какое контрольное значение?", read_context({"g1": "Узнать контрольное значение"}))
    wire = calls[0]["wire"]
    assert sum(len(section["evidence"]) for section in wire["approved_original"]["sections"]) > 8
    assert all(any(wire_evidence(wire, text)) for text in paragraphs)
    assert target in "\n".join(section["text"] for section in result.plan[0].sections)
    assert result.approved.document["source"]["sha256"] == hashlib.sha256(original).hexdigest()


def test_one_goal_for_width_and_height_retains_both_exact_original_fact_sections(monkeypatch):
    width, height = "Ширина этикетки: 70 мм.", "Высота этикетки: 40 мм."
    original = ("# Значения по умолчанию\n" + width + "\n" + height).encode()
    knowledge, _, _ = configure_reader([(DOC, "Параметры.md", original)],
                                       lambda wire, count: answer(wire, {"g1": [width, height]}), monkeypatch)
    result = knowledge.search_with_llm("Какие ширина и высота?", read_context({"g1": "Узнать ширину и высоту"}))
    text = "\n".join(section["text"] for section in result.plan[0].sections)
    assert width in text and height in text and result.plan[0].delivery_mode == "EXCERPT"
    saved = serialize_plan(result.plan)
    restored = restore_plan(knowledge, saved)
    blocks, _, _, more = render_plan(restored, saved)
    assert not more and width in blocks[0][0] and height in blocks[0][0]


@pytest.mark.parametrize("verified_later", [True, False])
def test_reader_timeout_allows_a_later_verified_original_but_never_proves_absence(monkeypatch, verified_later):
    target = "Контрольное значение: 170 мм."

    def read(wire, count):
        if count == 1:
            raise httpx.ReadTimeout("Optional candidate did not finish")
        return answer(wire, {"g1": target} if verified_later else {})

    knowledge, _, calls = configure_reader([(DOC, "Первый кандидат.md", b"# First\nOther topic."),
                                           (OTHER, "Справочная таблица.md", target.encode())],
                                          read, monkeypatch, order=[DOC, OTHER])
    context = read_context({"g1": "Узнать контрольное значение"})
    if verified_later:
        decision = knowledge.search_with_llm("Какое контрольное значение?", context)
        assert decision.approved.document["source"]["file_id"] == OTHER
    else:
        with pytest.raises(KnowledgeUnavailable, match="RequestTimedOut"):
            knowledge.search_with_llm("Какое контрольное значение?", context)
    assert len(calls) == 2


def test_multiple_originals_preserve_employee_goal_order_and_explicit_missing_coverage(monkeypatch):
    width = "Ширина устройства Альфа: 70 мм."
    check = "1. Откройте окно проверки устройства Бета."

    def read(wire, count):
        if wire["approved_original"]["filename"] == "Проверка Бета.md":
            return answer(wire, {"g2": check})
        return answer(wire, {"g1": width})

    knowledge, _, calls = configure_reader([(DOC, "Проверка Бета.md", check.encode()),
                                           (OTHER, "Справка Альфа.md", width.encode())],
                                          read, monkeypatch, order=[DOC, OTHER])
    goals = {"g1": "Узнать ширину Альфа", "g2": "Проверить Бета", "g3": "Восстановить неизвестную функцию"}
    result = knowledge.search_with_llm("Ширина Альфа, проверка Бета и другая функция.",
                                      read_context(goals, {"g1": "LOOKUP", "g2": "ACTION", "g3": "ACTION"}))
    assert len(calls) == 2 and result.result == "PARTIAL"
    assert [part.goal_ids for part in result.plan] == [("g1",), ("g2",)]
    assert [part.approved.document["source"]["file_id"] for part in result.plan] == [OTHER, DOC]
    assert result.missing_goals == (goals["g3"],)
    assert [part.delivery_mode for part in result.plan] == ["EXCERPT", "PROCEDURE"]


def test_related_first_original_does_not_win_before_a_later_exact_answer(monkeypatch):
    bluetooth = "Эта инструкция предназначена для подключения по Bluetooth."
    wifi = "Для подключения по Wi-Fi откройте сетевое окно."

    def read(wire, count):
        if wire["approved_original"]["filename"] == "Первый вариант.md":
            return answer(wire, {"g1": bluetooth}, status="RELATED")
        return answer(wire, {"g1": wifi})

    knowledge, _, calls = configure_reader([(DOC, "Первый вариант.md", bluetooth.encode()),
                                           (OTHER, "Второй вариант.md", wifi.encode())],
                                          read, monkeypatch, order=[DOC, OTHER])
    result = knowledge.search_with_llm("Нужно подключить по Wi-Fi.",
                                      read_context({"g1": "Подключить по Wi-Fi"}, {"g1": "ACTION"}))
    assert len(calls) == 2 and result.result == "MATCH"
    assert result.approved.document["source"]["file_id"] == OTHER
    assert not result.alternatives


def test_complete_absence_preserves_human_goal_descriptions(monkeypatch):
    knowledge, _, calls = configure_reader([(DOC, "Другая тема.md", "Справка по другой задаче.".encode())],
                                           lambda wire, count: answer(wire), monkeypatch)
    goals = {"g1": "Узнать отсутствующий параметр", "g2": "Выполнить отсутствующее действие"}
    result = knowledge.search_with_llm("Нужна помощь с другой системой.", read_context(goals))
    assert len(calls) == 1 and result.result == "NO_KNOWLEDGE"
    assert result.missing_goals == tuple(goals.values())


@pytest.mark.parametrize("invalid", ["supported_question", "missing_evidence", "related_mode"])
def test_callback_readers_obey_the_same_closed_status_contract_as_the_provider(invalid):
    text = "Ширина исходного устройства составляет 70 мм."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", text.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})

    def read(payload):
        reference = next(iter(payload["candidate_evidence"]))
        response = {"status": "SUPPORTED", "evidence_ids": [reference], "delivery_mode": "EXCERPT"}
        if invalid == "supported_question":
            response["question"] = "Какой тип устройства?"
        elif invalid == "missing_evidence":
            response = {"status": "MISSING", "evidence_ids": [reference]}
        else:
            response["status"] = "RELATED"
        return {"goals": {"g1": response}}

    knowledge.reader = read
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionReading"):
        knowledge.search_with_llm("Какая ширина?", read_context({"g1": "Узнать ширину"}))


def test_source_grounded_question_waits_until_a_second_original_was_read(monkeypatch):
    condition = "Для одного устройства используется один профиль, для двух устройств — два профиля."

    def read(wire, count):
        if wire["approved_original"]["filename"] != "Условия.md":
            return answer(wire)
        response = answer(wire, {"g1": condition}, status="CLARIFY")
        response["goals"]["g1"].update(question="Одно устройство нужно подключить или два?",
                                      clarification_kind="PHYSICAL_SETUP")
        return response

    knowledge, _, calls = configure_reader([(DOC, "Условия.md", condition.encode()),
                                           (OTHER, "Другой материал.md", "Справка другой программы.".encode()),
                                           (THIRD, "Третий материал.md", "Ещё одна отдельная справка.".encode())],
                                          read, monkeypatch, order=[DOC, OTHER, THIRD])
    result = knowledge.search_with_llm("Нужно подключить устройства.",
                                      read_context({"g1": "Подключить устройства"}, {"g1": "ACTION"}))
    assert [call["wire"]["approved_original"]["filename"] for call in calls] == ["Условия.md", "Другой материал.md"]
    assert result.result == "CLARIFY" and "Одно устройство" in result.question
    assert result.clarification_sources[0].document["source"]["file_id"] == DOC


def test_later_exact_answer_supersedes_an_earlier_source_question(monkeypatch):
    first = "Если тип окна неизвестен, выбор процедуры зависит от него."
    exact = "1. Откройте рабочее окно и выполните проверку."

    def read(wire, count):
        if wire["approved_original"]["filename"] == "Уточняющий материал.md":
            response = answer(wire, {"g1": first}, status="CLARIFY")
            response["goals"]["g1"].update(question="Какое рабочее окно сейчас видно?", clarification_kind="VISIBLE_STATE")
            return response
        return answer(wire, {"g1": exact})

    knowledge, _, calls = configure_reader([(DOC, "Уточняющий материал.md", first.encode()),
                                           (OTHER, "Точная инструкция.md", exact.encode())],
                                          read, monkeypatch, order=[DOC, OTHER])
    result = knowledge.search_with_llm("Выполнить проверку.",
                                      read_context({"g1": "Выполнить проверку"}, {"g1": "ACTION"}))
    assert len(calls) == 2 and result.result == "MATCH" and not result.question
    assert result.approved.document["source"]["file_id"] == OTHER


def test_exact_caption_alias_resolves_to_original_pixels_without_provider_image_fields(monkeypatch):
    caption = "Задняя панель с разъёмом питания. Штекер нельзя вставлять в разъём данных."
    original = instruction([caption])

    def read(wire, count):
        references = wire_evidence(wire, caption, exact=True)
        assert references
        return {"goals": {"g1": wire_goal("SUPPORTED", [references[0]], "EXCERPT")}}

    knowledge, _, _ = configure_reader([(DOC, "Схема корпуса.docx", original)], read, monkeypatch)
    result = knowledge.search_with_llm("Покажите разъём питания на картинке.", read_context({"g1": "Найти разъём питания"}))
    photos = [number for section in result.plan[0].sections for number in section["image_numbers"]]
    assert photos == [1]
    assert result.approved.illustrations[1].data == read_docx_images(original)[0].data
    assert result.approved.document["source"]["sha256"] == hashlib.sha256(original).hexdigest()


def test_complete_view_prefers_canonical_image_proofs_when_text_and_photo_have_identical_captions():
    caption = "Исходная схема задней панели рабочего устройства."
    knowledge, _ = scanner_provider([(DOC, "Схема.docx", instruction([caption]))])
    registry, descriptors = knowledge._complete_instruction_view(knowledge.read(DOC))
    images = {proof["text"] for proof in registry.values() if proof["support_kind"] == "IMAGE"}
    assert images
    assert not any(proof["text"] in images for proof in registry.values() if proof["support_kind"] == "TEXT")
    assert {reference for descriptor in descriptors.values() for reference in descriptor["evidence_ids"]} == set(registry)


def test_reader_source_finding_cannot_replace_the_original_answer(monkeypatch):
    value = "Искомая ширина составляет 70 мм."
    private_finding = "Произвольное внутреннее описание результата чтения."

    def read(wire, count):
        reply = answer(wire, {"g1": value})
        reply["goals"]["g1"]["source_finding"] = private_finding
        return reply

    knowledge, _, _ = configure_reader([(DOC, "Параметры.md", value.encode())], read, monkeypatch)
    result = knowledge.search_with_llm("Какая ширина?", read_context({"g1": "Узнать ширину"}))
    blocks, _, _, _ = render_plan(result.plan, serialize_plan(result.plan))
    assert value in blocks[0][0]
    assert private_finding not in "\n".join(text for text, _ in blocks)


def test_mutation_between_reading_and_plan_verification_rejects_the_stale_original(monkeypatch):
    text = "Ширина исходного устройства составляет 70 мм."
    knowledge, state = scanner_provider([(DOC, "Параметры.md", text.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})

    def read(payload):
        reference = next(identity for identity, proof in payload["candidate_evidence"].items() if text in proof["text"])
        state["originals"][DOC] = "Ширина устройства после обновления составляет 80 мм.".encode()
        return {"goals": {"g1": {"status": "SUPPORTED", "evidence_ids": [reference], "delivery_mode": "EXCERPT"}}}

    knowledge.reader = read
    with pytest.raises(KnowledgeUnavailable, match="SourceChangedDuringSelection"):
        knowledge.search_with_llm("Какая ширина?", read_context({"g1": "Узнать ширину"}))


def test_explicit_legacy_selector_keeps_its_original_protocol_without_using_the_reader(monkeypatch):
    text = "1. Откройте окно проверки и подтвердите рабочие параметры."
    knowledge, _ = scanner_provider([(DOC, "Проверка.md", text.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    calls = []

    def select(payload):
        calls.append(payload)
        return {"result": "MATCH", **selected_part(payload, DOC, "g1", text)}

    knowledge.selector = select
    monkeypatch.setattr(knowledge, "_search_read_instructions", lambda *args: pytest.fail("Legacy selector used reader"))
    result = knowledge.search_with_llm("Выполнить проверку.", read_context({"g1": "Выполнить проверку"}, {"g1": "ACTION"}))
    assert len(calls) == 1 and "candidate_sections" in calls[0]
    assert result.result == "MATCH" and result.approved.document["source"]["file_id"] == DOC


def test_nominated_original_is_read_first_and_verified_before_answering(monkeypatch):
    identities = [str(uuid5(NAMESPACE_URL, "discovery-test/" + str(index))) for index in range(9)]
    value = "Искомое значение: 170 мм."
    rows = [(identity, f"Материал {index}.md", (value if index == 8 else "Материал по другой теме.").encode())
            for index, identity in enumerate(identities)]
    knowledge, _, calls = configure_reader(rows,
        lambda wire, count: answer(wire, {"g1": value}) if wire["approved_original"]["filename"] == "Материал 8.md"
        else answer(wire), monkeypatch, order=identities)
    nominations = []

    def nominate(payload):
        nominations.append(payload)
        return InstructionNomination(file_ids=[identities[8]])

    knowledge.nominator = nominate
    result = knowledge.search_with_llm("Найти специальное значение.", read_context({"g1": "Узнать специальное значение"}))
    assert len(calls) == 1 and len(nominations) == 1
    assert nominations[0]["eligible_file_ids"] == identities
    assert result.result == "MATCH" and result.approved.document["source"]["file_id"] == identities[8]
    assert value in "\n".join(section["text"] for section in result.plan[0].sections)


def test_nomination_outside_the_unread_approved_catalogue_is_rejected(monkeypatch):
    identities = [str(uuid5(NAMESPACE_URL, "invalid-discovery-test/" + str(index))) for index in range(9)]
    rows = [(identity, f"Материал {index}.md", "Материал по другой теме.".encode())
            for index, identity in enumerate(identities)]
    knowledge, _, _ = configure_reader(rows, lambda wire, count: answer(wire), monkeypatch, order=identities)
    knowledge.nominator = lambda payload: InstructionNomination(file_ids=["foreign-source"])
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionNomination"):
        knowledge.search_with_llm("Найти специальное значение.", read_context({"g1": "Узнать специальное значение"}))


def test_explicit_nominator_runs_before_a_lexical_first_match(monkeypatch):
    value = "Искомое значение: 170 мм."
    knowledge, _, calls = configure_reader([(DOC, "Точная справка.md", value.encode()),
                                           (OTHER, "Другой материал.md", "Материал по другой теме.".encode())],
                                          lambda wire, count: answer(wire, {"g1": value}), monkeypatch,
                                          order=[DOC, OTHER])
    nominations = []

    def nominate(payload):
        nominations.append(payload)
        return InstructionNomination(file_ids=[DOC])

    knowledge.nominator = nominate
    result = knowledge.search_with_llm("Найти специальное значение.", read_context({"g1": "Узнать специальное значение"}))
    assert len(calls) == 1 and result.result == "MATCH"
    assert len(nominations) == 1 and nominations[0]["eligible_file_ids"] == [DOC, OTHER]
