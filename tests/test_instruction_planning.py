"""Generic retrieval, request-local sections and verified multi-source plans."""

import copy
import io
import json
from zipfile import ZipFile

import httpx
import pytest

from app.services.rag.catalog import build_catalog_entry, instruction_sections
from app.services.rag.talk import MAX_SELECTION_CONTEXT_CHARS, KnowledgeUnavailable, TalkKnowledge
from tests.test_openwebui_instruction_images import instruction, office
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, OTHER


def context(*goals):
    return {"instruction_planning": True, "requested_goals": {
        f"g{index}": goal for index, goal in enumerate(goals, 1)
    }}


def test_selection_context_retains_current_multi_object_conditions_without_repeated_evidence():
    message = "Уточняю: у меня два устройства, оба по Wi-Fi. Bluetooth не используется."
    facts = [{"key": key, "value": value, "entity": entity, "attribute": key,
              "evidence": message * 3, "negated": negated}
             for entity in ("device_1", "device_2")
             for key, value, negated in (("equipment_type", "принтер", False),
                                         ("connection", "Wi-Fi", False),
                                         ("connection", "Bluetooth", True))]
    state = {"facts": facts, "goals": {"g1": {"text": "Подключить устройства", "evidence": message * 3}},
             "messages": [message * 4], "question": "", "question_field": None}
    original = copy.deepcopy(state)
    bounded = TalkKnowledge._instruction_context({"current_message": message, "task_state": state})

    assert bounded["current_message"] == message
    assert {fact["entity"] for fact in bounded["task_state"]["facts"]} == {"device_1", "device_2"}
    assert len(bounded["task_state"]["facts"]) == 6
    assert all("evidence" not in fact for fact in bounded["task_state"]["facts"])
    assert bounded["task_state"]["goals"] == {"g1": {"text": "Подключить устройства"}}
    assert len(json.dumps(bounded, ensure_ascii=False)) <= MAX_SELECTION_CONTEXT_CHARS
    assert state == original


def selected_part(payload, file_id, goal_id, text):
    reference_id = next(key for key, value in payload["candidate_evidence"][file_id].items()
                        if text in value["text"])
    section_id = next(key for key, value in payload["candidate_sections"][file_id].items()
                      if reference_id in value["evidence_ids"])
    proof = payload["candidate_evidence"][file_id][reference_id]
    return {"file_id": file_id, "section_ids": [section_id], "covered_goals": [{
        "goal_id": goal_id, "evidence": proof["text"], "support_kind": proof["support_kind"],
        "image_numbers": proof["image_numbers"], "section_id": section_id,
    }]}


def test_new_unstructured_instruction_keeps_complete_procedure_and_original_photo():
    caption = "Настройку яркости выполните по исходной фотографии этого устройства."
    body = "Перед настройкой сохраните параметры.\nВыполните первое действие.\nВыполните второе действие."
    knowledge, state = scanner_provider([(DOC, "Новый материал.docx", instruction([caption], extra_paragraph=body))])
    knowledge.selector = lambda payload: {"result": "MATCH", **selected_part(payload, DOC, "g1", caption)}
    result = knowledge.search_with_llm("Хочу изменить яркость устройства", context("Изменить яркость"))
    assert result.result == "MATCH" and result.approved is result.plan[0].approved
    assert list(result.plan[0].sections) == instruction_sections(result.approved.document)
    assert result.plan[0].approved.illustrations[1].data
    assert len([request for request in state["calls"] if "/api/v1/files/" in request.url.path]) == 2


def test_informational_excerpt_keeps_selected_facts_and_mandatory_conditions_without_actions_or_photos():
    width = "Label width: 101,5 мм."
    height = "Label height: 101,5 мм."
    warning = "Не отключайте устройство до завершения текущей печати."
    prerequisite = "Перед настройкой сохраните параметры."
    unrelated = "Переведите выключатель питания в положение OFF для сброса."
    caption = "Удерживая кнопку, включите питание для заводского сброса принтера."
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"
                   for text in [warning, prerequisite, width, height, unrelated])
    knowledge, _ = scanner_provider([(DOC, "Справочник принтера.docx",
                                     instruction([caption], extra_paragraph=body))])

    def select(payload):
        part = selected_part(payload, DOC, "g1", width)
        part["section_ids"] += selected_part(payload, DOC, "g1", height)["section_ids"]
        return {"result": "MATCH", **part, "delivery_mode": "EXCERPT"}

    knowledge.selector = select
    result = knowledge.search_with_llm("Какие ширина и высота этикетки по умолчанию?",
                                       context("Узнать стандартные ширину и высоту"))
    sections = result.plan[0].sections
    texts = [section["text"] for section in sections]
    assert width in texts and height in texts and warning in texts and prerequisite in texts
    assert unrelated not in texts and caption not in texts
    assert not any(section["image_numbers"] for section in sections)


def test_workbook_excerpt_keeps_exact_value_with_original_headers_without_foreign_rows():
    original = office({
        "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                           '<sheets><sheet name="Телефоны" r:id="s1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships><Relationship Id="s1" '
                                    'Target="/xl/worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row><c r="A1" t="inlineStr"><is><t>Рабочие телефоны</t></is></c></row>'
            '<row><c r="A3" t="inlineStr"><is><t>Сотрудник</t></is></c>'
            '<c r="B3" t="inlineStr"><is><t>Добавочный номер</t></is></c></row>'
            '<row><c r="A4" t="inlineStr"><is><t>Руководство</t></is></c><c r="B4"><v>126</v></c></row>'
            '<row><c r="A5" t="inlineStr"><is><t>Отдел кадров</t></is></c><c r="B5"><v>134</v></c></row>'
            '</sheetData></worksheet>',
    })
    knowledge, _ = scanner_provider([(DOC, "Добавочные телефоны.xlsx", original)])
    knowledge.selector = lambda payload: {
        "result": "MATCH", **selected_part(payload, DOC, "g1", "Руководство"), "delivery_mode": "EXCERPT",
    }
    result = knowledge.search_with_llm("Как позвонить руководству?", context("Узнать добавочный руководства"))
    texts = [section["text"] for section in result.plan[0].sections]
    assert texts == ["Лист: Телефоны", "A1: Рабочие телефоны", "A3: Сотрудник | B3: Добавочный номер",
                     "A4: Руководство | B4: 126"]


def test_excerpt_proof_must_belong_to_explicitly_selected_section():
    warning = "Не отключайте питание во время выполнения операции."
    fact = "Стандартная высота этикетки равна 101,5 мм."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", (warning + "\n" + fact).encode())])

    def select(payload):
        part = selected_part(payload, DOC, "g1", warning)
        part["section_ids"] = selected_part(payload, DOC, "g1", fact)["section_ids"]
        return {"result": "MATCH", **part, "delivery_mode": "EXCERPT"}

    knowledge.selector = select
    with pytest.raises(KnowledgeUnavailable, match="InstructionSectionEvidenceMismatch"):
        knowledge.search_with_llm("Какая стандартная высота этикетки?", context("Узнать высоту"))


def test_interleaved_excerpts_in_one_procedure_group_keep_their_own_facts():
    first = "Стандартная ширина этикетки равна 101,5 мм."
    middle = "Рабочий телефон поддержки указан в карточке подразделения."
    last = "Стандартная высота этикетки равна 50 мм."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", (first + "\n" + last).encode()),
                                     (OTHER, "Телефон.md", middle.encode())])

    def select(payload):
        part = selected_part(payload, DOC, "g1", first)
        final = selected_part(payload, DOC, "g3", last)
        part["delivery_mode"] = "EXCERPT"
        final["delivery_mode"] = "EXCERPT"
        other = {**selected_part(payload, OTHER, "g2", middle), "delivery_mode": "EXCERPT"}
        return {"result": "PLAN", "parts": [part, final, other], "missing_goal_ids": []}

    knowledge.selector = select
    result = knowledge.search_with_llm("Ширина, затем телефон поддержки, затем высота",
                                       context("Узнать ширину", "Узнать телефон", "Узнать высоту"))
    assert result.result == "MATCH"
    assert [part.goal_ids for part in result.plan] == [("g1",), ("g2",), ("g3",)]
    assert [section["text"] for section in result.plan[0].sections] == [first]
    assert [section["text"] for section in result.plan[2].sections] == [last]


def test_excerpt_part_cannot_share_sections_between_independent_goals():
    first = "Стандартная ширина этикетки равна 101,5 мм."
    last = "Стандартная высота этикетки равна 50 мм."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", (first + "\n" + last).encode())])

    def select(payload):
        part = selected_part(payload, DOC, "g1", first)
        final = selected_part(payload, DOC, "g2", last)
        part["section_ids"] += final["section_ids"]
        part["covered_goals"] += final["covered_goals"]
        return {"result": "MATCH", **part, "delivery_mode": "EXCERPT"}

    knowledge.selector = select
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionExcerpt"):
        knowledge.search_with_llm("Узнать ширину и высоту", context("Узнать ширину", "Узнать высоту"))


def test_short_new_instruction_is_selectable_by_exact_backend_reference_only():
    text = "Нажмите F8."
    knowledge, _ = scanner_provider([(DOC, "Открытие окна.md", text.encode())])
    original_transport = knowledge.transport

    def handler(request):
        if request.url.path != "/api/chat/completions":
            return original_transport.handle_request(request)
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        reference, proof = next(iter(wire["candidates"]["s1"].items()))
        assert proof["text"] == text and len(text) < 20
        section = next(iter(wire["candidate_sections"]["s1"]))
        schema = body["response_format"]["json_schema"]["schema"]
        match = next(row for row in schema["anyOf"] if row["properties"]["result"]["const"] == "MATCH")
        assert match["properties"]["covered_goals"]["minItems"] == 1
        assert schema["$defs"]["GoalReference_s1"]["properties"]["evidence_id"]["enum"] == [reference]
        choice = {"result": "MATCH", "file_id": "s1", "section_ids": [section],
                  "covered_goals": [{"goal_id": "g1", "evidence_id": reference}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    knowledge.transport = httpx.MockTransport(handler)
    result = knowledge.search_with_llm("Открыть окно", context("Открыть окно"))
    assert result.plan[0].sections[0]["text"] == text
    assert result.approved.document["content"] == text


def test_reference_adapter_requires_explicit_delivery_mode_and_restores_an_exact_excerpt():
    fact = "Стандартная ширина этикетки равна 101,5 мм."
    action = "Для сброса параметров включите питание, удерживая кнопку."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", (fact + "\n" + action).encode())])
    original_transport = knowledge.transport

    def handler(request):
        if request.url.path != "/api/chat/completions":
            return original_transport.handle_request(request)
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        reference = next(key for key, value in wire["candidates"]["s1"].items() if fact in value["text"])
        section = next(key for key, value in wire["candidate_sections"]["s1"].items()
                       if reference in value["evidence_ids"])
        schema = body["response_format"]["json_schema"]["schema"]
        match = next(row for row in schema["anyOf"] if row["properties"]["result"]["const"] == "MATCH")
        assert "delivery_mode" in match["required"]
        assert match["properties"]["delivery_mode"]["enum"] == ["PROCEDURE", "EXCERPT"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "result": "MATCH", "file_id": "s1", "delivery_mode": "EXCERPT", "section_ids": [section],
            "covered_goals": [{"goal_id": "g1", "evidence_id": reference}],
        })}}]})

    knowledge.transport = httpx.MockTransport(handler)
    result = knowledge.search_with_llm("Какая ширина этикетки по умолчанию?", context("Узнать ширину этикетки"))
    assert [section["text"] for section in result.plan[0].sections] == [fact]


def test_document_headings_remain_searchable_without_proving_a_procedure():
    heading = "Первый запуск рабочего приложения"
    action = "Откройте приложение и введите выданный рабочий адрес."
    source = "# " + heading + "\n## Подключение\n" + action
    knowledge, _ = scanner_provider([(DOC, "Первый запуск.md", source.encode())])

    def select(payload):
        references = payload["candidate_evidence"][DOC]
        assert all(heading not in proof["text"] for proof in references.values())
        assert all(proof["text"] != "## Подключение" for proof in references.values())
        return {"result": "MATCH", **selected_part(payload, DOC, "g1", action)}

    knowledge.selector = select
    result = knowledge.search_with_llm(heading, context(heading))
    texts = [section["text"] for section in result.plan[0].sections]
    assert "## Подключение" in texts and action in texts


def test_heading_alone_cannot_produce_an_action_plan():
    knowledge, _ = scanner_provider([(DOC, "Только заголовок.md", "# Первый запуск приложения".encode())])

    def select(payload):
        assert not payload["candidate_evidence"][DOC]
        return {"result": "NO_KNOWLEDGE", "missing_goal_ids": ["g1"]}

    knowledge.selector = select
    result = knowledge.search_with_llm("Первый запуск приложения", context("Запустить приложение"))
    assert result.result == "NO_KNOWLEDGE" and not result.plan


def test_legacy_free_quote_still_requires_twenty_characters():
    text = "Нажмите F8."
    knowledge, _ = scanner_provider([(DOC, "Открытие окна.md", text.encode())])
    knowledge.selector = lambda payload: {"result": "MATCH", "file_id": DOC, "covered_goals": [{
        "goal_id": "g1", "evidence": text, "support_kind": "TEXT", "image_numbers": [],
    }]}
    with pytest.raises(KnowledgeUnavailable, match="InstructionEvidenceMismatch"):
        knowledge.search_with_llm("Открыть окно", {"requested_goals": {"g1": "Открыть окно"}})


def test_short_planning_proof_cannot_be_an_arbitrary_substring_of_an_original():
    text = "Нажмите F8."
    knowledge, _ = scanner_provider([(DOC, "Открытие окна.md", text.encode())])

    def select(payload):
        part = selected_part(payload, DOC, "g1", text)
        part["covered_goals"][0]["evidence"] = "Н"
        return {"result": "MATCH", **part}

    knowledge.selector = select
    with pytest.raises(KnowledgeUnavailable, match="InstructionEvidenceReferenceMismatch"):
        knowledge.search_with_llm("Открыть окно", context("Открыть окно"))


def test_multiline_original_image_caption_keeps_image_evidence_for_each_exact_fragment():
    action = "Для увеличения громкости отсканируйте нужный настроечный код на листе."
    caption = "Лист аппаратных настроек рабочего оборудования.\n" + action + (
        "\nДля изменения подсветки используйте отмеченный код на исходном листе."
    )
    with ZipFile(io.BytesIO(instruction([caption]))) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    # XML attribute line feeds use character references in actual DOCX files.
    parts["word/document.xml"] = parts["word/document.xml"].replace(b"\n", b"&#10;")
    knowledge, _ = scanner_provider([(DOC, "Новые настройки.docx", office(parts))])

    def select(payload):
        part = selected_part(payload, DOC, "g1", action)
        proof = part["covered_goals"][0]
        assert proof["support_kind"] == "IMAGE" and proof["image_numbers"] == [1]
        return {"result": "MATCH", **part}

    knowledge.selector = select
    result = knowledge.search_with_llm("Увеличить громкость оборудования", context("Увеличить громкость"))
    image = result.plan[0].approved.illustrations[1]
    assert "\n" in image.caption and action in image.caption
    assert result.plan[0].approved.document["source"]["sha256"]


def test_view_keeps_original_image_when_six_text_sections_score_higher():
    caption = "Считайте отмеченный символ, чтобы аппарат подавал более тихий сигнал."
    paragraphs = "".join(f"<w:p><w:r><w:t>Громкость оборудования. Сведения о громкости, часть {number}."
                         "</w:t></w:r></w:p>" for number in range(6))
    knowledge, _ = scanner_provider([(DOC, "Настройка оборудования.docx",
                                     instruction([caption], extra_paragraph=paragraphs))])

    def select(payload):
        part = selected_part(payload, DOC, "g1", caption)
        assert part["covered_goals"][0]["support_kind"] == "IMAGE"
        assert len(payload["candidate_evidence"][DOC]) <= 8
        return {"result": "MATCH", **part}

    knowledge.selector = select
    result = knowledge.search_with_llm("Уменьшить громкость оборудования", context("Уменьшить громкость"))
    assert result.result == "MATCH" and result.plan[0].approved.illustrations[1].data


def test_compound_goals_use_two_verified_sources_in_requested_order():
    first = "Настройте яркость устройства по выбранному режиму на фотографии."
    second = "Измените размер шрифта рабочего приложения в его окне настроек."
    knowledge, _ = scanner_provider([(DOC, "Оборудование.docx", instruction([first])),
                                     (OTHER, "Приложение.docx", instruction([second]))])
    knowledge.selector = lambda payload: {"result": "PLAN", "parts": [
        selected_part(payload, OTHER, "g2", second), selected_part(payload, DOC, "g1", first),
    ], "missing_goal_ids": []}
    result = knowledge.search_with_llm("Изменить яркость и размер шрифта", context("Яркость", "Размер шрифта"))
    assert result.result == "MATCH" and len(result.plan) == 2
    assert [part.goal_ids for part in result.plan] == [("g1",), ("g2",)]
    assert [part.approved.document["source"]["file_id"] for part in result.plan] == [DOC, OTHER]
    assert result.covered_goals == ("Яркость", "Размер шрифта")


@pytest.mark.parametrize("grouped", [False, True])
def test_interleaved_goals_from_same_source_keep_only_their_own_procedure_groups(grouped):
    first = "Настройте громкость устройства по указанному текстовому режиму."
    middle = "Измените размер шрифта рабочего приложения в его окне настроек."
    last = "Настройте подсветку устройства по указанному текстовому режиму."
    source = "Не отключайте устройство до окончания работы.\n# Громкость\n" + first + "\n# Подсветка\n" + last
    knowledge, _ = scanner_provider([(DOC, "Оборудование.md", source.encode()),
                                     (OTHER, "Приложение.md", middle.encode())])

    def select(payload):
        a1 = selected_part(payload, DOC, "g1", first)
        b = selected_part(payload, OTHER, "g2", middle)
        a3 = selected_part(payload, DOC, "g3", last)
        if grouped:
            a1["covered_goals"] += a3["covered_goals"]
            a1["section_ids"] += a3["section_ids"]
            parts = [a1, b]
        else:
            parts = [a1, b, a3]
        return {"result": "PLAN", "parts": parts, "missing_goal_ids": []}

    knowledge.selector = select
    result = knowledge.search_with_llm("Сначала громкость, затем шрифт, затем подсветка",
                                       context("Громкость", "Размер шрифта", "Подсветка"))
    assert result.result == "MATCH"
    assert [part.goal_ids for part in result.plan] == [("g1",), ("g2",), ("g3",)]
    assert [part.approved.document["source"]["file_id"] for part in result.plan] == [DOC, OTHER, DOC]
    assert first in "\n".join(section["text"] for section in result.plan[0].sections)
    assert last not in "\n".join(section["text"] for section in result.plan[0].sections)
    assert last in "\n".join(section["text"] for section in result.plan[2].sections)
    assert first not in "\n".join(section["text"] for section in result.plan[2].sections)
    assert all(any(section["kind"] == "WARNING" for section in part.sections)
               for part in [result.plan[0], result.plan[2]])


def test_indivisible_procedure_interleaving_requires_grounded_confirmation_before_actions():
    first = "Настройте громкость устройства по указанному текстовому режиму."
    middle = "Измените размер шрифта рабочего приложения в его окне настроек."
    last = "Настройте подсветку устройства по указанному текстовому режиму."
    knowledge, _ = scanner_provider([(DOC, "Оборудование.md", (first + "\n" + last).encode()),
                                     (OTHER, "Приложение.md", middle.encode())])

    def select(payload):
        a1 = selected_part(payload, DOC, "g1", first)
        a3 = selected_part(payload, DOC, "g3", last)
        a1["covered_goals"] += a3["covered_goals"]
        a1["section_ids"] += a3["section_ids"]
        return {"result": "PLAN", "parts": [a1, selected_part(payload, OTHER, "g2", middle)],
                "missing_goal_ids": []}

    knowledge.selector = select
    result = knowledge.search_with_llm("Громкость, затем шрифт, затем подсветка",
                                       context("Громкость", "Размер шрифта", "Подсветка"))
    assert result.result == "CLARIFY" and not result.plan and result.approved is None
    assert "предложенном порядке" in result.question and result.question.endswith("?")
    assert result.proposed_goal_order == ("g1", "g3", "g2")
    assert result.clarification_sources[0].document["source"]["file_id"] == DOC
    declined = knowledge.search_with_llm("Сохранить первоначальный порядок", {
        **context("Громкость", "Размер шрифта", "Подсветка"), "rejected_goal_orders": [["g1", "g3", "g2"]],
    })
    assert declined.result == "CLARIFY" and not declined.proposed_goal_order
    assert "отдельно" in declined.question


@pytest.mark.parametrize("delivery_mode", ["PROCEDURE", "EXCERPT"])
@pytest.mark.parametrize("invalid", ["unknown_section", "other_section", "duplicate_goal", "changed_original"])
def test_plans_preserve_source_section_goal_and_freshness_proofs(invalid, delivery_mode):
    text = "Откройте настройки рабочего приложения и измените размер шрифта."
    knowledge, state = scanner_provider([(DOC, "Размер шрифта.md", ("# Действия\n" + text +
                                                               "\n# Другое\nОтдельная независимая процедура.").encode())])

    def select(payload):
        part = selected_part(payload, DOC, "g1", text)
        part["delivery_mode"] = delivery_mode
        if invalid == "unknown_section":
            part["section_ids"] = ["section-invented"]
        elif invalid == "other_section":
            part["section_ids"] = [next(key for key, value in payload["candidate_sections"][DOC].items()
                                        if any("независимая" in payload["candidate_evidence"][DOC][ref]["text"]
                                               for ref in value["evidence_ids"]))]
        elif invalid == "duplicate_goal":
            part["covered_goals"] *= 2
        else:
            state["originals"][DOC] += b" changed"
        return {"result": "MATCH", **part}

    knowledge.selector = select
    with pytest.raises(KnowledgeUnavailable):
        knowledge.search_with_llm("Изменить размер шрифта", context("Изменить шрифт"))


def test_absence_requires_expanded_candidates_and_can_find_an_unseen_source():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 7)]
    rows = [(file_id, f"Материал {number}.md", f"Готовая инструкция для нового приложения номер {number}.".encode())
            for number, file_id in enumerate(ids)]
    knowledge, _ = scanner_provider(rows)
    calls = []

    def select(payload):
        calls.append(copy.deepcopy(payload))
        if not payload["expanded_search"]:
            return {"result": "NO_KNOWLEDGE", "missing_goal_ids": ["g1"]}
        part = selected_part(payload, ids[-1], "g1", "номер 5")
        return {"result": "MATCH", **part}

    knowledge.selector = select
    result = knowledge.search_with_llm("Новое приложение", context("Открыть приложение"))
    assert len(calls) == 2 and calls[1]["expanded_search"] is True
    assert len(calls[0]["candidates"]) == 4 and len(calls[1]["candidates"]) == 6
    assert result.approved.document["source"]["file_id"] == ids[-1]


class MemoryCatalog:
    def __init__(self, originals):
        self.values = {item.document["source"]["file_id"]: build_catalog_entry(item) for item in originals}

    def reconcile(self, files):
        self.values = {key: value for key, value in self.values.items() if key in files}

    def entries(self, files):
        return self.values

    def load_verified(self, file_id, original, files):
        return None

    def store(self, approved, **kwargs):
        self.values[approved.document["source"]["file_id"]] = build_catalog_entry(approved)


def test_strong_original_content_hit_survives_unrelated_vector_and_filename_votes():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 7)]
    action = "Чтобы уменьшить громкость устройства, выберите тихий режим по исходной инструкции."
    rows = [(file_id, f"Настройка устройства {number}.md", "Настройка устройства для подключения к программе.".encode())
            for number, file_id in enumerate(ids[:-1], 1)]
    rows.append((ids[-1], "Памятка.md", action.encode()))
    knowledge, _ = scanner_provider(rows, chunks=[(key, "Настройка устройства") for key in ids[:3]])
    knowledge.set_catalog(MemoryCatalog([knowledge.read(file_id) for file_id in ids]))
    calls = []

    def select(payload):
        calls.append(payload)
        assert ids[-1] in payload["candidates"]
        assert len(payload["candidates"]) == 4
        return {"result": "MATCH", **selected_part(payload, ids[-1], "g1", action)}

    knowledge.selector = select
    result = knowledge.search_with_llm("Настройка устройства. Уменьшить громкость.", context("Уменьшить громкость"))
    assert result.approved.document["source"]["file_id"] == ids[-1] and len(calls) == 1


def test_full_content_catalogue_retrieves_new_subject_without_filename_rules_or_all_downloads():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 11)]
    rows = [(file_id, f"Материал {number}.md", ("Обычная справка рабочего места." if number < 9 else
                                             "Экспортируйте отчёт нового сервиса в формате CSV.").encode())
            for number, file_id in enumerate(ids)]
    knowledge, state = scanner_provider(rows)
    knowledge.set_catalog(MemoryCatalog([knowledge.read(file_id) for file_id in ids]))
    state["calls"].clear()
    calls = []

    def select(payload):
        calls.append(payload)
        return {"result": "MATCH", **selected_part(payload, ids[-1], "g1", "Экспортируйте")}

    knowledge.selector = select
    result = knowledge.search_with_llm("Экспорт отчёта в CSV", context("Экспортировать отчёт"))
    assert result.approved.document["source"]["file_id"] == ids[-1] and len(calls) == 1
    assert len([request for request in state["calls"] if "/api/v1/files/" in request.url.path]) == 5


def test_per_user_exclusion_does_not_remove_source_from_shared_catalogue():
    text = "Откройте настройки приложения и измените размер шрифта."
    knowledge, _ = scanner_provider([(DOC, "Первый.md", text.encode()), (OTHER, "Второй.md", text.encode())])
    catalog = MemoryCatalog([knowledge.read(DOC), knowledge.read(OTHER)])
    knowledge.set_catalog(catalog)
    knowledge.selector = lambda payload: {"result": "MATCH", **selected_part(payload, OTHER, "g1", text)}
    knowledge.search_with_llm("Размер шрифта", context("Изменить шрифт"), excluded=[DOC])
    assert set(catalog.values) == {DOC, OTHER}


def test_new_upload_is_indexed_on_demand_without_a_manual_sync():
    text = "Откройте новый сервис и экспортируйте отчёт в формате CSV."
    knowledge, _ = scanner_provider([(DOC, "Новый материал.md", text.encode())])
    catalog = MemoryCatalog([])
    knowledge.set_catalog(catalog)
    knowledge.selector = lambda payload: {"result": "MATCH", **selected_part(payload, DOC, "g1", text)}
    result = knowledge.search_with_llm("Экспорт отчёта CSV", context("Экспорт отчёта"))
    assert result.result == "MATCH" and DOC in catalog.values


def test_two_hundred_sources_keep_a_bounded_prompt_and_find_new_upload_at_end():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 201)]
    target = "Экспортируйте отчёт нового сервиса в формате CSV через его меню."
    rows = [(file_id, f"Материал {number:03}.md", (target if number == 200 else
                                               "Обычная инструкция для рабочего места.").encode())
            for number, file_id in enumerate(ids, 1)]
    knowledge, state = scanner_provider(rows)
    catalog = MemoryCatalog([knowledge.read(file_id) for file_id in ids[:-1]])
    knowledge.set_catalog(catalog)
    state["calls"].clear()
    calls = []

    def select(payload):
        calls.append(payload)
        assert payload["catalogue_total"] == 200 and payload["catalogue_complete"] is False
        assert len(payload["catalogue"]) <= 32
        assert len(json.dumps(payload["catalogue"], ensure_ascii=False)) <= 12_000
        assert set(payload["catalogue_cards"]) <= set(payload["catalogue"])
        assert ids[-1] in payload["candidates"] and ids[-1] in payload["catalogue"]
        return {"result": "MATCH", **selected_part(payload, ids[-1], "g1", target)}

    knowledge.selector = select
    result = knowledge.search_with_llm("Экспортировать отчёт CSV", context("Экспорт отчёта CSV"))
    assert result.approved.document["source"]["file_id"] == ids[-1] and len(calls) == 1
    assert len(catalog.values) == 200
    assert len([request for request in state["calls"] if "/api/v1/files/" in request.url.path]) == 5


def test_incomplete_cold_catalogue_reports_indexing_instead_of_false_absence():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 11)]
    knowledge, _ = scanner_provider([(file_id, f"Материал {number}.md", b"Documented procedure for an application.")
                                     for number, file_id in enumerate(ids)])
    knowledge.set_catalog(MemoryCatalog([]))
    knowledge.selector = lambda payload: {"result": "NO_KNOWLEDGE", "missing_goal_ids": ["g1"]}
    with pytest.raises(KnowledgeUnavailable, match="SelectionIndexIncomplete"):
        knowledge.search_with_llm("New application", context("New task"))


def test_complete_lexical_catalogue_can_establish_absence_during_vector_outage():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 7)]
    knowledge, _ = scanner_provider([(file_id, f"Материал {index}.md", b"Ordinary documented action.")
                                     for index, file_id in enumerate(ids)])
    knowledge.set_catalog(MemoryCatalog([knowledge.read(file_id) for file_id in ids]))
    original_transport = knowledge.transport

    def handler(request):
        if request.url.path == "/api/v1/retrieval/query/collection":
            return httpx.Response(503)
        return original_transport.handle_request(request)

    knowledge.transport = httpx.MockTransport(handler)
    calls = []
    knowledge.selector = lambda payload: calls.append(payload) or {"result": "NO_KNOWLEDGE",
                                                                 "missing_goal_ids": ["g1"]}
    result = knowledge.search_with_llm("Unsupported application", context("Unsupported task"))
    assert result.result == "NO_KNOWLEDGE" and len(calls) == 2
    assert set(calls[0]["catalogue_cards"]) == set(ids)


def test_real_reference_adapter_decodes_selected_section_and_original_image():
    caption = "Измените громкость устройства с помощью настроечного кода на фотографии."
    knowledge, _ = scanner_provider([(DOC, "Новая настройка.docx", instruction([caption]))])
    original_transport = knowledge.transport
    calls = []

    def handler(request):
        if request.url.path != "/api/chat/completions":
            return original_transport.handle_request(request)
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        calls.append(wire)
        assert wire["instruction_planning"] is True and wire["evidence_references"] is True
        assert "candidate_evidence" not in wire and "candidate_image_captions" not in wire
        refs = wire["candidates"]["s1"]
        reference = next(key for key, value in refs.items() if value["support_kind"] == "IMAGE")
        section = next(key for key, value in wire["candidate_sections"]["s1"].items()
                       if reference in value["evidence_ids"])
        schema = body["response_format"]["json_schema"]["schema"]
        match = next(branch for branch in schema["anyOf"] if branch["properties"]["result"]["const"] == "MATCH")
        assert match["properties"]["section_ids"]["items"]["enum"]
        response = {"result": "MATCH", "file_id": "s1", "section_ids": [section],
                    "covered_goals": [{"goal_id": "g1", "evidence_id": reference}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)}}]})

    knowledge.transport = httpx.MockTransport(handler)
    result = knowledge.search_with_llm("Изменить громкость", context("Изменить громкость"))
    assert len(calls) == 1 and result.plan[0].approved.illustrations[1].data


def test_real_reference_adapter_binds_multi_source_parts_to_source_sections():
    first = "Измените яркость оборудования с помощью кода на фотографии."
    second = "Откройте настройки приложения и установите нужный размер шрифта."
    knowledge, _ = scanner_provider([(DOC, "Оборудование.docx", instruction([first])),
                                     (OTHER, "Программа.docx", instruction([second]))])
    original_transport = knowledge.transport

    def handler(request):
        if request.url.path != "/api/chat/completions":
            return original_transport.handle_request(request)
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        parts = []
        for file_id, goal_id, evidence in [("s1", "g1", first), ("s2", "g2", second)]:
            reference = next(key for key, value in wire["candidates"][file_id].items() if evidence in value["text"])
            section = next(key for key, value in wire["candidate_sections"][file_id].items()
                           if reference in value["evidence_ids"])
            parts.append({"file_id": file_id, "section_ids": [section],
                          "covered_goals": [{"goal_id": goal_id, "evidence_id": reference}]})
        schema = body["response_format"]["json_schema"]["schema"]
        branch = next(row for row in schema["anyOf"] if row["properties"]["result"]["const"] == "PLAN")
        assert branch["properties"]["parts"]["maxItems"] == 2
        assert schema["$defs"]["PartReference_s1"]["properties"]["file_id"]["const"] == "s1"
        assert "delivery_mode" in schema["$defs"]["PartReference_s1"]["required"]
        excerpt = schema["$defs"]["ExcerptPartReference_s1"]
        assert excerpt["properties"]["covered_goals"]["maxItems"] == 1
        assert excerpt["properties"]["delivery_mode"]["const"] == "EXCERPT"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "result": "PLAN", "parts": parts, "missing_goal_ids": [],
        })}}]})

    knowledge.transport = httpx.MockTransport(handler)
    result = knowledge.search_with_llm("Изменить яркость и размер шрифта", context("Яркость", "Шрифт"))
    assert result.result == "MATCH" and len(result.plan) == 2
    assert [part.goal_ids for part in result.plan] == [("g1",), ("g2",)]


def test_reference_ids_disambiguate_identical_quotes_in_interleaved_procedure_groups():
    action = "Нажмите кнопку сохранения выбранного профиля."
    source = "# Личный профиль\n" + action + "\n# Рабочий профиль\n" + action
    middle = "Измените размер шрифта в окне рабочего приложения."
    knowledge, _ = scanner_provider([(DOC, "Профили.md", source.encode()), (OTHER, "Шрифт.md", middle.encode())])
    original_transport = knowledge.transport

    def handler(request):
        if request.url.path != "/api/chat/completions":
            return original_transport.handle_request(request)
        wire = json.loads(json.loads(request.content)["messages"][-1]["content"])
        source_alias = next(key for key, name in wire["catalogue"].items() if name == "Профили.md")
        other_alias = next(key for key, name in wire["catalogue"].items() if name == "Шрифт.md")

        def anchor(file_id, title, text, goal_id):
            section_id, section = next((key, value) for key, value in wire["candidate_sections"][file_id].items()
                                       if value["title"] == title and any(
                                           wire["candidates"][file_id][ref]["text"] == text
                                           for ref in value["evidence_ids"]))
            reference = next(ref for ref in section["evidence_ids"] if wire["candidates"][file_id][ref]["text"] == text)
            return section_id, {"goal_id": goal_id, "evidence_id": reference}

        first_section, first_goal = anchor(source_alias, "Личный профиль", action, "g1")
        last_section, last_goal = anchor(source_alias, "Рабочий профиль", action, "g3")
        middle_section, middle_goal = anchor(other_alias, middle, middle, "g2")
        choice = {"result": "PLAN", "parts": [
            {"file_id": source_alias, "section_ids": [first_section, last_section],
             "covered_goals": [first_goal, last_goal]},
            {"file_id": other_alias, "section_ids": [middle_section], "covered_goals": [middle_goal]},
        ], "missing_goal_ids": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    knowledge.transport = httpx.MockTransport(handler)
    result = knowledge.search_with_llm("Личный профиль, затем шрифт, затем рабочий профиль",
                                       context("Личный профиль", "Шрифт", "Рабочий профиль"))
    assert [part.goal_ids for part in result.plan] == [("g1",), ("g2",), ("g3",)]
    first_text = "\n".join(section["text"] for section in result.plan[0].sections)
    last_text = "\n".join(section["text"] for section in result.plan[2].sections)
    assert "Личный профиль" in first_text and "Рабочий профиль" not in first_text
    assert "Рабочий профиль" in last_text and "Личный профиль" not in last_text
