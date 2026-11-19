"""Known task kinds constrain rendering while aliases remain discovery-only."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from app.services.llm.instruction_selection import (
    InstructionReferenceSelection, instruction_goal_metadata, reference_selection_schema,
)
from app.services.llm.openwebui import OpenWebUILLM
from app.services.rag.talk import KnowledgeUnavailable, TalkKnowledge
from tests.test_instruction_planning import context, selected_part
from tests.test_openwebui_instruction_images import instruction
from tests.test_openwebui import settings
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, OTHER


def schema_payload(kinds):
    return {"instruction_planning": True, "requested_goals": {key: "Цель " + key for key in kinds},
            "requested_goal_kinds": kinds,
            "catalogue": {"s1": "Материал.md"},
            "candidates": {"s1": {"t1": {"text": "Исходное действие или справочное значение.",
                                          "support_kind": "TEXT", "image_numbers": []}}},
            "candidate_sections": {"s1": {"section-1": {"evidence_ids": ["t1"]}}}}


@pytest.mark.parametrize(("kind", "mode"), [("LOOKUP", "EXCERPT"), ("ACTION", "PROCEDURE")])
def test_known_single_goal_kind_leaves_only_its_delivery_mode_in_closed_grammar(kind, mode):
    schema = reference_selection_schema(InstructionReferenceSelection.model_json_schema(), schema_payload({"g1": kind}))
    matches = [branch for branch in schema["anyOf"] if branch["properties"]["result"]["const"] == "MATCH"]
    assert matches and all(branch["properties"]["delivery_mode"]["enum"] == [mode] for branch in matches)
    assert all("delivery_mode" in branch["required"] for branch in matches)


def test_mixed_plan_parts_can_cover_only_goal_ids_matching_their_mode():
    payload = schema_payload({"g1": "LOOKUP", "g2": "ACTION"})
    schema = reference_selection_schema(InstructionReferenceSelection.model_json_schema(), payload)
    assert not any(branch["properties"]["result"]["const"] == "MATCH" for branch in schema["anyOf"])
    plan = next(branch for branch in schema["anyOf"] if branch["properties"]["result"]["const"] == "PLAN")
    for reference in plan["properties"]["parts"]["items"]["anyOf"]:
        part = schema["$defs"][reference["$ref"].split("/")[-1]]["properties"]
        mode = part["delivery_mode"]["const"]
        goal_ref = part["covered_goals"]["items"]["$ref"].split("/")[-1]
        goal_ids = schema["$defs"][goal_ref]["properties"]["goal_id"]["enum"]
        assert goal_ids == (["g1"] if mode == "EXCERPT" else ["g2"])


@pytest.mark.parametrize(("kind", "wrong_mode"), [("LOOKUP", "PROCEDURE"), ("ACTION", "EXCERPT")])
def test_backend_rejects_wrong_mode_even_when_a_provider_bypasses_the_grammar(kind, wrong_mode):
    fact = "Ширина носителя равна 70 мм."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", fact.encode())])
    calls = []

    def select(payload):
        calls.append(payload)
        return {"result": "MATCH", "delivery_mode": wrong_mode, **selected_part(payload, DOC, "g1", fact)}

    knowledge.selector = select
    request = {**context("Узнать ширину"), "requested_goal_kinds": {"g1": kind}}
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionDeliveryMode"):
        knowledge.search_with_llm("Какая ширина?", request)
    assert len(calls) == 2 and calls[-1]["delivery_mode_retry"] is True


def test_known_lookup_enters_scoped_planning_even_without_legacy_planning_flag():
    fact = "Стандартная ширина равна 70 мм."
    action = "Для сброса устройства выключите питание и удерживайте кнопку."
    knowledge, _ = scanner_provider([(DOC, "Параметры.md", (fact + "\n" + action).encode())])
    knowledge.selector = lambda payload: {
        "result": "MATCH", "delivery_mode": "EXCERPT", **selected_part(payload, DOC, "g1", fact),
    }
    result = knowledge.search_with_llm("Какая стандартная ширина?", {
        "requested_goals": {"g1": "Узнать стандартную ширину"}, "requested_goal_kinds": {"g1": "LOOKUP"},
    })
    assert [section["text"] for section in result.plan[0].sections] == [fact]


@pytest.mark.parametrize("metadata", [{"requested_goal_kinds": {}}, {"goal_search_queries": {}},
                                      {"requested_goal_kinds": {}, "goal_search_queries": {"g1": []}}])
def test_empty_optional_goal_metadata_keeps_the_legacy_selection_protocol(metadata):
    text = "Откройте рабочий клиент Маяк и выберите свой профиль."
    knowledge, _ = scanner_provider([(DOC, "Клиент Маяк.md", text.encode())])

    def select(payload):
        assert not payload.get("instruction_planning") and "candidate_sections" not in payload
        return {"result": "MATCH", "file_id": DOC, "covered_goals": [{
            "goal_id": "g1", "evidence": text, "support_kind": "TEXT", "image_numbers": [],
        }]}

    knowledge.selector = select
    result = knowledge.search_with_llm("Открыть клиент Маяк", {
        "requested_goals": {"g1": "Открыть клиент Маяк"}, **metadata,
    })
    assert result.result == "MATCH" and not result.plan


def test_mixed_action_and_lookup_plan_keeps_each_source_scope():
    fact = "Добавочный руководителя смены — 126."
    unrelated = "Добавочный другого отдела — 999."
    steps = "Откройте программу.\nВыберите рабочую папку."
    knowledge, _ = scanner_provider([(DOC, "Справочник.md", (fact + "\n" + unrelated).encode()),
                                     (OTHER, "Рабочая программа.md", steps.encode())])
    knowledge.selector = lambda payload: {"result": "PLAN", "parts": [
        {**selected_part(payload, DOC, "g1", fact), "delivery_mode": "EXCERPT"},
        {**selected_part(payload, OTHER, "g2", "Откройте программу"), "delivery_mode": "PROCEDURE"},
    ]}
    result = knowledge.search_with_llm("Найти телефон и открыть программу", {
        **context("Найти добавочный руководителя", "Открыть рабочую папку"),
        "requested_goal_kinds": {"g1": "LOOKUP", "g2": "ACTION"},
    })
    assert [section["text"] for section in result.plan[0].sections] == [fact]
    assert [section["text"] for section in result.plan[1].sections] == steps.splitlines()


def test_english_discovery_aliases_retrieve_an_opaque_new_source_without_changing_the_goal():
    problem = "Узнать стандартные ширину и высоту этикетки"
    rows = [("a", "Материал А.md", "Ширина и высота упаковочного материала для кассеты."),
            ("b", "Материал Б.md", "Рабочий справочник пользователя."),
            ("c", "Материал В.md", "Размер рулона и ленты для устройства."),
            ("d", "Материал Г.md", "Правила открытия приложения."),
            ("target", "Материал Д.md", "Factory default settings. Label width: 70 mm. Label height: 40 mm.")]
    entries = {key: SimpleNamespace(card={"title": title}, content=text) for key, title, text in rows}
    files = {key: {"filename": title} for key, title, _ in rows}
    request = {"requested_goal_kinds": {"g1": "LOOKUP"},
               "goal_search_queries": {"g1": ["factory default label width height"]}}
    unchanged = copy.deepcopy(request)
    ranked = TalkKnowledge._instruction_ranked_sources(problem, {"g1": problem}, entries, files, ["c"], context=request)
    assert "target" in ranked[:4]
    assert request == unchanged


def test_aliases_rank_original_table_fragments_without_becoming_evidence():
    facts = ["Label width: 70 mm.", "Label height: 40 mm."]
    content = "\n".join([f"Device setting item {index}: enabled." for index in range(12)] + facts)
    knowledge, _ = scanner_provider([(DOC, "Новая таблица.md", content.encode())])
    approved = knowledge.read(DOC)
    ordinary, _ = TalkKnowledge._instruction_source_view(approved, "Какие ширина и высота по умолчанию?")
    discovered, _ = TalkKnowledge._instruction_source_view(
        approved, "Какие ширина и высота по умолчанию?", search_queries=["default label width height"],
    )
    assert not any(fact in proof["text"] for fact in facts for proof in ordinary.values())
    assert all(any(fact in proof["text"] for proof in discovered.values()) for fact in facts)
    assert all(proof["text"] in content for proof in discovered.values())


@pytest.mark.parametrize("mixed", [False, True])
def test_lookup_keeps_matching_aligned_table_rows_instead_of_unrelated_photos_or_guards(mixed):
    from xml.sax.saxutils import escape

    table = ["Parameter             Standard setting", "Speed       100 mm/s", "Second speed       80 mm/s",
             "Density       8", "Canvas width       70 mm", "width unit", "Canvas height       40 mm", "height unit"]
    paragraphs = ["Factory configuration is restored after initialization.", *table,
                  "Требуется изменять служебное задание.", "Требуется изменять интерфейс."]
    body = "".join("<w:p><w:r><w:t>" + escape(text) + "</w:t></w:r></w:p>" for text in paragraphs)
    captions = ["Factory standard interface setup in the diagnostic program.",
                "Factory standard automatic job setup in the diagnostic program."]
    knowledge, _ = scanner_provider([(DOC, "Будущий справочник.docx", instruction(captions, extra_paragraph=body))])
    approved = knowledge.read(DOC)
    refs, _ = TalkKnowledge._instruction_source_view(
        approved, "Узнать заводские размеры носителя", search_queries=["factory standard canvas size"],
        lookup=True, mixed=mixed,
    )
    texts = [proof["text"] for proof in refs.values()]
    assert any("Canvas width" in text for text in texts) and any("Canvas height" in text for text in texts)
    assert len(refs) <= (10 if mixed else 8)
    if mixed:
        assert {number for proof in refs.values() for number in proof["image_numbers"]} == {1, 2}
    else:
        assert all(not proof["image_numbers"] for proof in refs.values())


def test_diagnostic_source_context_retains_confirmed_scope_without_copying_source_body():
    active = {"sources": [{"source": {"file_id": DOC, "filename": "Вход в установленное рабочее приложение.docx"},
                           "sections": [{"text": "forbidden-source-body" * 800}]}],
              "last_question": "Какой текст ошибки появился?"}
    original = copy.deepcopy(active)
    bounded = TalkKnowledge._instruction_context({"diagnostic_request": True, "active_guide_context": active,
                                                  "current_message": "Неверный пароль"})
    assert bounded["diagnostic_request"] is True
    assert bounded["active_guide_context"]["sources"][0]["file_id"] == DOC
    assert "установленное" in bounded["active_guide_context"]["sources"][0]["filename"]
    assert "forbidden-source-body" not in json.dumps(bounded)
    assert bounded["current_message"] == "Неверный пароль" and active == original


def test_adapter_omits_discovery_aliases_from_model_context_and_does_not_mutate_the_request():
    text = "Добавочный руководителя — 126."
    payload = {"requested_goals": {"g1": "Узнать добавочный руководителя"},
               "goal_search_queries": {"g1": ["supervisor extension reference"]},
               "catalogue": {DOC: "Справочник.md"}, "candidates": {DOC: text},
               "context": {"goal_search_queries": {"g1": ["supervisor extension reference"]},
                           "task_state": {"goals": {"g1": {"text": "Узнать добавочный руководителя",
                                                          "search_queries": ["supervisor extension reference"]}}}}}
    unchanged = copy.deepcopy(payload)

    def handler(request):
        wire = json.loads(json.loads(request.content)["messages"][-1]["content"])
        assert "goal_search_queries" not in wire and "goal_search_queries" not in wire["context"]
        assert "search_queries" not in wire["context"]["task_state"]["goals"]["g1"]
        assert "supervisor extension reference" not in json.dumps(wire, ensure_ascii=False)
        reference = next(iter(wire["candidates"]["s1"]))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "result": "MATCH", "file_id": "s1", "covered_goals": [{"goal_id": "g1", "evidence_id": reference}],
        })}}]})

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    result = model.select_instruction(payload)
    assert result.covered_goals[0].evidence == text
    assert payload == unchanged


@pytest.mark.parametrize("metadata", [
    {"requested_goal_kinds": {"unknown": "LOOKUP"}},
    {"requested_goal_kinds": {"g1": "OTHER"}},
    {"requested_goal_kinds": {"g1": []}},
    {"goal_search_queries": {"unknown": ["query"]}},
    {"goal_search_queries": {"g1": "query"}},
    {"goal_search_queries": {"g1": ["query"] * 4}},
    {"goal_search_queries": {"g1": ["x" * 161]}},
])
def test_optional_goal_metadata_is_bounded_and_scoped_to_requested_goal_ids(metadata):
    with pytest.raises(ValueError):
        instruction_goal_metadata(metadata, {"g1": "Цель"})
