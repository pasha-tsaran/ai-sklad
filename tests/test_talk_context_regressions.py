"""Source-scope rejection and employee context survive clarification turns."""

import copy
from types import SimpleNamespace

import pytest

from app.services.chat import talk_dialogue as chat
from app.services.chat.confirmation_scope import (
    confirmation_scope_entries, confirmation_scope_identity, render_confirmation_scopes,
)
from app.services.chat.task_state import remember_task_message, update_task_state
from app.services.rag.catalog import instruction_sections
from app.services.rag.talk import KnowledgeUnavailable
from tests.test_confirmation_scope import original


def state():
    return {"facts": [], "goals": {"g1": {"text": "Уменьшить громкость", "evidence": "тише",
                                            "kind": "ACTION"}},
            "messages": ["Настройка сканера", "Нужно тише"], "unknown_answers": [],
            "question": "", "question_field": None}


def guide(approved):
    return {"query": "Нужно тише", "opening_query": "Настройка сканера", "task_state": state(),
            "source": approved.document["source"], "excluded": [], "shown": [],
            "answered_questions": ["Что настроить?"],
            "accepted_instruction_offer": {"accepted_goals": {"g1": "Уменьшить громкость"}}}


def test_scope_identity_ignores_goal_paraphrase_and_plan_state():
    approved = original("# Настройка громкости\n\n1. Нажмите кнопку.")
    first = guide(approved)
    entries = confirmation_scope_entries(approved)
    changed = copy.deepcopy(first)
    changed["task_state"]["goals"]["g1"]["text"] = "Сделать сканер тише"
    changed["instruction_plan"] = {"parts": [{"cursor": 3}]}
    assert chat._confirmation_identity(first, entries) == chat._confirmation_identity(changed, entries)
    renamed = copy.deepcopy(entries)
    renamed[0]["source"]["filename"] = "Переименован.docx"
    assert confirmation_scope_identity(renamed) == confirmation_scope_identity(entries)


def test_method_change_in_same_file_has_a_different_identity():
    approved = original("# Подключение\n\n## Wi-Fi\n\n1. Откройте приложение.\n\n"
                        "## Bluetooth\n\n1. Выберите устройство.")
    sections = instruction_sections(approved.document)
    entries = []
    for title in ("Wi-Fi", "Bluetooth"):
        part = SimpleNamespace(approved=approved, sections=tuple(s for s in sections if title in s["title"]))
        entries.append(confirmation_scope_entries(approved, [part]))
    assert confirmation_scope_identity(entries[0]) != confirmation_scope_identity(entries[1])


def test_confirmation_correction_retains_user_state_from_nested_guide(monkeypatch):
    approved = original("# Подключение TSC\n\n1. Откройте приложение.")
    pending = chat._confirmation_turn(approved, guide(approved), "Подключить принтер").flow
    captured = []
    monkeypatch.setattr(chat, "_support_turn", lambda message, flow, *args: captured.append(flow) or flow)
    result = chat._understanding_turn("Нет, у меня Urovo по Bluetooth", pending, {}, SimpleNamespace(),
                                      SimpleNamespace(), SimpleNamespace())
    assert result is captured[0]
    assert result["task_state"] == state()
    assert result["opening_query"] == "Настройка сканера"
    assert result["answered_questions"] == ["Что настроить?"]
    assert result["accepted_instruction_offer"]["accepted_goals"]
    assert result["rejected_scope_entries"] == [pending["understanding_confirmation"]["scope_entries"]]
    assert result["reselect_source"]
    assert not {"source", "instruction_plan", "shown", "required_source"} & result.keys()


def test_bare_rejection_remembers_method_and_asks_once():
    approved = original("# Подключение TSC\n\n1. Откройте приложение.")
    pending = chat._confirmation_turn(approved, guide(approved), "Подключить принтер").flow
    turn = chat._understanding_turn("Нет", pending, {}, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert turn.blocks[0][0].count("?") == 1
    rejected = {**guide(approved), **turn.flow}
    repeated = chat._confirmation_turn(approved, rejected, "Другая формулировка той же задачи")
    assert repeated.action == "NO_KNOWLEDGE"
    assert not repeated.flow.get("clarification_question")


def test_legacy_rejection_hash_is_honored():
    approved = original("# Подключение TSC")
    flow = guide(approved)
    flow["rejected_scope_ids"] = [chat._legacy_confirmation_identity(flow, confirmation_scope_entries(approved))]
    assert chat._confirmation_turn(approved, flow, "Подключить принтер").action == "NO_KNOWLEDGE"


def test_source_volume_proof_uses_literal_operation_instead_of_long_caption_prefix():
    text = "# Настройки сканера\n\n## Назначение\n\n" + "Подсветка сканера бывает разной. " * 10
    text += "\n\nДоступные уровни громкости — тихий, средний и громкий.\n\n1. Отсканируйте код."
    approved = original(text)
    sections = instruction_sections(approved.document)
    operation = next(s for s in sections if "уровни громкости" in s["text"])
    part = SimpleNamespace(approved=approved, sections=tuple(sections), confirmation_evidence=({
        "section_id": operation["id"], "text": operation["text"], "support_kind": "TEXT", "image_numbers": [],
    },))
    entries = confirmation_scope_entries(approved, [part])
    assert any(item["text"] == operation["text"] for item in entries[0]["context"])
    assert "Отсканируйте" not in render_confirmation_scopes(entries)
    part.confirmation_evidence = ({"section_id": operation["id"], "text": "Несуществующее обещание",
                                   "support_kind": "TEXT", "image_numbers": []},)
    assert "Несуществующее" not in render_confirmation_scopes(confirmation_scope_entries(approved, [part]))


def test_operation_proof_keeps_actual_method_conditions_and_omits_a_truncated_tail():
    approved = original("# Настройка звука\n\n## Только в приложении Альфа\n\n"
                        "Громкость бывает высокой и низкой.\n\nСредний ряд задаёт громкость сканера.")
    sections = instruction_sections(approved.document)
    operation = next(s for s in sections if "высокой" in s["text"])
    truncated = next(s for s in sections if "Средний ряд" in s["text"])
    part = SimpleNamespace(approved=approved, sections=tuple(sections), confirmation_evidence=({
        "section_id": operation["id"], "text": operation["text"], "support_kind": "TEXT", "image_numbers": [],
    }, {"section_id": truncated["id"], "text": "Средний ряд задаёт", "support_kind": "TEXT",
        "image_numbers": []}))
    visible = render_confirmation_scopes(confirmation_scope_entries(approved, [part]))
    assert "Только в приложении Альфа" in visible
    assert "Громкость бывает высокой и низкой." in visible
    assert "Средний ряд задаёт»" not in visible


def test_normalized_middle_fragment_recovers_the_complete_original_volume_sentence():
    lighting = ("Верхний ряд меняет подсветку, отсканируйте код для выбора параметра, "
                "который показан на листе рядом с остальными вариантами настройки подсветки и "
                "другими подписями для этой модели устройства.")
    volume = ("Средний  ряд задаёт\nминимальную, среднюю или максимальную громкость; "
              "нижний — слабую, среднюю или сильную подсветку.")
    command = "Выполните настройку и отсканируйте следующий код после выбора нужного режима."
    caption = lighting + "\n  " + volume + "\n\n" + command
    approved = original("# Настройки сканера\n\n" + caption)
    approved.document["images"] = [{"number": 1, "caption": caption}]
    approved.document["layout"] = [{"text": "# Настройки сканера", "paragraph": 1, "image_numbers": []},
                                    {"text": caption, "paragraph": 2, "image_numbers": [1]}]
    sections = instruction_sections(approved.document)
    operation = next(section for section in sections if section["image_numbers"] == [1])
    normalized = " ".join(operation["text"].split())
    volume_start = normalized.index("Средний ряд")
    quote = normalized[volume_start - 90:volume_start + 90]
    assert len(quote) == 180 and quote not in operation["text"]
    assert not quote.endswith(".") and not quote.startswith("Средний")
    part = SimpleNamespace(approved=approved, sections=tuple(sections), confirmation_evidence=({
        "section_id": operation["id"], "text": quote, "support_kind": "IMAGE", "image_numbers": [1],
    },))
    entries = confirmation_scope_entries(approved, [part])
    assert entries[0]["context"] == [{"section_id": operation["id"], "text": " ".join(volume.split())}]
    visible = render_confirmation_scopes(entries)
    assert "громкость" in visible and "Средний ряд задаёт минимальную" in visible
    assert "отсканируйте" not in visible and "Выполните" not in visible
    assert approved.document["images"][0]["number"] == 1


def test_goal_kind_override_keeps_goal_identity_order_text_and_evidence():
    flow = {"task_state": state()}
    prior = copy.deepcopy(flow["task_state"])
    chat._apply_goal_kind_overrides(SimpleNamespace(goal_kind_overrides={"g1": "LOOKUP"}), flow)
    assert flow["task_state"]["goals"]["g1"] == {**prior["goals"]["g1"], "kind": "LOOKUP"}
    for override in ({"g8": "LOOKUP"}, {"g1": "OTHER"}, []):
        with pytest.raises(KnowledgeUnavailable, match="InvalidGoalKindOverride"):
            chat._apply_goal_kind_overrides(SimpleNamespace(goal_kind_overrides=override), flow)


def test_replacement_evidence_lasts_only_for_the_actual_update():
    message = "Вместо громкости нужно настроить подсветку"
    model = SimpleNamespace(understand_task=lambda *args: {
        "replace_goals": True, "replacement_evidence": "Вместо громкости",
        "goals": [{"text": "Настроить подсветку", "evidence": "настроить подсветку"}],
    })
    updated = update_task_state(message, state(), "", model)
    assert updated["goal_replacement_evidence"] == "Вместо громкости"
    assert "goal_replacement_evidence" not in remember_task_message("Да", updated)


def test_only_a_conflicting_known_fact_clears_rejected_methods():
    previous = {"facts": [{"key": "connection", "value": "Bluetooth", "evidence": "Bluetooth"}]}
    assert chat._corrected_known_fact(previous, {"facts": [{"key": "connection", "value": "Wi-Fi",
                                                           "evidence": "Wi-Fi"}]}, "На самом деле Wi-Fi")
    assert not chat._corrected_known_fact(previous, {"facts": [*previous["facts"],
        {"key": "symptoms", "value": "Мигает лампочка", "evidence": "Мигает лампочка"}]}, "Мигает лампочка")


def test_unknown_condition_retains_task_and_allows_a_different_observable_detail(monkeypatch):
    question = "Первый принтер подключён кабелем или без проводов?"
    pending = {"question": question, "clarification_condition": "PHYSICAL_CONNECTION",
               "goal_ids": ["g1"], "sources": []}
    flow = {"task_state": state(), "clarification_question": question,
            "pending_clarification": pending, "opening_query": "Настройка сканера"}
    seen = []
    def retrieve(message, retained, *args):
        seen.append(copy.deepcopy(retained))
        return chat.Turn("REPLY", [("Второй принтер подключён кабелем или без проводов?", None)], flow={})
    monkeypatch.setattr(chat, "_retrieve_support_turn", retrieve)
    knowledge = SimpleNamespace(settings=SimpleNamespace(talk_confirm_understanding=True))
    model = SimpleNamespace(understand_task=lambda *args: pytest.fail("Unknown answer must retain the task"))
    turn = chat._support_turn("Не знаю", flow, {}, model, knowledge)
    assert turn.flow["task_state"]["goals"] == state()["goals"]
    assert turn.flow["task_state"]["unknown_answers"][0]["question"] == question
    assert turn.flow["answered_conditions"] == [{**pending, "answer": "Не знаю"}]
    assert seen[0]["reselect_source"]


def test_typed_condition_does_not_block_another_device_question_and_forwards_history():
    first = "Первый принтер подключён кабелем или без проводов?"
    second = "Второй принтер подключён кабелем или без проводов?"
    calls = []
    decision = SimpleNamespace(result="CLARIFY", question=second, goal_kind_overrides={},
                               clarification_condition="PHYSICAL_CONNECTION", clarification_sources=())
    def search(query, **options):
        calls.append(options["context"])
        return decision
    knowledge = SimpleNamespace(uses_llm_selection=True, search_with_llm=search,
                                settings=SimpleNamespace(talk_confirm_understanding=True))
    flow = {"task_state": state(), "task_query": "Подключить принтеры", "answered_questions": [first],
            "answered_conditions": [{"question": first, "clarification_condition": "PHYSICAL_CONNECTION",
                                      "goal_ids": ["g1"], "sources": [], "answer": "Не знаю"}],
            "rejected_scope_ids": ["old"], "rejected_scope_entries": []}
    turn = chat._retrieve_support_turn("Не знаю", flow, {}, SimpleNamespace(), knowledge)
    assert turn.blocks == [(second, None)]
    assert len(calls) == 1
    assert turn.flow["pending_clarification"]["clarification_condition"] == "PHYSICAL_CONNECTION"
    assert calls[0]["short_term"]["messages"][0]["content"] == "Настройка сканера"
    assert calls[0]["answered_conditions"] == flow["answered_conditions"]


@pytest.mark.parametrize("message,delta,clears", [
    ("Нужно сделать тише", {"goals": [{"goal_id": "g1", "text": "Сделать сканер тише",
                                       "evidence": "Нужно сделать тише"}]}, False),
    ("На самом деле Wi-Fi", {"facts": [{"key": "connection", "value": "Wi-Fi", "evidence": "Wi-Fi"}]}, True),
    ("Теперь горит лампа", {"facts": [{"key": "symptoms", "value": "горит лампа",
                                      "evidence": "горит лампа"}]}, False),
    ("Вместо громкости настроить подсветку", {"replace_goals": True,
        "replacement_evidence": "Вместо громкости", "goals": [{"text": "Настроить подсветку",
                                                               "evidence": "настроить подсветку"}]}, True),
    ("Другая задача. Открыть почту", {"goals": [{"text": "Открыть почту", "evidence": "Открыть почту"}]}, True),
])
def test_normal_task_updates_clear_rejections_only_for_changed_purpose_or_conflicting_fact(
        monkeypatch, message, delta, clears):
    previous = state()
    previous["facts"] = [{"key": "connection", "value": "Bluetooth", "evidence": "Bluetooth"}]
    flow = {"task_state": previous, "rejected_scope_ids": ["scope"], "rejected_scope_entries": [[]],
            "opening_query": "Настройка сканера",
            "accepted_instruction_offer": {"accepted_goals": {"g1": "Уменьшить громкость"}}}
    monkeypatch.setattr(chat, "_retrieve_support_turn",
                        lambda message, retained, *args: chat.Turn("REPLY", [("Проверено", None)], flow=retained))
    model = SimpleNamespace(understand_task=lambda *args: delta)
    knowledge = SimpleNamespace(settings=SimpleNamespace(talk_confirm_understanding=True))
    turn = chat._support_turn(message, flow, {}, model, knowledge)
    assert bool(turn.flow.get("rejected_scope_ids")) is not clears
    assert bool(turn.flow.get("rejected_scope_entries")) is not clears
    if not clears:
        assert turn.flow["opening_query"] == "Настройка сканера"
        assert turn.flow["accepted_instruction_offer"]["accepted_goals"]

