"""Private dialogue evaluation has independent source, condition and image oracles."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.llm.mock import MockLLM
from app.services.chat.talk_dialogue import SELECTION_UNCLEAR
from integrations.openwebui.approved_instruction_images import read_docx_images
from scripts.evaluate_instruction_dialogues import (
    LOCAL_COLLECTION, ConfirmationDecision, DialogueCase, EmployeeBridge, EvaluationSession, ImageAnchor, LayDecision,
    LocalOriginals, TaskCondition,
    check_visible_scopes, condition_matches, grade_output, load_cases, next_user_reply, ordered_groups_present,
    preconsent_action_in_text, repeated_long_paragraphs, run_variant, write_transcript,
)
from tests.test_openwebui_instruction_images import instruction


SOURCE = "a" * 64
OTHER = "b" * 64


def test_duplicate_paragraph_audit_requires_distinct_messages_and_keeps_the_text_private():
    paragraph = "Исходное описание проверочного действия. " * 4
    assert repeated_long_paragraphs([{"text": paragraph, "message_id": "same"},
                                     {"text": paragraph, "message_id": "same"}]) == []
    repeated = repeated_long_paragraphs([{"text": paragraph, "message_id": "first"},
                                         {"text": paragraph, "message_id": "second"},
                                         {"text": "Короткий вопрос", "message_id": "third"}])
    assert len(repeated) == 1 and repeated[0]["message_ids"] == ["first", "second"]
    assert set(repeated[0]) == {"sha256", "characters", "message_ids"}


def case(**expected):
    return DialogueCase.model_validate({
        "id": "example", "target_source": {"filename": "Инструкция.docx", "sha256": SOURCE},
        "variants": [{"id": "plain", "opening": "Помогите разобраться"}],
        "scenario": {"goal": "Сделать сигнал тише", "observable_facts": ["Это считыватель товарных кодов."]},
        "expected": {"outcomes": ["GUIDE"], "allowed_sources": [{"filename": "Инструкция.docx", "sha256": SOURCE}],
                     "required_quotes": ["Выберите тихий режим"], **expected},
    })


def block(text="Выберите тихий режим", *, source=SOURCE, image=None):
    return {"text": text, "source": {"sha256": source, "filename": "Инструкция.docx"}, "image_sha256": image}


def grade(chosen, blocks, **values):
    return grade_output(chosen, blocks, values.get("flow", {}), confirmed=values.get("confirmed", True),
                        clarified=values.get("clarified", False), variant=chosen.variants[0],
                        image_digest=lambda anchor: (anchor.source_sha256, str(anchor.number)))


def test_same_filename_does_not_authorize_a_different_source_version():
    failures, _ = grade(case(), [block(source=OTHER)])
    assert "UnrequestedInstruction" in failures


@pytest.mark.parametrize("failure,blocks,settings", [
    ("RequiredActionMissing", [block("Настройка оборудования")], {}),
    ("AnswerBeforeConfirmation", [block()], {"confirmed": False}),
    ("ForbiddenActionDelivered", [block("Выберите тихий режим. Удалите настройку программы.")], {}),
])
def test_final_grade_checks_actions_and_prior_consent(failure, blocks, settings):
    failures, _ = grade(case(forbidden_quotes=["Удалите настройку программы"]), blocks, **settings)
    assert failure in failures


def test_ordered_photo_oracle_preserves_repeated_identical_positions():
    assert ordered_groups_present([(SOURCE, "same"), (SOURCE, "same")],
                                  [[(SOURCE, "same")], [(SOURCE, "same")]])
    assert not ordered_groups_present([(SOURCE, "same")], [[(SOURCE, "same")], [(SOURCE, "same")]])
    assert not ordered_groups_present([(SOURCE, "2"), (SOURCE, "1")], [[(SOURCE, "1")], [(SOURCE, "2")]])


def test_required_photo_checks_source_identity_and_original_ordinal():
    chosen = case(required_images=[{"source_sha256": SOURCE, "number": 1},
                                   {"source_sha256": SOURCE, "number": 2}])
    failures, _ = grade(chosen, [block(image="1"), block(image="2")])
    assert not failures
    failures, _ = grade(chosen, [block(image="2"), block(image="1")])
    assert "RequiredOriginalPhotographsMissingOrReordered" in failures


def test_an_extra_picture_from_the_same_document_is_not_automatically_permitted():
    chosen = case(allowed_images=[{"source_sha256": SOURCE, "number": 1}])
    failures, _ = grade(chosen, [block(image="1"), block(image="2")])
    assert "UnrequestedOriginalPhotograph" in failures


def test_confirmation_and_clarification_have_separate_closed_choices():
    assert ConfirmationDecision.model_json_schema()["properties"]["choice"]["enum"] == ["YES", "NO"]
    assert LayDecision.model_json_schema()["properties"]["choice"]["enum"] == ["FACTS", "GOAL", "UNKNOWN"]
    with pytest.raises(ValueError):
        ConfirmationDecision(choice="GOAL", correction_kind="NONE")


def test_scope_oracle_rejects_a_generic_summary_hiding_the_selected_application():
    from app.services.chat.talk_dialogue import _confirmation_turn
    from tests.test_confirmation_scope import flow, original

    approved = original("# Вход в почту webmail\n\n## Вход в корпоративную почту через браузер\n\n1. Откройте сайт.")
    turn = _confirmation_turn(approved, flow(approved), "Открыть почту")
    confirmation = turn.flow["understanding_confirmation"]
    assert check_visible_scopes(confirmation, [turn.blocks[0][0]], lambda file_id: approved) == 1
    with pytest.raises(ValueError, match="InstructionScopeNotVisible"):
        check_visible_scopes(confirmation, ["Я понял задачу так — Открыть почту. Всё верно?"], lambda file_id: approved)


def test_scope_oracle_checks_original_context_and_all_selected_sources():
    from app.services.chat.talk_dialogue import _confirmation_turn
    from tests.test_confirmation_scope import flow, original

    approved = original("# Печать с ТСД\n\n## Сценарий\n\nДва принтера по Wi-Fi.\n\n1. Откройте приложение.")
    turn = _confirmation_turn(approved, flow(approved), "Настроить печать")
    confirmation = turn.flow["understanding_confirmation"]
    forged = copy.deepcopy(confirmation)
    forged["scope_entries"][0]["context"][0]["text"] = "Один принтер по Bluetooth"
    with pytest.raises(ValueError, match="InstructionScopeContextNotOriginal"):
        check_visible_scopes(forged, [turn.blocks[0][0]], lambda file_id: approved)
    confirmation["guide_flow"]["instruction_plan"] = {"parts": [
        {"source": approved.document["source"]}, {"source": original("# Другое", identity="b").document["source"]},
    ]}
    with pytest.raises(ValueError, match="InstructionScopeSourceMissing"):
        check_visible_scopes(confirmation, [turn.blocks[0][0]], lambda file_id: approved)


def test_scope_oracle_requires_the_selected_conditional_child_prefix():
    from app.services.chat.instruction_plan import serialize_plan
    from app.services.chat.talk_dialogue import _confirmation_turn
    from app.services.rag.catalog import instruction_sections
    from tests.test_confirmation_scope import flow, original

    approved = original("# Печать PDF\n\n1. Проверьте принтер.\n\nЕсли картинки не печатаются, "
                        "отдельный способ: «Дополнительно → Печатать как изображение → ОК».")
    part = SimpleNamespace(approved=approved, sections=tuple(instruction_sections(approved.document)),
                           goal_ids=("g1",), goals=("Напечатать картинки",), delivery_mode="PROCEDURE")
    guide = {**flow(approved), "instruction_plan": serialize_plan([part])}
    turn = _confirmation_turn(approved, guide, "Напечатать картинки", parts=[part])
    confirmation = turn.flow["understanding_confirmation"]
    assert check_visible_scopes(confirmation, [turn.blocks[0][0]], lambda file_id: approved) == 1
    confirmation["scope_entries"][0]["context"] = []
    with pytest.raises(ValueError, match="InstructionConditionalScopeNotDisclosed"):
        check_visible_scopes(confirmation, [turn.blocks[0][0]], lambda file_id: approved)


def test_employee_bridge_uses_independent_primitive_decisions(tmp_path, monkeypatch):
    def respond(_):
        request_path, = tmp_path.glob("*.request.json")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert request["payload"]["employee_messages"] == ["Сделайте тише"]
        assert request["response_schema"]["properties"]["choice"]["enum"] == ["YES", "NO"]
        request_path.with_name(request["request_id"] + ".response.json").write_text(json.dumps({
            "request_id": request["request_id"], "decision": {"choice": "YES", "correction_kind": "NONE"},
        }), encoding="utf-8")

    monkeypatch.setattr("scripts.evaluate_instruction_dialogues.time.sleep", respond)
    bridge = EmployeeBridge(tmp_path)
    reply = bridge.decide(ConfirmationDecision, {"employee_messages": ["Сделайте тише"]})
    assert reply.choice == "YES" and bridge.calls == 1


def test_employee_bridge_rejects_a_response_for_another_request(tmp_path, monkeypatch):
    def respond(_):
        request_path, = tmp_path.glob("*.request.json")
        request_path.with_name(request_path.name.replace(".request.", ".response.")).write_text(json.dumps({
            "request_id": "different", "decision": {"choice": "YES", "correction_kind": "NONE"},
        }), encoding="utf-8")

    monkeypatch.setattr("scripts.evaluate_instruction_dialogues.time.sleep", respond)
    with pytest.raises(ValueError, match="EmployeeBridgeResponseMismatch"):
        EmployeeBridge(tmp_path).decide(ConfirmationDecision, {})


def test_equivalent_approved_alternative_can_have_its_own_quote_and_image():
    chosen = case(allowed_sources=[{"filename": "A.docx", "sha256": SOURCE},
                                  {"filename": "B.docx", "sha256": OTHER}], required_quotes=[],
                  required_quote_groups=[["Выберите тихий режим", "Отключите громкий сигнал"]],
                  required_image_groups=[[{"source_sha256": SOURCE, "number": 1},
                                          {"source_sha256": OTHER, "number": 2}]])
    failures, _ = grade(chosen, [block("Отключите громкий сигнал", source=OTHER, image="2")])
    assert not failures


def test_negative_connection_is_not_proven_by_a_positive_fact_with_the_same_quote():
    condition = TaskCondition(values=["Bluetooth"], negated=True)
    bad = {"key": "connection", "value": "Bluetooth", "evidence": "Блютуз не используется"}
    assert not condition_matches(condition, [bad])
    assert condition_matches(condition, [{**bad, "negated": True}])
    chosen = case(task_conditions=[condition.model_dump()])
    failures, _ = grade(chosen, [block()], flow={"task_state": {"facts": [bad]}})
    assert "TaskConditionMissingOrContradicted" in failures


@pytest.mark.parametrize("goals", [{}, {"g1": {"kind": "ACTION"}}, {"g1": {"text": "Узнать значение"}},
                                  {"g1": {"kind": "LOOKUP"}, "g2": {"kind": "ACTION"}}])
def test_factual_answer_requires_lookup_intent_even_when_its_source_and_quote_match(goals):
    chosen = case(goal_kind="LOOKUP")
    failures, _ = grade(chosen, [block()], flow={"task_state": {"goals": goals}})
    assert "TaskGoalKindMismatch" in failures


def test_matching_explicit_lookup_intents_pass_the_goal_kind_oracle():
    chosen = case(goal_kind="LOOKUP")
    failures, _ = grade(chosen, [block()], flow={"task_state": {"goals": {
        "g1": {"kind": "LOOKUP"}, "g2": {"kind": "LOOKUP"},
    }}})
    assert not failures


def test_goal_kind_oracle_does_not_reject_an_honest_absence_without_a_technical_answer():
    chosen = case(goal_kind="LOOKUP", outcomes=["NO_KNOWLEDGE"])
    failures, outcome = grade(chosen, [], flow={"escalation": True})
    assert outcome == "NO_KNOWLEDGE" and "TaskGoalKindMismatch" not in failures


def test_connection_must_belong_to_the_correct_device():
    condition = TaskCondition(key="connection", values=["Wi-Fi"], entity_values=["первый принтер"])
    assert not condition_matches(condition, [{"key": "connection", "value": "Wi-Fi", "entity": "второй принтер"}])
    assert condition_matches(condition, [{"key": "connection", "value": "Wi-Fi", "entity": "первый принтер"}])


def test_user_simulator_chooses_existing_facts_without_generating_new_values():
    selected = []

    class Model:
        def decide(self, schema, payload):
            selected.append(copy.deepcopy(payload))
            return LayDecision(choice="FACTS", fact_ids=[0])

    result = {"flow": {"clarification_question": "Какое устройство перед вами?"}, "texts": []}
    message, issue, indices = next_user_reply(Model(), case(), result, used_facts=set())
    assert message == "Это считыватель товарных кодов." and issue is None and indices == [0]
    assert selected[0]["observable_facts"] == {"0": "Это считыватель товарных кодов."}
    _, issue, _ = next_user_reply(Model(), case(), result, used_facts={0})
    assert issue == "RepeatedKnownConditionQuestion"


def test_technical_question_is_counted_as_a_quality_failure_and_gets_unknown():
    model = SimpleNamespace(decide=lambda *args, **kwargs: pytest.fail("No technical answer may be invented"))
    result = {"flow": {"clarification_question": "Какой протокол и модель оборудования?"}, "texts": []}
    message, issue, _ = next_user_reply(model, case(), result, used_facts=set())
    assert message.startswith("Не знаю") and issue == "TechnicalKnowledgeQuestion"


def test_known_condition_from_the_opening_is_audited_without_a_prior_fact_selection():
    model = SimpleNamespace(decide=lambda *args: LayDecision(
        choice="FACTS", fact_ids=[0], repeats_known_condition=True,
    ))
    result = {"flow": {"clarification_question": "Какое устройство перед вами?"}, "texts": [],
              "evaluation_employee_messages": ["Это считыватель товарных кодов."]}
    _, issue, _ = next_user_reply(model, case(), result, used_facts=set())
    assert issue == "RepeatedKnownConditionQuestion"


@pytest.mark.parametrize("text", [
    "Вы уже пробовали нажать Ctrl+Alt+End и выбрать смену пользователя?",
    "Нажмите кнопку запуска. Какое окно открылось?",
    "На компьютере можно открыть меню. Откройте настройки устройства.",
    "Что появляется после Win+R?",
])
def test_new_actions_cannot_hide_in_source_less_questions_before_consent(text):
    assert preconsent_action_in_text([text], ["Хочу войти под своим пользователем."])


@pytest.mark.parametrize("text", [
    "Это компьютер или телефон?",
    "Я понял задачу так — войти в рабочую программу. Правильно я понял?",
    "Что произошло после Ctrl+Alt+End?",
])
def test_observational_questions_and_already_reported_actions_are_available(text):
    assert not preconsent_action_in_text([text], ["Я нажал Ctrl + Alt + End."])


def test_wrong_summary_is_corrected_instead_of_blindly_confirmed():
    model = SimpleNamespace(decide=lambda *args, **kwargs: ConfirmationDecision(choice="NO",
                                                                             correction_kind="CONTRADICTS_KNOWN"))
    result = {"flow": {"understanding_confirmation": {"summary": "Удалить устройство из программы"}}, "texts": []}
    message, issue, _ = next_user_reply(model, case(), result, used_facts=set())
    assert message.startswith("Нет.") and "Сделать сигнал тише" in message and issue == "WrongUnderstanding"


def test_newly_disclosed_condition_is_a_recorded_hypothesis_correction():
    model = SimpleNamespace(decide=lambda *args, **kwargs: ConfirmationDecision(choice="NO",
                                                                             correction_kind="MISSING_DETAILS"))
    result = {"flow": {"understanding_confirmation": {"summary": "Изменить подсветку"}}, "texts": []}
    message, issue, _ = next_user_reply(model, case(), result, used_facts=set())
    assert message.startswith("Нет.") and issue == "HiddenConditionHypothesisCorrected"


def test_empty_summary_cannot_receive_automatic_consent():
    model = SimpleNamespace(decide=lambda *args, **kwargs: pytest.fail("Empty summary is not consentable"))
    result = {"flow": {"understanding_confirmation": {"summary": ""}}, "texts": []}
    with pytest.raises(ValueError, match="EmptyUnderstandingSummary"):
        next_user_reply(model, case(), result, used_facts=set())


def test_source_less_text_cannot_hide_a_forbidden_action():
    blocks = [block(), {"text": "Удалите настройку программы.", "source": None, "image_sha256": None}]
    failures, _ = grade(case(forbidden_quotes=["Удалите настройку программы"]), blocks)
    assert "ForbiddenActionDelivered" in failures


def test_arbitrary_generic_question_cannot_pass_a_source_grounded_clarification_case():
    chosen = case(outcomes=["CLARIFY"], required_quotes=[], required_clarification_quote_groups=[["параметры", "ИТ"]])
    failures, _ = grade(chosen, [{"text": "Что именно нужно сделать?", "source": None}])
    assert "UngroundedClarification" in failures and "RequiredClarificationBoundaryMissing" in failures
    failures, outcome = grade_output(chosen, [{"text": "Есть ли актуальные параметры от ИТ?", "source": None}], {},
                                      confirmed=False, clarified=True, variant=chosen.variants[0],
                                      image_digest=lambda anchor: (SOURCE, "image"),
                                      clarification_sources=[{"sha256": SOURCE}])
    assert outcome == "CLARIFY" and not failures


def test_load_cases_rejects_reused_case_and_variant_ids(tmp_path):
    path = tmp_path / "private-cases.json"
    original = case().model_dump()
    path.write_text(json.dumps({"version": 1, "cases": [original, original]}), encoding="utf-8")
    with pytest.raises(ValueError, match="InvalidEvaluationCaseIds"):
        load_cases(path)
    original["variants"] *= 2
    path.write_text(json.dumps({"version": 1, "cases": [original]}), encoding="utf-8")
    with pytest.raises(ValueError, match="InvalidEvaluationVariantIds"):
        load_cases(path)


def test_transcript_contains_original_immutable_image_asset(tmp_path):
    data = b"original bytes"
    digest = hashlib.sha256(data).hexdigest()
    target = tmp_path / "transcript.md"
    write_transcript(target, {"case": "a", "variant": "b"}, [
        {"role": "assistant", "text": "Исходное фото", "image_sha256": digest},
    ], lambda wanted: data if wanted == digest else pytest.fail("Wrong image"))
    assert (tmp_path / "assets" / (digest + ".png")).read_bytes() == data
    assert "![Исходная фотография]" in target.read_text(encoding="utf-8")


def test_production_controller_and_recorded_sender_remain_isolated(tmp_path):
    (tmp_path / "instruction.md").write_text("Описание программы", encoding="utf-8")
    originals = LocalOriginals(tmp_path)
    settings = Settings(_env_file=None, nextcloud_base_url="https://talk.example.invalid",
                        openwebui_api_key="test-key", openwebui_knowledge_ids=[
                            "00000000-0000-4000-8000-000000000001",
                        ])
    session = EvaluationSession(settings, source_transport=originals.transport)
    session.model = MockLLM()
    try:
        session.start_case()
        room = session.room_id
        result = session.turn("Привет")
        assert result["texts"] and all(row["status"] == "SENT" for row in result["deliveries"])
        assert session.sender.writes and str(session.engine.url).startswith("sqlite:///")
        session.start_case()
        assert session.room_id != room and session.sequence == 0 and session.sender.writes == []
    finally:
        session.close()


def test_expected_image_anchor_has_positive_original_ordinal():
    with pytest.raises(ValueError):
        ImageAnchor(source_sha256=SOURCE, number=0)


def test_original_photo_passes_the_production_outbox_and_recording_transport(tmp_path):
    caption = "Сканер штрихкодов. Для уменьшения громкости используйте настроечный штрихкод на исходном листе."
    data = instruction([caption])
    (tmp_path / "37_Параметры сканера штрихкодов.docx").write_bytes(data)
    originals = LocalOriginals(tmp_path)
    settings = Settings(_env_file=None, nextcloud_base_url="https://talk.example.invalid",
                        openwebui_api_key="test-key", openwebui_knowledge_ids=[LOCAL_COLLECTION])
    session = EvaluationSession(settings, source_transport=originals.transport)
    session.model = MockLLM()
    try:
        assert session.prepare()["complete"]
        session.start_case()
        first = session.turn("У сканера штрихкодов нужно уменьшить громкость.")
        assert first["flow"].get("understanding_confirmation")
        delivered = session.turn("Да, правильно")
        original = read_docx_images(data)[0]
        wanted = hashlib.sha256(original.data).hexdigest()
        assert [photo["sha256"] for photo in delivered["photos"]] == [wanted]
        assert [write["image_sha256"] for write in session.sender.writes if write["image_sha256"]] == [wanted]
        assert [block["image_numbers"] for block in delivered["blocks"] if block["image_sha256"]] == [[1]]
        assert session.image_bytes(wanted) == original.data
    finally:
        session.close()


def test_missing_current_source_is_reported_per_variant_without_a_false_pass():
    session = SimpleNamespace(start_case=lambda: None, inventory={},
                              model=SimpleNamespace(calls=0), user_model=SimpleNamespace(calls=0))
    metadata, transcript = run_variant(session, case(), case().variants[0])
    assert metadata["status"] == "error" and metadata["issues"] == ["CorpusSourceMissingOrChanged"]
    assert transcript == []


def test_selection_failure_is_not_accepted_as_an_ordinary_clarifying_question():
    session = SimpleNamespace(
        start_case=lambda: None, inventory={SOURCE: {}}, model=SimpleNamespace(calls=0),
        user_model=SimpleNamespace(calls=0), image_digest=lambda anchor: (SOURCE, "image"),
        turn=lambda message: {"flow": {}, "texts": [SELECTION_UNCLEAR],
                              "blocks": [{"text": SELECTION_UNCLEAR, "source": None}]},
    )
    metadata, _ = run_variant(session, case(), case().variants[0])
    assert metadata["status"] == "error" and metadata["issues"] == ["ControllerServiceError"]


def test_generic_work_advice_cannot_recover_later_and_be_reported_as_a_pass():
    session = SimpleNamespace(
        start_case=lambda: None, inventory={SOURCE: {}}, model=SimpleNamespace(calls=0),
        user_model=SimpleNamespace(calls=0), image_digest=lambda anchor: (SOURCE, "image"),
        turn=lambda message: {"flow": {}, "texts": ["Попробуйте открыть настройки и проверить подключение."],
                              "blocks": [{"text": "Попробуйте открыть настройки и проверить подключение.", "source": None}]},
    )
    metadata, _ = run_variant(session, case(), case().variants[0])
    assert metadata["status"] == "error" and metadata["issues"] == ["UnstructuredWorkReply"]


def test_continuous_mode_preserves_the_existing_test_room():
    session = SimpleNamespace(start_case=lambda: pytest.fail("Continuous mode must keep its room"), inventory={},
                              model=SimpleNamespace(calls=0), user_model=SimpleNamespace(calls=0))
    metadata, _ = run_variant(session, case(), case().variants[0], reset_room=False)
    assert metadata["issues"] == ["CorpusSourceMissingOrChanged"]


def test_rejected_later_summary_revokes_the_earlier_consent():
    replies = [
        {"flow": {"understanding_confirmation": {"summary": "Сделать сигнал тише"}},
         "texts": ["Верно ли понял?"], "blocks": []},
        {"flow": {"understanding_confirmation": {"summary": "Удалить запись программы"}},
         "texts": ["Верно ли понял?"], "blocks": []},
        {"flow": {}, "texts": ["Выберите тихий режим"], "blocks": [block()]},
    ]
    decisions = [ConfirmationDecision(choice="YES", correction_kind="NONE"),
                 ConfirmationDecision(choice="NO", correction_kind="CONTRADICTS_KNOWN")]
    session = SimpleNamespace(
        start_case=lambda: None, inventory={SOURCE: {}}, model=SimpleNamespace(calls=0),
        user_model=SimpleNamespace(calls=0, decide=lambda *args, **kwargs: decisions.pop(0)),
        turn=lambda message: copy.deepcopy(replies.pop(0)), image_digest=lambda anchor: (SOURCE, "image"),
    )
    metadata, _ = run_variant(session, case(), case().variants[0])
    assert metadata["confirmed"] is False
    assert "WrongUnderstanding" in metadata["issues"] and "AnswerBeforeConfirmation" in metadata["issues"]


def test_two_identical_rejected_hypotheses_stop_without_ten_model_turns():
    turns = []

    def turn(message):
        turns.append(message)
        return {"flow": {"understanding_confirmation": {"summary": "Другая цель"}}, "texts": [], "blocks": []}

    session = SimpleNamespace(
        start_case=lambda: None, inventory={SOURCE: {}}, model=SimpleNamespace(calls=0),
        user_model=SimpleNamespace(calls=0, decide=lambda *args: ConfirmationDecision(
            choice="NO", correction_kind="CONTRADICTS_KNOWN",
        )), turn=turn, image_digest=lambda anchor: (SOURCE, "image"),
    )
    metadata, _ = run_variant(session, case(), case().variants[0])
    assert len(turns) == 2 and "RepeatedRejectedUnderstanding" in metadata["issues"]


def test_unknown_reply_cannot_be_followed_by_the_same_field_reworded():
    replies = [
        {"flow": {"clarification_question": "В какой программе?", "task_state": {
            "question_field": "application", "goals": {"g1": {"text": "Сделать сигнал тише"}},
        }}, "texts": [], "blocks": []},
        {"flow": {"clarification_question": "Как называется открытое приложение?", "task_state": {
            "question_field": "application", "goals": {"g1": {"text": "Сделать сигнал тише"}},
        }}, "texts": [], "blocks": []},
        {"flow": {"task_state": {"goals": {"g1": {"text": "Сделать сигнал тише"}}}},
         "texts": ["Выберите тихий режим"], "blocks": [block()]},
    ]
    session = SimpleNamespace(
        start_case=lambda: None, inventory={SOURCE: {}}, model=SimpleNamespace(calls=0),
        user_model=SimpleNamespace(calls=0, decide=lambda *args: LayDecision(choice="UNKNOWN", fact_ids=[])),
        turn=lambda message: copy.deepcopy(replies.pop(0)), image_digest=lambda anchor: (SOURCE, "image"),
    )
    metadata, _ = run_variant(session, case(require_confirmation=False), case().variants[0])
    assert "RepeatedUnknownQuestionField" in metadata["issues"]


def test_unknown_reply_preserves_an_already_stated_task():
    replies = [
        {"flow": {"clarification_question": "В какой программе?", "task_state": {
            "question_field": "application", "goals": {"g1": {"text": "Сделать сигнал тише"}},
        }}, "texts": [], "blocks": []},
        {"flow": {}, "texts": ["Выберите тихий режим"], "blocks": [block()]},
    ]
    session = SimpleNamespace(
        start_case=lambda: None, inventory={SOURCE: {}}, model=SimpleNamespace(calls=0),
        user_model=SimpleNamespace(calls=0, decide=lambda *args: LayDecision(choice="UNKNOWN", fact_ids=[])),
        turn=lambda message: copy.deepcopy(replies.pop(0)), image_digest=lambda anchor: (SOURCE, "image"),
    )
    metadata, _ = run_variant(session, case(require_confirmation=False), case().variants[0])
    assert "TaskGoalLostAfterUnknownAnswer" in metadata["issues"]
