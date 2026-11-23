import copy
import json
from time import perf_counter

import httpx
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from app.core.logging import event
from app.schemas.contracts import (
    AnalysisResult, AnswerResult, DialogueResult, EscalationExtraction, FollowupResult, InstructionOfferReply, SummaryResult,
    SupportTaskUnderstanding, UnderstandingReply,
)
from app.services.chat.task_state import is_unknown_answer, task_evidence_authority
from app.services.chat.task_evidence import build_user_evidence, constrain_user_evidence, resolve_user_evidence
from app.services.llm.prompts import (
    ANALYZE_TASK, COMPOSE_TASK, DIALOGUE_TASK, ESCALATION_EXTRACTION_TASK, FOLLOWUP_TASK,
    INSTRUCTION_OFFER_TASK, SUMMARY_TASK, SUPPORT_TASK_TASK, UNDERSTANDING_REPLY_TASK, stage_messages,
)
from app.services.llm.instruction_selection import (
    InstructionReferenceSelection, instruction_goal_metadata, reference_selection_schema, resolve_instruction_references,
)


_TASK_REQUIRED_FIELDS = ("new_task", "facts", "goals", "question", "question_field")


def _instruction_contract_repair(error, stage):
    """Expose stable backend codes without model input or validation-error payloads."""
    code = error.args[0] if type(error) is ValueError and len(error.args) == 1 else ""
    if code not in {
        "InvalidInstructionReadingGoals", "UnknownInstructionReadingEvidence", "InvalidInstructionReadingMode",
        "MissingInstructionReadingEvidence", "UnsafeInstructionReadingQuestion", "InvalidInstructionReadingQuestion",
        "UnexpectedInstructionReadingQuestion", "UnexpectedInstructionReadingMode", "UnexpectedInstructionReadingEvidence",
        "InvalidInstructionVerifiedScope", "IncompleteInstructionVerificationContract",
        "UnknownInstructionVerificationEvidence", "InstructionVerificationEvidenceTooLarge",
        "MissingInstructionVerificationOriginalImage", "InvalidInstructionVerificationQuestion",
        "UnexpectedInstructionVerificationQuestion", "IncompleteInstructionVerificationAnswerPlan",
        "UnexpectedInstructionVerificationAnswerPlan", "ContradictoryInstructionVerification",
        "InvalidInstructionVerificationIllustration", "InstructionVerificationAnswerTooLarge",
        "UnknownRelatedInstructionEvidence", "InvalidRelatedInstructionScope",
    }:
        code = "InvalidInstruction" + stage + "Format"
    guidance = {
        "InvalidInstructionVerificationIllustration":
            "В illustration_refs выбирай только ID реально доступных исходных IMAGE из approved_original.",
        "InstructionVerificationAnswerTooLarge":
            "Общий список answer_refs и illustration_refs после удаления повторов должен содержать не более 96 ID.",
        "ContradictoryInstructionVerification":
            "EXACT операция, EXACT условия и полная готовность READY/true требуют PASS и полного плана ссылок. "
            "Для MISSING отрази действительную неподтверждённую операцию, несовместимые условия или отсутствие материалов.",
        "MissingInstructionVerificationOriginalImage":
            "Для PASS с ORIGINAL_IMAGE выбери ID самого доступного исходного изображения в material_refs. "
            "Если назначение или изображение отсутствует, отрази фактическое отсутствие материала.",
        "IncompleteInstructionVerificationAnswerPlan":
            "Независимо выбери все нужные исходные разделы в answer_refs и доказательство операции в operation_refs.",
        "UnknownInstructionVerificationEvidence":
            "Выбирай только различные ID из approved_original в каждой отдельной роли.",
        "InvalidInstructionVerificationQuestion":
            "Уточняй одно неизвестное наблюдаемое условие только для EXACT операции с готовыми материалами.",
        "InvalidInstructionReadingMode": "Согласуй режим с видом фактической задачи и статусом.",
    }.get(code, "Согласуй обязательные поля, статус и конечные исходные ID по переданной схеме.")
    return code + ". " + guidance


def _instruction_schema_variants(base, payload):
    """Constrain each outcome independently before accepting its source evidence."""
    if payload.get("evidence_references") is True:
        return reference_selection_schema(base, payload)
    goals = list(payload.get("requested_goals", {}))
    if not goals:
        return base
    properties = base["properties"]
    outcomes = set(properties["result"]["enum"])
    if payload.get("read_available") is False:
        outcomes.discard("READ")
    if payload.get("allow_clarification") is False:
        outcomes.discard("CLARIFY")
    definitions = copy.deepcopy(base.get("$defs", {}))
    loaded = list(payload.get("candidates", {}))
    catalogue = list(payload.get("catalogue", {}))
    goal_count = len(goals)

    def closed(fields, required):
        return {"type": "object", "properties": fields, "required": required, "additionalProperties": False}

    coverage = definitions["InstructionGoalCoverage"]["properties"]
    goal_id = {**coverage["goal_id"], "enum": goals}
    text = {"goal_id": goal_id, "evidence": {**coverage["evidence"], "minLength": 20},
            "support_kind": {"type": "string", "const": "TEXT"},
            "image_numbers": {**coverage["image_numbers"], "maxItems": 0}}
    image = {**text, "support_kind": {"type": "string", "const": "IMAGE"},
             "image_numbers": {**coverage["image_numbers"], "minItems": 1}}
    definitions["InstructionGoalCoverage"] = {"anyOf": [
        closed(text, ["goal_id", "evidence", "support_kind"]),
        closed(image, ["goal_id", "evidence", "support_kind", "image_numbers"]),
    ]}
    if loaded:
        definitions["InstructionAlternative"]["properties"]["file_id"] = {"type": "string", "enum": loaded}
        definitions["InstructionAlternative"]["properties"]["evidence"]["minLength"] = 20
    common = {name: properties[name] for name in ("understanding", "new_request") if name in properties}
    branches = []

    def add(result, fields, required):
        if result in outcomes:
            branches.append(closed({"result": {"type": "string", "const": result}, **fields},
                                   ["result", *required]))

    if loaded:
        file_id = {"type": "string", "enum": loaded}
        covered = {**properties["covered_goals"], "minItems": goal_count, "maxItems": goal_count}
        add("MATCH", {"file_id": file_id, "covered_goals": covered, **common}, ["file_id", "covered_goals"])
        if goal_count > 1:
            add("PARTIAL", {"file_id": file_id,
                            "covered_goals": {**covered, "minItems": 1, "maxItems": goal_count - 1},
                            "missing_goal_ids": {**properties["missing_goal_ids"],
                                                 "items": {"type": "string", "enum": goals}, "minItems": 1,
                                                 "maxItems": goal_count - 1, "uniqueItems": True}, **common},
                ["file_id", "covered_goals", "missing_goal_ids"])
        alternatives = {**properties["alternatives"], "minItems": 1}
        add("RELATED", {"alternatives": alternatives, "new_request": properties["new_request"]}, ["alternatives"])
    missing = {**properties["missing_goal_ids"], "items": {"type": "string", "enum": goals},
               "minItems": goal_count, "maxItems": goal_count, "const": goals}
    add("NO_KNOWLEDGE", {"missing_goal_ids": missing, "new_request": properties["new_request"]}, ["missing_goal_ids"])
    if catalogue:
        add("READ", {"file_id": {"type": "string", "enum": catalogue},
                     "new_request": properties["new_request"]}, ["file_id"])
    question_fields = {"question": {**properties["question"], "minLength": 5},
                       "new_request": properties["new_request"]}
    required = ["question"]
    strict_questions = payload.get("source_clarification_required") is True
    if strict_questions and loaded:
        question_fields.update(
            clarification_kind={"type": "string", "enum": [
                "USER_GOAL", "VISIBLE_STATE", "PHYSICAL_SETUP", "WORK_HISTORY",
            ]},
            clarification_sources={**properties["clarification_sources"], "minItems": 1},
        )
        required += ["clarification_kind", "clarification_sources"]
    if not strict_questions or loaded:
        add("CLARIFY", question_fields, required)
    return {"title": base.get("title", "InstructionSelection"), "$defs": definitions, "anyOf": branches}


class OpenWebUILLM:
    supports_source_clarification = True
    supports_instruction_planning = True
    supports_guide_context = True

    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport
        self.name = f"openwebui:{settings.openwebui_model}"[:255]

    def agent_action(self, payload):
        from app.services.llm.support_agent import agent_action

        return agent_action(self, payload)

    def classify_task_relation(self, message, memory, recent):
        from app.services.llm.support_agent import classify_task_relation

        return classify_task_relation(self, message, memory, recent)

    def _call(self, schema, task, payload, *, fast_selection=False, max_output_tokens=None):
        output_limit = 4096 if schema is SupportTaskUnderstanding else 1536
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or not 128 <= max_output_tokens <= output_limit
        ):
            raise ValueError("InvalidOutputTokenBudget")
        selection_schema = None
        if fast_selection:
            selection_schema = schema.model_json_schema()
            if schema is SupportTaskUnderstanding:
                selection_schema["required"] = list(dict.fromkeys([
                    *selection_schema.get("required", []), *_TASK_REQUIRED_FIELDS,
                ]))
                if payload.get("user_evidence_references") is True:
                    selection_schema = constrain_user_evidence(
                        selection_schema, payload["user_evidence"], payload["message"],
                    )
            if payload.get("instruction_planning") is not True and "parts" in selection_schema["properties"]:
                selection_schema["properties"].pop("parts")
                selection_schema["properties"].pop("section_ids")
                selection_schema["properties"]["delivery_mode"].pop("title", None)
                selection_schema["properties"]["result"]["enum"].remove("PLAN")
                selection_schema.get("$defs", {}).pop("InstructionPartSelection", None)
                selection_schema.get("$defs", {}).pop("PartReference", None)
            allowed_ids = list(payload.get("catalogue", {}))
            if "file_id" in selection_schema["properties"]:
                selection_schema["properties"]["file_id"] = (
                    {"anyOf": [{"type": "string", "enum": allowed_ids}, {"type": "null"}], "default": None}
                    if allowed_ids else {"type": "null", "default": None}
                )
            alternative_schema = selection_schema.get("$defs", {}).get("InstructionAlternative")
            if alternative_schema and allowed_ids:
                alternative_schema["properties"]["file_id"]["enum"] = allowed_ids
            if "alternatives" in selection_schema["properties"] and not allowed_ids:
                selection_schema["properties"]["alternatives"]["maxItems"] = 0
            if "clarification_sources" in selection_schema["properties"]:
                if payload.get("source_clarification_required") is True:
                    selection_schema["required"] = [*selection_schema.get("required", []),
                                                    "clarification_sources", "clarification_kind"]
                    if not allowed_ids:
                        selection_schema["properties"]["clarification_sources"]["maxItems"] = 0
                else:
                    selection_schema["properties"].pop("clarification_sources")
                    selection_schema["properties"].pop("clarification_kind")
                    selection_schema["properties"].pop("clarification_condition", None)
            if "covered_goals" in selection_schema["properties"] and not payload.get("requested_goals"):
                selection_schema["properties"].pop("covered_goals")
                selection_schema["properties"].pop("missing_goal_ids")
                selection_schema.get("$defs", {}).pop("InstructionGoalCoverage", None)
                selection_schema["properties"]["result"]["enum"].remove("PARTIAL")
            coverage = selection_schema.get("$defs", {}).get("InstructionGoalCoverage")
            if coverage and payload.get("requested_goals"):
                goal_ids = list(payload["requested_goals"])
                coverage["properties"]["goal_id"]["enum"] = goal_ids
                selection_schema["properties"]["missing_goal_ids"]["items"]["enum"] = goal_ids
                selection_schema["required"] = [*selection_schema.get("required", []),
                                                "covered_goals", "missing_goal_ids"]
            task_goal = selection_schema.get("$defs", {}).get("SupportTaskGoal")
            if task_goal:
                task_goal["required"] = list(dict.fromkeys([
                    *task_goal.get("required", []), "goal_id", "kind", "search_queries",
                ]))
                goal_ids = list(payload.get("previous_state", {}).get("goals", {}))
                task_goal["properties"]["goal_id"] = (
                    {"anyOf": [{"type": "string", "enum": goal_ids}, {"type": "null"}], "default": None}
                    if goal_ids else {"type": "null", "default": None}
                )
            if payload.get("allow_clarification") is False and "result" in selection_schema["properties"]:
                selection_schema["properties"]["result"]["enum"].remove("CLARIFY")
            if payload.get("read_available") is False and "result" in selection_schema["properties"]:
                outcomes = selection_schema["properties"]["result"].get("enum", [])
                if "READ" in outcomes:
                    outcomes.remove("READ")
            if payload.get("requested_goals") and "covered_goals" in selection_schema["properties"]:
                selection_schema = _instruction_schema_variants(selection_schema, payload)
        # LangChain orchestrates prompt -> existing internal transport -> strict validation.
        # No cloud model, autonomous tools or external memory service is introduced.
        request_options = {"selection_schema": selection_schema}
        if max_output_tokens is not None:
            request_options["max_output_tokens"] = max_output_tokens
        chain = (
            RunnableLambda(lambda value: stage_messages(schema, task, value, include_schema=not fast_selection))
            | RunnableLambda(lambda messages: self._request(messages, **request_options)
                             if fast_selection else self._request(messages))
            | RunnableLambda(schema.model_validate_json)
        )
        with tracing_context(enabled=False, parent=False):
            return chain.invoke(payload, config={"callbacks": []})

    def _request(self, messages, *, selection_schema=None, max_output_tokens=None, thinking=False, timeout_cap=None,
                 talk_agent=False):
        if (type(thinking) is not bool or thinking and (
                selection_schema is None or type(max_output_tokens) is not int or max_output_tokens != 8192)):
            raise ValueError("InvalidInstructionThinkingOptions")
        body = {
            "model": self.settings.openwebui_model,
            "stream": False,
            "tool_ids": [],
            # An explicit empty tools list prevents profile built-ins from resolving.
            "tools": [],
            "features": {"web_search": False},
            "temperature": 0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        if self.settings.openwebui_num_ctx is not None:
            body["params"] = {"num_ctx": self.settings.openwebui_num_ctx}
        if selection_schema is not None:
            # Open WebUI 0.11 forwards params through native Ollama options.
            # Selection needs a small grounded JSON decision and a short quote.
            output_limit = max_output_tokens if max_output_tokens is not None else 512
            body.update({
                "max_tokens": output_limit,
                "params": {**body.get("params", {}), "think": False, "max_tokens": output_limit, "temperature": 0},
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "instruction_selection", "schema": selection_schema,
                }},
            })
            if thinking:
                body["params"].update(think=True, num_predict=8192)
        if selection_schema is not None and selection_schema.get("title") == "AgentAction":
            shared_params = getattr(self.settings, "openwebui_generation_params", {})
            body["params"] = {**body.get("params", {}), **shared_params}
            if "temperature" in shared_params:
                body["temperature"] = shared_params["temperature"]
        started = perf_counter()
        timeout = max(self.settings.llm_timeout_seconds, 120) if thinking else self.settings.llm_timeout_seconds
        if timeout_cap is not None:
            timeout = min(timeout, timeout_cap)
        if talk_agent:
            timeout = httpx.Timeout(timeout, read=max(timeout, 180))
        with httpx.Client(
            timeout=timeout,
            transport=self.transport, follow_redirects=False
        ) as client:
            response = client.post(
                self.settings.openwebui_base_url.rstrip("/") + "/api/chat/completions",
                headers={"Authorization": "Bearer " + self.settings.openwebui_api_key.get_secret_value()},
                json=body,
            )
            response.raise_for_status()
            result = response.json()
            choice = result["choices"][0]
            content = choice["message"]["content"]
        if selection_schema is not None:
            usage = result.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            reason = choice.get("finish_reason")
            try:
                event("llm.instruction_response", total_ms=round((perf_counter() - started) * 1000, 3),
                      response_schema=selection_schema.get("title"),
                      input_chars=sum(len(message["content"]) for message in messages),
                      output_chars=len(content) if isinstance(content, str) else None,
                      finish_reason=reason if reason in {"stop", "length", "tool_calls", "content_filter"} else "other",
                      prompt_tokens=usage.get("prompt_tokens") if type(usage.get("prompt_tokens")) is int else None,
                      completion_tokens=usage.get("completion_tokens") if type(usage.get("completion_tokens")) is int else None)
            except Exception:
                pass  # Diagnostics never change the model's validated reply.
            if reason == "length":
                raise ValueError("IncompleteInstructionSelection")
        # No regex extraction / accepting trailing prose: fail closed on invalid output.
        return content

    def select_instruction(self, payload):
        from app.services.rag.talk import (
            INSTRUCTION_SELECTION_TASK, MAX_SELECTION_PAYLOAD_CHARS, InstructionSelection, KnowledgeUnavailable,
        )

        # Request-local codes keep the catalogue and response grammar small.
        # Original texts and source evidence remain byte-for-byte unchanged.
        catalogue = payload.get("catalogue", {})
        instruction_goal_metadata(payload, payload.get("requested_goals") or {})
        aliases = {file_id: f"s{number}" for number, file_id in enumerate(catalogue, 1)}
        originals = {alias: file_id for file_id, alias in aliases.items()}
        candidates = payload.get("candidates", {})
        if any(file_id not in aliases for file_id in candidates):
            raise ValueError("UnknownInstructionCandidate")
        wire_payload = {
            **payload,
            "catalogue": {aliases[file_id]: name for file_id, name in catalogue.items()},
            "candidates": {aliases[file_id]: content for file_id, content in candidates.items()},
        }
        wire_payload.pop("goal_search_queries", None)
        if "catalogue_cards" in payload:
            cards = payload["catalogue_cards"]
            if not isinstance(cards, dict) or any(file_id not in aliases for file_id in cards):
                raise ValueError("UnknownInstructionCatalogueCard")
            wire_payload["catalogue_cards"] = {aliases[key]: value for key, value in cards.items()}
        if "candidate_image_numbers" in payload:
            image_numbers = payload["candidate_image_numbers"]
            if not isinstance(image_numbers, dict) or any(file_id not in candidates for file_id in image_numbers):
                raise ValueError("UnknownInstructionImageCandidate")
            wire_payload["candidate_image_numbers"] = {
                aliases[file_id]: numbers for file_id, numbers in image_numbers.items()
            }
        image_captions = payload.get("candidate_image_captions", {})
        if (not isinstance(image_captions, dict) or any(file_id not in candidates for file_id in image_captions)
                or any(not isinstance(captions, dict)
                       or any(type(number) is not int or number < 1 or not isinstance(caption, str)
                              for number, caption in captions.items()) for captions in image_captions.values())):
            raise ValueError("UnknownInstructionImageCaption")
        if "candidate_image_captions" in payload:
            wire_payload["candidate_image_captions"] = {
                aliases[file_id]: captions for file_id, captions in image_captions.items()
            }
        context = payload.get("context")
        if isinstance(context, dict):
            wire_context = dict(context)
            wire_context.pop("goal_search_queries", None)
            task_state = wire_context.get("task_state")
            if isinstance(task_state, dict) and isinstance(task_state.get("goals"), dict):
                wire_context["task_state"] = {**task_state, "goals": {
                    key: {name: value for name, value in goal.items() if name != "search_queries"}
                    if isinstance(goal, dict) else goal for key, goal in task_state["goals"].items()
                }}
            for name in ("shown_source", "verified_previous_source"):
                source = context.get(name)
                if isinstance(source, dict) and "file_id" in source:
                    if source["file_id"] in aliases:
                        wire_context[name] = {**source, "file_id": aliases[source["file_id"]]}
                    else:
                        wire_context.pop(name, None)
            wire_payload["context"] = wire_context
        call_options = {"fast_selection": True}
        goals = payload.get("requested_goals")
        if goals:
            call_options["max_output_tokens"] = 768 if len(goals) <= 2 else 1024 if len(goals) <= 4 else 1536
            from app.services.rag.evidence import build_evidence_registry

            prepared = payload.get("candidate_evidence")
            if prepared is None:
                registries = {aliases[file_id]: build_evidence_registry(content, image_captions.get(file_id, {}))
                              for file_id, content in candidates.items()}
            else:
                if not isinstance(prepared, dict) or set(prepared) != set(candidates):
                    raise ValueError("UnknownInstructionEvidenceCandidate")
                registries = {}
                for file_id, registry in prepared.items():
                    source = " ".join(candidates[file_id].split())
                    if not isinstance(registry, dict) or len(registry) > 32:
                        raise ValueError("InvalidInstructionEvidenceRegistry")
                    for reference_id, proof in registry.items():
                        if (not isinstance(reference_id, str) or not isinstance(proof, dict)
                                or set(proof) != {"text", "support_kind", "image_numbers"}
                                or not isinstance(proof["text"], str) or not 1 <= len(proof["text"]) <= 200
                                or proof["text"] not in source or proof["support_kind"] not in {"TEXT", "IMAGE"}
                                or not isinstance(proof["image_numbers"], list)
                                or any(type(number) is not int or number < 1 for number in proof["image_numbers"])
                                or proof["support_kind"] == "TEXT" and proof["image_numbers"]):
                            raise ValueError("InvalidInstructionEvidenceRegistry")
                    registries[aliases[file_id]] = registry
            sections = payload.get("candidate_sections")
            if sections is not None:
                if not isinstance(sections, dict) or set(sections) != set(candidates):
                    raise ValueError("UnknownInstructionSectionCandidate")
                wire_payload["candidate_sections"] = {aliases[key]: value for key, value in sections.items()}
            wire_payload.update(candidates=registries, evidence_references=True)
            wire_payload.pop("candidate_evidence", None)
            wire_payload.pop("candidate_image_captions", None)
            wire_payload.pop("candidate_image_numbers", None)
        if len(json.dumps(wire_payload, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS:
            raise KnowledgeUnavailable("SelectionPayloadTooLarge", "instruction_selection")
        if goals:
            references = self._call(InstructionReferenceSelection, INSTRUCTION_SELECTION_TASK,
                                    wire_payload, **call_options)
            choice = InstructionSelection.model_validate(resolve_instruction_references(
                references, registries, planning=payload.get("instruction_planning") is True,
                sections=wire_payload.get("candidate_sections"),
            ))
        else:
            choice = self._call(InstructionSelection, INSTRUCTION_SELECTION_TASK, wire_payload, **call_options)
        decoded = choice.model_dump()
        source_ids = [choice.file_id] if choice.file_id is not None else []
        source_ids.extend(alternative.file_id for alternative in choice.alternatives)
        source_ids.extend(source.file_id for source in choice.clarification_sources)
        source_ids.extend(part.file_id for part in choice.parts)
        if any(source_id not in originals for source_id in source_ids):
            raise ValueError("UnknownInstructionSourceAlias")
        if choice.file_id is not None:
            decoded["file_id"] = originals[choice.file_id]
        for alternative in decoded["alternatives"]:
            alternative["file_id"] = originals[alternative["file_id"]]
        for source in decoded["clarification_sources"]:
            source["file_id"] = originals[source["file_id"]]
        for part in decoded["parts"]:
            part["file_id"] = originals[part["file_id"]]
        return InstructionSelection.model_validate(decoded)

    def read_instruction(self, payload):
        """Read one complete approved original with exact, source-bound references."""
        from app.services.llm.instruction_reader import (
            InstructionReading, READING_TASK, prepare_reading, resolve_reading,
        )
        from app.services.rag.talk import MAX_SELECTION_PAYLOAD_CHARS, KnowledgeUnavailable

        independent = getattr(self.verify_instruction, "__func__", None) is OpenWebUILLM.verify_instruction
        wire, schema, reverse = prepare_reading(payload, allow_mode_review=independent)
        if len(json.dumps(wire, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS:
            raise KnowledgeUnavailable("ReadingPayloadTooLarge", "instruction_reading")
        for attempt in range(2):
            messages = stage_messages(InstructionReading, READING_TASK, wire, include_schema=False)
            raw = self._request(messages, selection_schema=schema,
                                max_output_tokens=1536 if len(payload["requested_goals"]) <= 4 else 4096)
            try:
                reading = resolve_reading(InstructionReading.model_validate_json(raw), payload, reverse,
                                          allow_mode_review=independent)
            except ValueError as error:
                if attempt:
                    raise
                wire["repair_reason"] = _instruction_contract_repair(error, "Reading")
            else:
                return self.verify_instruction(payload, reading)
        raise ValueError("InvalidInstructionReading")

    def verify_instruction(self, payload, reading, *, for_delivery=True):
        """Independently verify each proposed answer against the complete original."""
        from app.services.llm.instruction_reader import InstructionReading, resolve_reading
        from app.services.llm.instruction_verification import (
            ACTION_PLAN_REVIEW_TASK, VERIFICATION_TASK, InstructionVerification,
            action_plan_review_preserves_scope,
            normalize_native_verification_refs, prepare_action_plan_review, prepare_verification, resolve_verification,
        )
        from app.services.rag.talk import MAX_SELECTION_PAYLOAD_CHARS, KnowledgeUnavailable

        result = InstructionReading.model_validate(reading).model_copy(deep=True)
        identities = {reference: reference for reference in payload["candidate_evidence"]}
        resolve_reading(result, payload, identities, allow_mode_review=True)
        for goal_id, answer in list(result.goals.items()):
            if answer.status not in {"SUPPORTED", "RELATED"}:
                continue
            wire, schema, reverse = prepare_verification(payload, goal_id, answer)
            review_baseline = None
            for attempt in range(2):
                if len(json.dumps(wire, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS:
                    raise KnowledgeUnavailable("VerificationPayloadTooLarge", "instruction_verification")
                task = VERIFICATION_TASK
                if attempt and "action_plan_review" in wire:
                    task += ACTION_PLAN_REVIEW_TASK
                messages = [{"role": "system", "content": task},
                            {"role": "user", "content": json.dumps(wire, ensure_ascii=False)}]
                options = {"max_output_tokens": 2048}
                if (review_baseline is None and for_delivery and self.settings.openwebui_verification_think
                        and answer.status == "SUPPORTED"):
                    options = {"max_output_tokens": 8192, "thinking": True}
                try:
                    raw = self._request(messages, selection_schema=schema, **options)
                except (httpx.HTTPError, ValueError):
                    if review_baseline is None:
                        raise
                    verification, checked = review_baseline
                    break
                try:
                    verification = InstructionVerification.model_validate_json(raw)
                    verification = normalize_native_verification_refs(verification)
                    checked = resolve_verification(verification, payload, goal_id, reverse, answer)
                    if review_baseline is not None:
                        first_verification, first_checked = review_baseline
                        contract_fields = {"operation_match", "condition_match", "goal_kind",
                                           "required_material_kind", "material_state", "all_materials_available"}
                        if (checked.status != "SUPPORTED" or checked.delivery_mode != first_checked.delivery_mode
                                or verification.model_dump(include=contract_fields)
                                != first_verification.model_dump(include=contract_fields)
                                or not action_plan_review_preserves_scope(payload, goal_id, first_checked, checked)):
                            verification, checked = review_baseline
                        break
                    review = (prepare_action_plan_review(payload, goal_id, verification, checked, reverse,
                                                         reader_evidence_ids=answer.evidence_ids)
                              if for_delivery and answer.status == "SUPPORTED" and not attempt else None)
                    if review is not None:
                        review_wire = {**wire, "action_plan_review": review}
                        if len(json.dumps(review_wire, ensure_ascii=False)) <= MAX_SELECTION_PAYLOAD_CHARS:
                            review_baseline = verification, checked
                            wire = review_wire
                            continue
                    break
                except ValueError as error:
                    if review_baseline is not None:
                        verification, checked = review_baseline
                        break
                    if attempt:
                        raise
                    wire["repair_reason"] = _instruction_contract_repair(error, "Verification")
            result.goals[goal_id] = (self._instruction_illustrations(payload, goal_id, verification, checked)
                                     if for_delivery else checked)
        return resolve_reading(result, payload, identities)

    def _instruction_illustrations(self, payload, goal_id, verification, answer):
        from app.services.llm.instruction_verification import (
            ILLUSTRATION_TASK, InstructionIllustrationSelection, compact_illustration_wire,
            prepare_illustrations, resolve_illustrations,
        )
        from app.services.rag.talk import MAX_SELECTION_PAYLOAD_CHARS

        prepared = prepare_illustrations(payload, goal_id, verification, answer)
        if prepared is None:
            return answer
        wire, schema, reverse = prepared
        capacity = 96 - len(answer._verified_scope["answer_refs"])
        if capacity <= 0:
            return answer
        if capacity < 24:
            schema["properties"]["illustration_refs"]["maxItems"] = capacity
        for attempt in range(2):
            if len(json.dumps(wire, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS:
                wire = compact_illustration_wire(wire)
                if len(json.dumps(wire, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS:
                    return answer
            messages = [{"role": "system", "content": ILLUSTRATION_TASK},
                        {"role": "user", "content": json.dumps(wire, ensure_ascii=False)}]
            raw = self._request(messages, selection_schema=schema, max_output_tokens=512)
            try:
                selected = InstructionIllustrationSelection.model_validate_json(raw)
                return resolve_illustrations(selected, payload, goal_id, reverse, answer)
            except ValueError as error:
                if attempt:
                    raise
                wire["repair_reason"] = _instruction_contract_repair(error, "Verification")
        raise ValueError("InvalidInstructionVerificationIllustration")

    def related_instruction(self, payload):
        """Offer a same-purpose alternative only after validating its actual source goal."""
        from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
        from app.services.llm.instruction_related import (
            RELATED_TASK, RelatedInstructionSelection, prepare_related, resolve_related, unrelated_instruction,
        )
        from app.services.rag.talk import MAX_SELECTION_PAYLOAD_CHARS, KnowledgeUnavailable

        wire, schema, reverse = prepare_related(payload)
        if not wire["source_scopes"]:
            return unrelated_instruction()
        for attempt in range(2):
            if len(json.dumps(wire, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS:
                raise KnowledgeUnavailable("RelatedPayloadTooLarge", "instruction_related")
            messages = [{"role": "system", "content": RELATED_TASK},
                        {"role": "user", "content": json.dumps(wire, ensure_ascii=False)}]
            raw = self._request(messages, selection_schema=schema, max_output_tokens=512)
            try:
                proposal = resolve_related(RelatedInstructionSelection.model_validate_json(raw), payload, reverse)
                break
            except ValueError as error:
                if attempt:
                    raise
                wire["repair_reason"] = _instruction_contract_repair(error, "Related")
        if proposal.status == "UNRELATED":
            return proposal
        goal_id = proposal.original_goal_id
        kinds = payload.get("requested_goal_kinds", {})
        kind = kinds.get(goal_id)
        mode = "EXCERPT" if kind == "LOOKUP" else "PROCEDURE"
        alternative = {**payload, "requested_goals": {goal_id: proposal.source_goal}, "context": {},
                       "requested_goal_kinds": {goal_id: kind} if kind is not None else {}}
        reading = InstructionReading(goals={goal_id: SourceGoalReading(
            status="SUPPORTED", evidence_ids=proposal.evidence_ids, delivery_mode=mode,
        )})
        verify = self.verify_instruction
        checked = (verify(alternative, reading, for_delivery=False)
                   if getattr(verify, "__func__", None) is OpenWebUILLM.verify_instruction
                   else verify(alternative, reading)).goals[goal_id]
        if checked.status != "SUPPORTED":
            return unrelated_instruction()
        return proposal.model_copy(update={"evidence_ids": checked.evidence_ids})

    def nominate_instructions(self, payload):
        """Nominate unread approved originals without generating a user answer."""
        from app.services.llm.instruction_discovery import (
            DISCOVERY_TASK, MAX_DISCOVERY_PAYLOAD_CHARS, ORIGINAL_ORDER_TASK, InstructionNomination,
            prepare_discovery, prepare_original_order, resolve_discovery,
        )

        wire, schema, reverse = prepare_discovery(payload)
        if not reverse:
            return InstructionNomination(file_ids=[])
        for attempt in range(2):
            if len(json.dumps(wire, ensure_ascii=False)) > MAX_DISCOVERY_PAYLOAD_CHARS:
                raise ValueError("InstructionDiscoveryPayloadTooLarge")
            messages = stage_messages(InstructionNomination, DISCOVERY_TASK, wire, include_schema=False)
            raw = self._request(messages, selection_schema=schema, max_output_tokens=256)
            try:
                nomination = InstructionNomination.model_validate_json(raw)
                resolved = resolve_discovery(nomination, reverse)
            except ValueError:
                if attempt:
                    raise
                wire["repair_reason"] = "Выбери до трёх разных file_ids только из catalogue."
                continue
            if attempt == 0:
                comparison = prepare_original_order(payload, nomination.file_ids, reverse)
                if comparison is not None:
                    return self._order_nominated_originals(comparison, nomination, reverse, resolved,
                                                           ORIGINAL_ORDER_TASK)
            return resolved
        raise ValueError("InvalidInstructionNomination")

    def _order_nominated_originals(self, comparison, nomination, reverse, fallback, task):
        """An optional comparison can only reorder the files already nominated."""
        wire, schema = comparison
        try:
            raw = self._request([{"role": "system", "content": task}, {"role": "user", "content": wire}],
                                selection_schema=schema, max_output_tokens=1024, timeout_cap=20)
            data = json.loads(raw)
            order = data["file_order"]
            if (set(data) == {"file_order"} and isinstance(order, list)
                    and len(order) == len(nomination.file_ids) and set(order) == set(nomination.file_ids)):
                return fallback.model_copy(update={"file_ids": [reverse[alias] for alias in order]})
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            pass
        return fallback

    def interpret_instruction_offer(self, message, offered):
        return self._call(InstructionOfferReply, INSTRUCTION_OFFER_TASK,
                          {"message": message, "offered": offered}, fast_selection=True)

    def adopt_instruction_offer(self, payload):
        from app.services.llm.offer_adoption import classify_offer_adoption

        return classify_offer_adoption(self, payload)

    def interpret_understanding_reply(self, message, understanding):
        reply = self._call(UnderstandingReply, UNDERSTANDING_REPLY_TASK,
                           {"message": message, "understanding": understanding}, fast_selection=True)
        if reply.result == "UNCLEAR":
            if reply.evidence:
                raise ValueError("UnexpectedUnderstandingEvidence")
        elif not reply.evidence or reply.evidence not in message:
            raise ValueError("InvalidUnderstandingEvidence")
        return reply

    def understand_task(self, message, previous_state, previous_question, *, guide_context=None):
        if not isinstance(message, str) or not message.strip() or len(message) > 6000:
            raise ValueError("InvalidTaskMessage")
        references = build_user_evidence(task_evidence_authority(message, previous_state, False))
        if not references:
            raise ValueError("InvalidTaskMessage")
        payload = {
            "message": message, "previous_state": previous_state, "previous_question": previous_question,
            "user_evidence_references": True, "user_evidence": references,
            "current_user_evidence_ids": [key for key, text in references.items() if text in message],
        }
        if guide_context is not None:
            if not isinstance(guide_context, dict) or len(json.dumps(guide_context, ensure_ascii=False)) > 10000:
                raise ValueError("InvalidTaskGuideContext")
            payload["guide_context"] = guide_context
        for attempt in range(2):
            reply = self._call(SupportTaskUnderstanding, SUPPORT_TASK_TASK, payload,
                               fast_selection=True, max_output_tokens=4096)
            missing = [field for field in _TASK_REQUIRED_FIELDS if field not in reply.model_fields_set]
            missing.extend(f"goals.{index}.{name}" for index, goal in enumerate(reply.goals)
                           for name in ("goal_id", "kind", "search_queries") if name not in goal.model_fields_set)
            authority = task_evidence_authority(message, previous_state, reply.new_task)
            invalid_evidence = []
            for field, items in (("facts", reply.facts), ("goals", reply.goals)):
                for index, item in enumerate(items):
                    if field == "facts" and item.evidence in references:
                        missing.extend(f"facts.{index}.{name}" for name in ("negated", "entity", "attribute")
                                       if name not in item.model_fields_set)
                    resolved = resolve_user_evidence(item.evidence, references, authority)
                    if resolved is None:
                        invalid_evidence.append(f"{field}.{index}.evidence")
                    else:
                        item.evidence = resolved
            for field in ("replacement_evidence", "unknown_answer_evidence"):
                evidence = getattr(reply, field)
                if evidence:
                    resolved = resolve_user_evidence(evidence, references, [message])
                    if resolved is None:
                        invalid_evidence.append(field)
                    else:
                        setattr(reply, field, resolved)
            try:
                event("llm.task_understanding", new_task=reply.new_task, replace_goals=reply.replace_goals,
                      goal_count=len(reply.goals), goal_ids=[goal.goal_id for goal in reply.goals],
                      fact_keys=[fact.key for fact in reply.facts], question_field=reply.question_field,
                      has_question=bool(reply.question.strip()), missing_fields=missing,
                      invalid_evidence_fields=invalid_evidence,
                      contract_retry=bool(attempt))
            except Exception:
                pass  # Diagnostics never change a validated task interpretation.
            if not missing and not invalid_evidence:
                current_goal = any(goal.evidence and goal.evidence in message for goal in reply.goals)
                pending_field = (isinstance(previous_state, dict) and bool(previous_question)
                                 and previous_state.get("question") == previous_question
                                 and previous_state.get("question_field") is not None)
                if (reply.unknown_answer_evidence and not is_unknown_answer(message) and current_goal
                        and (reply.new_task or not pending_field)):
                    # A non-choice reply can state a complete request without
                    # answering a task field. Keep its grounded facts/goals.
                    reply.unknown_answer_evidence = ""
                return reply
            payload = {**payload, "contract_retry": True}
            if missing:
                payload["missing_fields"] = missing
            if invalid_evidence:
                payload["invalid_evidence_fields"] = invalid_evidence
        raise ValueError("IncompleteTaskUnderstanding" if missing else "InvalidTaskEvidence")

    def analyze(self, message):
        return self.analyze_context(message, {})

    def dialogue(self, message, context):
        return self._call(DialogueResult, DIALOGUE_TASK, {"message": message, "context": context})

    def extract_escalation(self, message, context):
        return self._call(EscalationExtraction, ESCALATION_EXTRACTION_TASK, {"message": message, "context": context})

    def analyze_context(self, message, context):
        return self._call(AnalysisResult, ANALYZE_TASK, {"message": message, "context": context})

    def followup(self, question, context, approved_source):
        if isinstance(approved_source.get("excerpt_image_indices"), dict):
            from app.services.llm.followup_references import select_followup_references

            return select_followup_references(self, question, context, approved_source)
        return self._call(FollowupResult, FOLLOWUP_TASK, {
            "question": question, "context": context, "approved_source": approved_source,
        }, fast_selection=True, max_output_tokens=512)

    def compose(self, candidate):
        return self._call(
            AnswerResult,
            COMPOSE_TASK,
            {
                "solution_id": candidate.solution_id,
                "source_chunk_ids": [candidate.chunk_id],
                "title": candidate.title,
                "evidence": candidate.content,
                "steps": candidate.steps,
            },
        )

    def summarize(self, transcript):
        return self._call(
            SummaryResult,
            SUMMARY_TASK,
            {"transcript": transcript},
        )
