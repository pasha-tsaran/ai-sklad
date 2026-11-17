from time import perf_counter

import httpx
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from app.core.logging import event
from app.schemas.contracts import (
    AnalysisResult, AnswerResult, DialogueResult, EscalationExtraction, FollowupResult, InstructionOfferReply, SummaryResult,
    SupportTaskUnderstanding, UnderstandingReply,
)
from app.services.llm.prompts import (
    ANALYZE_TASK, COMPOSE_TASK, DIALOGUE_TASK, ESCALATION_EXTRACTION_TASK, FOLLOWUP_TASK,
    INSTRUCTION_OFFER_TASK, SUMMARY_TASK, SUPPORT_TASK_TASK, UNDERSTANDING_REPLY_TASK, stage_messages,
)


class OpenWebUILLM:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport
        self.name = f"openwebui:{settings.openwebui_model}"[:255]

    def _call(self, schema, task, payload, *, fast_selection=False, max_output_tokens=None):
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or not 128 <= max_output_tokens <= 1536
        ):
            raise ValueError("InvalidOutputTokenBudget")
        selection_schema = None
        if fast_selection:
            selection_schema = schema.model_json_schema()
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

    def _request(self, messages, *, selection_schema=None, max_output_tokens=None):
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
        started = perf_counter()
        with httpx.Client(
            timeout=self.settings.llm_timeout_seconds, transport=self.transport, follow_redirects=False
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
        from app.services.rag.talk import INSTRUCTION_SELECTION_TASK, InstructionSelection

        # Request-local codes keep the catalogue and response grammar small.
        # Original texts and source evidence remain byte-for-byte unchanged.
        catalogue = payload.get("catalogue", {})
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
        if "candidate_image_numbers" in payload:
            image_numbers = payload["candidate_image_numbers"]
            if not isinstance(image_numbers, dict) or any(file_id not in candidates for file_id in image_numbers):
                raise ValueError("UnknownInstructionImageCandidate")
            wire_payload["candidate_image_numbers"] = {
                aliases[file_id]: numbers for file_id, numbers in image_numbers.items()
            }
        context = payload.get("context")
        if isinstance(context, dict):
            wire_context = dict(context)
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
        choice = self._call(InstructionSelection, INSTRUCTION_SELECTION_TASK, wire_payload, **call_options)
        decoded = choice.model_dump()
        source_ids = [choice.file_id] if choice.file_id is not None else []
        source_ids.extend(alternative.file_id for alternative in choice.alternatives)
        if any(source_id not in originals for source_id in source_ids):
            raise ValueError("UnknownInstructionSourceAlias")
        if choice.file_id is not None:
            decoded["file_id"] = originals[choice.file_id]
        for alternative in decoded["alternatives"]:
            alternative["file_id"] = originals[alternative["file_id"]]
        return InstructionSelection.model_validate(decoded)

    def interpret_instruction_offer(self, message, offered):
        return self._call(InstructionOfferReply, INSTRUCTION_OFFER_TASK,
                          {"message": message, "offered": offered}, fast_selection=True)

    def interpret_understanding_reply(self, message, understanding):
        reply = self._call(UnderstandingReply, UNDERSTANDING_REPLY_TASK,
                           {"message": message, "understanding": understanding}, fast_selection=True)
        if reply.result == "UNCLEAR":
            if reply.evidence:
                raise ValueError("UnexpectedUnderstandingEvidence")
        elif not reply.evidence or reply.evidence not in message:
            raise ValueError("InvalidUnderstandingEvidence")
        return reply

    def understand_task(self, message, previous_state, previous_question):
        return self._call(SupportTaskUnderstanding, SUPPORT_TASK_TASK, {
            "message": message, "previous_state": previous_state, "previous_question": previous_question,
        }, fast_selection=True, max_output_tokens=1536)

    def analyze(self, message):
        return self.analyze_context(message, {})

    def dialogue(self, message, context):
        return self._call(DialogueResult, DIALOGUE_TASK, {"message": message, "context": context})

    def extract_escalation(self, message, context):
        return self._call(EscalationExtraction, ESCALATION_EXTRACTION_TASK, {"message": message, "context": context})

    def analyze_context(self, message, context):
        return self._call(AnalysisResult, ANALYZE_TASK, {"message": message, "context": context})

    def followup(self, question, context, approved_source):
        return self._call(FollowupResult, FOLLOWUP_TASK, {
            "question": question, "context": context, "approved_source": approved_source,
        })

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
