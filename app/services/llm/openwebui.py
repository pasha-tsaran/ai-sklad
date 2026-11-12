from time import perf_counter

import httpx
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from app.core.logging import event
from app.schemas.contracts import (
    AnalysisResult, AnswerResult, DialogueResult, EscalationExtraction, FollowupResult, InstructionOfferReply, SummaryResult,
)
from app.services.llm.prompts import (
    ANALYZE_TASK, COMPOSE_TASK, DIALOGUE_TASK, ESCALATION_EXTRACTION_TASK, FOLLOWUP_TASK,
    INSTRUCTION_OFFER_TASK, SUMMARY_TASK, stage_messages,
)


class OpenWebUILLM:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport
        self.name = f"openwebui:{settings.openwebui_model}"[:255]

    def _call(self, schema, task, payload, *, fast_selection=False):
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
        # LangChain orchestrates prompt -> existing internal transport -> strict validation.
        # No cloud model, autonomous tools or external memory service is introduced.
        chain = (
            RunnableLambda(lambda value: stage_messages(schema, task, value))
            | RunnableLambda(lambda messages: self._request(messages, selection_schema=selection_schema)
                             if fast_selection else self._request(messages))
            | RunnableLambda(schema.model_validate_json)
        )
        with tracing_context(enabled=False, parent=False):
            return chain.invoke(payload, config={"callbacks": []})

    def _request(self, messages, *, selection_schema=None):
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
        if selection_schema is not None:
            # Open WebUI 0.11 forwards params through native Ollama options.
            # Selection needs a small grounded JSON decision and a short quote.
            body.update({
                "max_tokens": 512,
                "params": {"think": False, "max_tokens": 512, "temperature": 0},
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

        return self._call(InstructionSelection, INSTRUCTION_SELECTION_TASK, payload, fast_selection=True)

    def interpret_instruction_offer(self, message, offered):
        return self._call(InstructionOfferReply, INSTRUCTION_OFFER_TASK,
                          {"message": message, "offered": offered}, fast_selection=True)

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
