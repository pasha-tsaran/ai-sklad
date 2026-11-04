import httpx
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from app.schemas.contracts import AnalysisResult, AnswerResult, DialogueResult, FollowupResult, SummaryResult
from app.services.llm.prompts import (
    ANALYZE_TASK, COMPOSE_TASK, DIALOGUE_TASK, FOLLOWUP_TASK, SUMMARY_TASK, stage_messages,
)


class OpenWebUILLM:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport
        self.name = f"openwebui:{settings.openwebui_model}"[:255]

    def _call(self, schema, task, payload):
        # LangChain orchestrates prompt -> existing internal transport -> strict validation.
        # No cloud model, autonomous tools or external memory service is introduced.
        chain = (
            RunnableLambda(lambda value: stage_messages(schema, task, value))
            | RunnableLambda(self._request)
            | RunnableLambda(schema.model_validate_json)
        )
        with tracing_context(enabled=False, parent=False):
            return chain.invoke(payload, config={"callbacks": []})

    def _request(self, messages):
        with httpx.Client(
            timeout=self.settings.llm_timeout_seconds, transport=self.transport, follow_redirects=False
        ) as client:
            response = client.post(
                self.settings.openwebui_base_url.rstrip("/") + "/api/chat/completions",
                headers={"Authorization": "Bearer " + self.settings.openwebui_api_key.get_secret_value()},
                json={
                    "model": self.settings.openwebui_model,
                    "stream": False,
                    "tool_ids": [],
                    # Open WebUI otherwise adds profile built-ins even when
                    # tool_ids is empty. Explicit tools opts out of resolution.
                    "tools": [],
                    "features": {"web_search": False},
                    "temperature": 0,
                    "max_tokens": 1200,
                    "response_format": {"type": "json_object"},
                    "messages": messages,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        # No regex extraction / accepting trailing prose: fail closed on invalid output.
        return content

    def analyze(self, message):
        return self.analyze_context(message, {})

    def dialogue(self, message, context):
        return self._call(DialogueResult, DIALOGUE_TASK, {"message": message, "context": context})

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
