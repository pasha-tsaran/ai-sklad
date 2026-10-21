from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    incident_id: str | None = Field(default=None, max_length=36)


class FeedbackRequest(StrictModel):
    attempt_id: str = Field(min_length=36, max_length=36)
    helped: StrictBool
    comment: str = Field(default="", max_length=2000)


class EscalationDetails(StrictModel):
    account_holder_name: str = Field(min_length=3, max_length=255)
    account_login: str = Field(min_length=1, max_length=100, pattern=r"^[^\r\n]+$")
    computer_name: str = Field(min_length=1, max_length=63, pattern=r"^[A-Za-z0-9_.-]+$")
    subject: str = Field(min_length=3, max_length=200, pattern=r"^[^\r\n]+$")
    additional_info: str = Field(default="", max_length=2000)


class AnalysisResult(StrictModel):
    intent: Literal["REPORT_PROBLEM", "REQUEST_HUMAN", "OTHER"]
    problem_type: str | None = Field(default=None, max_length=100)
    equipment_model: str | None = Field(default=None, max_length=255)
    search_query: str = Field(max_length=1000)


class AnswerResult(StrictModel):
    result: Literal["ANSWER", "NEED_MORE_INFO", "NO_KNOWLEDGE", "ESCALATE"]
    solution_id: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list, max_length=10)
    # Only these backend-owned phrases are allowed into the technical response.
    intro_key: Literal["TRY_APPROVED_STEPS", "CHECK_APPROVED_STEPS"] = "TRY_APPROVED_STEPS"


class SummaryResult(StrictModel):
    summary: str = Field(min_length=1, max_length=2000)
