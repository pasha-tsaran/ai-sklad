from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


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
    account_holder_name: str = Field(min_length=2, max_length=255)
    account_login: str = Field(min_length=1, max_length=100, pattern=r"^[^\r\n]+$")
    computer_name: str = Field(min_length=1, max_length=63, pattern=r"^[A-Za-z0-9_.-]+$")
    subject: str = Field(min_length=3, max_length=200, pattern=r"^[^\r\n]+$")
    additional_info: str = Field(default="", max_length=2000)


class EscalationExtraction(StrictModel):
    """Untrusted field candidates; backend must verify their verbatim evidence."""

    account_holder_name: str | None = Field(default=None, max_length=255)
    account_login: str | None = Field(default=None, max_length=100)
    computer_name: str | None = Field(default=None, max_length=63)
    subject: str | None = Field(default=None, max_length=200)
    additional_info: str | None = Field(default=None, max_length=2000)


class AnalysisResult(StrictModel):
    intent: Literal["REPORT_PROBLEM", "REQUEST_HUMAN", "OTHER"]
    problem_type: str | None = Field(default=None, max_length=100)
    equipment_model: str | None = Field(default=None, max_length=255)
    search_query: str = Field(max_length=1000)
    clarification_key: Literal["DEVICE_TYPE", "DEVICE_MODEL", "CONNECTION", "ERROR_TEXT", "SYMPTOMS"] | None = None


class DialogueResult(StrictModel):
    route: Literal["SUPPORT", "SOCIAL", "REFUSE"]
    reply: str = Field(default="", max_length=800)


class InstructionOfferReply(StrictModel):
    result: Literal["ACCEPT", "REJECT", "NEW_REQUEST", "UNCLEAR"]
    choice: StrictInt | None = Field(default=None, ge=1, le=3)
    evidence: str = Field(default="", max_length=300)


class UnderstandingReply(StrictModel):
    result: Literal["ACCEPT", "CORRECT", "NEW_REQUEST", "UNCLEAR"]
    evidence: str = Field(default="", max_length=300)


class SupportTaskFact(StrictModel):
    key: Literal["equipment_type", "equipment_model", "application", "connection", "completed", "scope",
                 "error_text", "symptoms", "current_step"]
    value: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=300)


class SupportTaskGoal(StrictModel):
    text: str = Field(min_length=1, max_length=250)
    evidence: str = Field(min_length=1, max_length=300)
    goal_id: str | None = Field(default=None, pattern=r"^g[1-9][0-9]*$")


class SupportUnknownAnswer(StrictModel):
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(default="", max_length=300)
    field: Literal["equipment_type", "goal", "connection", "application", "symptoms",
                   "error_text", "current_step", "other"] | None = None


class SupportTaskUnderstanding(StrictModel):
    new_task: StrictBool = False
    facts: list[SupportTaskFact] = Field(default_factory=list, max_length=12)
    goals: list[SupportTaskGoal] = Field(default_factory=list, max_length=8)
    replace_goals: StrictBool = False
    replacement_evidence: str = Field(default="", max_length=300)
    unknown_answer_evidence: str = Field(default="", max_length=300)
    question: str = Field(default="", max_length=300)
    question_field: Literal["equipment_type", "goal", "connection", "application", "symptoms",
                            "error_text", "current_step", "other"] | None = None


class FollowupResult(StrictModel):
    result: Literal["EXCERPT", "NO_KNOWLEDGE"]
    source_chunk_ids: list[str] = Field(default_factory=list, max_length=1)
    step_indices: list[StrictInt] = Field(default_factory=list, max_length=12)
    illustration_indices: list[StrictInt] = Field(default_factory=list, max_length=6)


class AnswerResult(StrictModel):
    result: Literal["ANSWER", "NEED_MORE_INFO", "NO_KNOWLEDGE", "ESCALATE"]
    solution_id: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list, max_length=10)
    # Only these backend-owned phrases are allowed into the technical response.
    intro_key: Literal["TRY_APPROVED_STEPS", "CHECK_APPROVED_STEPS"] = "TRY_APPROVED_STEPS"


class SummaryResult(StrictModel):
    summary: str = Field(min_length=1, max_length=2000)
