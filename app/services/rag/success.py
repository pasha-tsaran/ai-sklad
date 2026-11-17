"""Private, source-only shortcuts learned exclusively from explicit success."""

import copy
import hashlib
import json
import re
from datetime import timedelta, timezone
from uuid import UUID

from sqlalchemy import select

from app.models import Conversation, Feedback, Incident, Message, SolutionAttempt, SuccessfulRetrieval, now
from app.services.chat.memory import _redact
from app.services.chat.instruction_plan import immutable_plan, restore_plan, serialize_plan
from app.services.chat.task_state import task_query
from app.services.rag.query import normalized_query, source_compatible
from app.services.rag.talk import InstructionDecision, InstructionPlanPart, KnowledgeUnavailable
from integrations.openwebui.approved_instruction_images import allowed_source_filename

_HASH = re.compile(r"^[0-9a-f]{64}$")
_KEY_FIELDS = {"scope_hash", "query_hash", "context_hash"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _scope(settings):
    ids = settings.openwebui_knowledge_ids
    if not ids or any(not isinstance(value, str) for value in ids):
        return None
    try:
        collections = sorted({str(UUID(value)) for value in ids})
    except (ValueError, TypeError, AttributeError):
        return None
    return _digest({"base_url": settings.openwebui_base_url.rstrip("/"), "collections": collections})


def cache_key(settings, query, context):
    """Hash the complete problem plus current technical facts; ignore all PII/history."""
    text, scope = normalized_query(query), _scope(settings)
    if not text or not scope:
        return None
    state = context.get("task_state") if isinstance(context, dict) else None
    if isinstance(state, dict) and state.get("goals"):
        facts = {"version": 2,
                 "facts": sorted((fact["key"], normalized_query(fact["value"]),
                                  normalized_query(fact.get("entity", "")),
                                  normalized_query(fact.get("attribute", "")), fact.get("negated", False))
                                 for fact in state.get("facts", [])),
                 "goals": [normalized_query(goal.get("text", "") if isinstance(goal, dict) else goal)
                           for goal in state["goals"].values()],
                 "unknown_answers": sorted((normalized_query(item.get("question", "")),
                                            normalized_query(item.get("answer", "")))
                                           for item in state.get("unknown_answers", context.get("unknown_answers", [])))}
    else:
        incident = context.get("incident", {}) if isinstance(context, dict) else {}
        facts = {key: normalized_query(incident.get(key)) for key in ("equipment_model", "problem_type")}
    query_identity = {"purpose": "typed_task_v2", "goals": facts["goals"]} if "version" in facts else text
    return {"scope_hash": scope, "query_hash": _digest(query_identity), "context_hash": _digest(facts)}


def _valid_key(settings, key):
    return (isinstance(key, dict) and set(key) == _KEY_FIELDS
            and all(isinstance(value, str) and _HASH.fullmatch(value) for value in key.values())
            and key["scope_hash"] == _scope(settings))


def _valid_source(settings, source):
    if not isinstance(source, dict) or set(source) != {"file_id", "filename", "knowledge_id", "sha256"}:
        return False
    try:
        if str(UUID(source["file_id"])) != source["file_id"]:
            return False
    except (ValueError, TypeError, AttributeError):
        return False
    name = source["filename"]
    return (source["knowledge_id"] in settings.openwebui_knowledge_ids
            and isinstance(source["sha256"], str) and _HASH.fullmatch(source["sha256"])
            and allowed_source_filename(name))


def _proof(session, principal, incident_id, source):
    """A cache row cannot train itself or cross a conversation's actual principal."""
    incident = session.scalar(select(Incident).join(Conversation).where(
        Incident.id == incident_id, Conversation.user_id == principal,
        Incident.state == "RESOLVED", Incident.closed_at.is_not(None),
    ))
    if incident is None:
        return None
    attempts = session.scalars(select(SolutionAttempt).join(Feedback).where(
        SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "SUCCEEDED", Feedback.helped.is_(True),
    )).all()
    return incident if any(attempt.evidence.get("talk_source") == source for attempt in attempts) else None


def _hint_key(settings, query, incident_id):
    scope = _scope(settings)
    if not isinstance(query, str) or not query.strip() or len(query) > 6000 or scope is None:
        return None
    normalized = normalized_query(query)
    if not normalized:
        return None
    return {"scope_hash": scope, "query_hash": _digest(normalized),
            "context_hash": _digest({"purpose": "clarification_hint_v1", "incident_id": incident_id})}


def _opening_was_reported(session, incident_id, opening_query, source):
    attempts = session.execute(select(SolutionAttempt, Message.sequence).join(
        Message, SolutionAttempt.message_id == Message.id,
    ).join(Feedback, Feedback.attempt_id == SolutionAttempt.id).where(
        SolutionAttempt.incident_id == incident_id, SolutionAttempt.status == "SUCCEEDED",
        Feedback.helped.is_(True), Message.incident_id == incident_id,
    )).all()
    sequences = [sequence for attempt, sequence in attempts if attempt.evidence.get("talk_source") == source]
    if not sequences:
        return False
    return session.scalar(select(Message.id).where(
        Message.incident_id == incident_id, Message.role == "user", Message.content == opening_query,
        Message.sequence < max(sequences),
    ).limit(1)) is not None


def _plan_envelope(source):
    return isinstance(source, dict) and source.get("kind") == "verified_instruction_plan" and source.get("version") == 1


def _plan_proof(session, principal, incident_id, plan):
    incident = session.scalar(select(Incident).join(Conversation).where(
        Incident.id == incident_id, Conversation.user_id == principal,
        Incident.state == "RESOLVED", Incident.closed_at.is_not(None),
    ))
    if incident is None:
        return None
    attempts = session.scalars(select(SolutionAttempt).join(Feedback).where(
        SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "SUCCEEDED", Feedback.helped.is_(True),
    )).all()
    return incident if any(attempt.evidence.get("instruction_plan") == plan for attempt in attempts) else None


def _hint_state(state):
    if not isinstance(state, dict) or not state.get("goals"):
        return None
    retained = {key: copy.deepcopy(state.get(key, default)) for key, default in (
        ("facts", []), ("goals", {}), ("unknown_answers", []),
    )}
    retained.update(messages=[], question="", question_field=None, new_task=False)
    # A recalled interpretation must retain exact validated meaning. Sensitive
    # values can be redacted for memory elsewhere, but cannot train this shortcut.
    encoded = json.dumps(retained, ensure_ascii=False, sort_keys=True)
    if _redact(encoded) != encoded:
        return None
    try:
        task_query(retained)
    except (ValueError, TypeError):
        return None
    return retained


def remember_success(session, settings, principal, incident_id, key, source, *, opening_query=None,
                     plan=None, task_state=None):
    """Persist a reference only after an owned, explicit successful feedback row."""
    if not _valid_key(settings, key) or not _valid_source(settings, source):
        return None
    saved_plan = immutable_plan(plan) if plan is not None else None
    incident = (_plan_proof(session, principal, incident_id, saved_plan) if saved_plan
                else _proof(session, principal, incident_id, source))
    if incident is None:
        return None
    resolved = incident.closed_at
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    expires = resolved + timedelta(days=settings.memory_retention_days)
    if expires <= now():
        return None
    stored_key = dict(key)
    stored_source = source
    if saved_plan:
        if not isinstance(task_state, dict) or not task_state.get("goals"):
            return None
        stored_key["context_hash"] = _digest({"task_hash": key["context_hash"], "plan": saved_plan["digest"]})
        stored_source = {"kind": "verified_instruction_plan", "version": 1,
                         "task_hash": key["context_hash"], "plan": saved_plan}
    row = session.scalar(select(SuccessfulRetrieval).filter_by(user_id=principal, **stored_key))
    if row is None:
        row = SuccessfulRetrieval(user_id=principal, **stored_key)
        session.add(row)
    row.incident_id, row.source, row.expires_at = incident.id, copy.deepcopy(stored_source), expires
    hint_key = _hint_key(settings, opening_query, incident.id)
    if hint_key and _opening_was_reported(session, incident.id, opening_query, source):
        hint_source = copy.deepcopy(source)
        if saved_plan:
            retained_state = _hint_state(task_state)
            if retained_state is None:
                return row
            hint_source = {**copy.deepcopy(stored_source), "interpretation": retained_state}
        hint = session.scalar(select(SuccessfulRetrieval).filter_by(user_id=principal, **hint_key))
        if hint is None:
            hint = SuccessfulRetrieval(user_id=principal, **hint_key)
            session.add(hint)
        hint.incident_id, hint.source, hint.expires_at = incident.id, hint_source, expires
    return row


class SuccessfulTalkKnowledge:
    """Reuse a proven source, re-read current original; never cache an answer."""

    def __init__(self, delegate, factory, settings, principal, context):
        self.delegate, self.factory, self.settings = delegate, factory, settings
        self.principal, self.context = principal, context
        self.last_cache_key, self.cache_hit = None, False

    @property
    def uses_llm_selection(self):
        return bool(getattr(self.delegate, "uses_llm_selection", False))

    def _forget(self, key):
        with self.factory() as session, session.begin():
            row = session.scalar(select(SuccessfulRetrieval).filter_by(user_id=self.principal, **key))
            if row is not None:
                session.delete(row)

    def _lookup(self, key, excluded):
        with self.factory() as session, session.begin():
            row = session.scalar(select(SuccessfulRetrieval).filter_by(user_id=self.principal, **key))
            if row is None:
                return None
            source = copy.deepcopy(row.source)
            expires = row.expires_at
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if (expires <= now() or not _valid_source(self.settings, source)
                    or _proof(session, self.principal, row.incident_id, source) is None):
                session.delete(row)
                return None
            return None if source["file_id"] in excluded else source

    def search(self, query, excluded=(), fallback_query=None):
        problem = fallback_query if isinstance(fallback_query, str) and fallback_query.strip() else query
        self.last_cache_key = cache_key(self.settings, problem, self.context)
        self.cache_hit = False
        if self.last_cache_key:
            source = self._lookup(self.last_cache_key, excluded)
            if source:
                try:
                    approved = self.delegate.read(source["file_id"])
                    if (approved.document["source"] == source and source_compatible(
                        problem, approved.candidate.title + "\n" + approved.candidate.content, complete=True
                    )):
                        self.cache_hit = True
                        return approved
                except KnowledgeUnavailable:
                    pass
                self._forget(self.last_cache_key)
        return self.delegate.search(query, excluded=excluded, fallback_query=fallback_query)

    def confirmation_hint(self, query, excluded=()):
        """Suggest an owned past interpretation for confirmation, with no guide or cache hit."""
        key = _hint_key(self.settings, query, "lookup")
        if key is None:
            return None
        candidates = []
        plan_candidates = []
        with self.factory() as session, session.begin():
            rows = session.execute(select(SuccessfulRetrieval, Incident).join(
                Incident, SuccessfulRetrieval.incident_id == Incident.id,
            ).where(
                SuccessfulRetrieval.user_id == self.principal,
                SuccessfulRetrieval.scope_hash == key["scope_hash"],
                SuccessfulRetrieval.query_hash == key["query_hash"],
            ).order_by(Incident.closed_at.desc(), SuccessfulRetrieval.created_at.desc()).limit(21)).all()
            if len(rows) > 20:
                return None
            for row, incident in rows:
                hint_key = _hint_key(self.settings, query, incident.id)
                if row.context_hash != hint_key["context_hash"]:
                    continue
                source = copy.deepcopy(row.source)
                expires = row.expires_at
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if _plan_envelope(source):
                    plan, state = source.get("plan"), _hint_state(source.get("interpretation"))
                    if (expires <= now() or not isinstance(plan, dict) or state is None
                            or _plan_proof(session, self.principal, incident.id, plan) is None):
                        session.delete(row)
                    elif not any(part.get("source", {}).get("file_id") in excluded
                                 for part in plan.get("parts", [])):
                        plan_candidates.append((hint_key, plan, state))
                    continue
                if (expires <= now() or not _valid_source(self.settings, source)
                        or _proof(session, self.principal, incident.id, source) is None):
                    session.delete(row)
                    continue
                problem = _redact(incident.description.strip() or incident.subject.strip())
                if source["file_id"] not in excluded and problem and len(problem) <= 6000:
                    candidates.append((hint_key, source, problem))
        verified = {}
        for hint_key, plan, state in plan_candidates:
            try:
                parts = restore_plan(self.delegate, plan)
            except (KnowledgeUnavailable, KeyError, TypeError, ValueError):
                self._forget(hint_key)
                continue
            signature = _digest({"plan": plan["digest"], "state": state})
            verified[signature] = {"problem": task_query(state), "source": parts[0].approved.document["source"],
                                   "plan": serialize_plan(parts), "task_state": state}
        for hint_key, source, problem in candidates:
            try:
                approved = self.delegate.read(source["file_id"])
            except KnowledgeUnavailable:
                self._forget(hint_key)
                continue
            if approved.document["source"] != source:
                self._forget(hint_key)
                continue
            interpretation = {"source": source, "problem": normalized_query(problem)}
            verified.setdefault(_digest(interpretation), {"problem": problem, "source": source})
        return next(iter(verified.values())) if len(verified) == 1 else None

    def search_with_llm(self, query, context=None, excluded=(), fallback_query=None):
        """A previous success can nominate a source for the model to check again."""
        problem = fallback_query if isinstance(fallback_query, str) and fallback_query.strip() else query
        selection_context = dict(context or self.context)
        self.last_cache_key = cache_key(self.settings, problem, selection_context)
        self.cache_hit = False
        if self.last_cache_key and selection_context.get("instruction_planning"):
            previous_plan = self._lookup_plan(self.last_cache_key, selection_context, excluded)
            if previous_plan is not None:
                self.cache_hit = True
                return previous_plan
        verified = None
        if self.last_cache_key:
            source = self._lookup(self.last_cache_key, excluded)
            if source:
                try:
                    approved = self.delegate.read(source["file_id"])
                except KnowledgeUnavailable:
                    approved = None
                if approved is not None and approved.document["source"] == source:
                    verified = source
                    selection_context["verified_previous_source"] = {
                        "file_id": source["file_id"], "sha256": source["sha256"],
                    }
                else:
                    self._forget(self.last_cache_key)
        decision = self.delegate.search_with_llm(query, context=selection_context,
                                                 excluded=excluded, fallback_query=fallback_query)
        self.cache_hit = bool(verified and decision.approved is not None
                              and decision.approved.document["source"] == verified)
        return decision

    def _lookup_plan(self, key, context, excluded):
        candidates = []
        with self.factory() as session, session.begin():
            rows = session.scalars(select(SuccessfulRetrieval).where(
                SuccessfulRetrieval.user_id == self.principal,
                SuccessfulRetrieval.scope_hash == key["scope_hash"],
                SuccessfulRetrieval.query_hash == key["query_hash"],
            ).limit(101)).all()
            if len(rows) > 100:
                return None
            for row in rows:
                envelope = row.source
                if not _plan_envelope(envelope) or envelope.get("task_hash") != key["context_hash"]:
                    continue
                plan = envelope.get("plan")
                expires = row.expires_at
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if (expires <= now() or not isinstance(plan, dict)
                        or _plan_proof(session, self.principal, row.incident_id, plan) is None):
                    session.delete(row)
                    continue
                if not any(part.get("source", {}).get("file_id") in excluded for part in plan.get("parts", [])):
                    candidates.append((copy.deepcopy(plan), {name: getattr(row, name) for name in _KEY_FIELDS}))
        verified = {}
        for plan, stored_key in candidates:
            try:
                parts = restore_plan(self.delegate, plan)
            except (KnowledgeUnavailable, KeyError, TypeError, ValueError):
                self._forget(stored_key)
                continue
            requested = context.get("requested_goals", {})
            labels = {normalized_query(text): goal_id for goal_id, text in requested.items()}
            if (len(labels) != len(requested) or any(normalized_query(goal) not in labels
                                                   for part in parts for goal in part.goals)):
                continue
            covered = [normalized_query(goal) for part in parts for goal in part.goals]
            if len(covered) != len(requested) or set(covered) != set(labels):
                continue
            remapped = tuple(InstructionPlanPart(part.approved,
                tuple(labels[normalized_query(goal)] for goal in part.goals), part.goals,
                part.section_ids, part.sections) for part in parts)
            verified[plan["digest"]] = InstructionDecision("MATCH", approved=parts[0].approved,
                covered_goals=tuple(goal for part in parts for goal in part.goals), plan=remapped)
        return next(iter(verified.values())) if len(verified) == 1 else None

    def read(self, file_id):
        return self.delegate.read(file_id)

    def scanner_context(self):
        method = getattr(self.delegate, "scanner_context", None)
        return method() if callable(method) else {"status": "ok", "sources": [], "categories": [], "models": []}

    def is_current(self, source):
        return self.delegate.is_current(source)
