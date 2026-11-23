"""Bounded agent loop; reads have no side effects, replies commit through the Talk outbox."""

import copy
import json

import httpx
from pydantic import ValidationError

from app.core.logging import event
from app.services.chat.memory import _redact
from app.services.chat.policies import direct_intent, requests_support_form, resolution_feedback
from app.services.chat.talk_feedback import reorder_candidates, source_key
from app.services.chat.task_state import starts_new_task
from app.services.llm.support_agent import AgentAction, TaskMemory
from app.services.rag.talk import KnowledgeUnavailable


PAGE_CHARS = 12000
MODEL_UNAVAILABLE = "Сейчас не удалось подготовить ответ. Повторите вопрос немного позже."


def run_agent(message, flow, context, llm, knowledge, settings):
    from app.services.chat.talk_dialogue import Turn, UNAVAILABLE

    saved = copy.deepcopy(flow.get("agent", {}))
    legacy_memory = {"goal": str(flow.get("query", ""))[:1000],
                     "current_step": json.dumps(flow.get("instruction_plan", {}).get("shown", flow.get("shown", []))),
                     "next_step": str(flow.get("instruction_plan", {}).get("cursor", flow.get("next_offset", ""))),
                     "expected_reply": "Продолжение или вопрос по текущему шагу" if flow.get("source") else ""}
    memory = TaskMemory.model_validate(saved.get("memory", legacy_memory)).model_dump()
    recent = context.get("short_term", {}).get("messages", [])
    next_sequence = max((row.get("sequence", 0) for row in recent), default=0) + 1
    boundary = saved.get("start_sequence")
    if type(boundary) is int:
        recent = [row for row in recent if row.get("sequence", 0) >= boundary]
    classify = getattr(llm, "classify_task_relation", None)
    relation = "new" if starts_new_task(message) else "current"
    if relation == "current" and callable(classify) and (memory["goal"] or memory["expected_reply"]
                                                        or saved.get("sources") or flow.get("source")):
        try:
            relation = classify(message, memory, recent[-4:])
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            event("talk.agent_task_relation_failed", error_code=type(error).__name__)
            return Turn("REPLY", [(MODEL_UNAVAILABLE, None)], flow=copy.deepcopy(flow))
    if relation == "unclear":
        return Turn("REPLY", [("Это продолжение предыдущего вопроса или новая задача? Уточните, пожалуйста.", None)],
                    flow=copy.deepcopy(flow))
    if relation == "new":
        boundary, recent = max(next_sequence, boundary or 0), []
        flow, saved, memory = {}, {}, TaskMemory().model_dump()
        # Persist the boundary even if knowledge is unavailable for the new task.
        flow = {"agent": {"version": 1, "memory": memory, "sources": [], "start_sequence": boundary}}
    event("talk.agent_task_relation", relation=relation)
    # Older conversations enter with their context and frozen progress, not a new confirmation gate.
    legacy = {key: copy.deepcopy(flow[key]) for key in (
        "query", "source", "shown", "next_offset", "has_more", "instruction_plan",
    ) if key in flow} if not saved else {}
    feedback = context.get("agent_feedback", {})
    agent_context = ({"short_term": {"messages": recent}} if type(boundary) is int else
                     {key: value for key, value in context.items() if key != "agent_feedback"})
    payload = {"message": message, "context": agent_context, "memory": memory,
               "active_sources": saved.get("sources", []), "previous_progress": legacy, "tool_results": []}
    loaded, ranges = {}, {}
    allowed = {source["file_id"] for source in saved.get("sources", []) if isinstance(source, dict)
               and isinstance(source.get("file_id"), str)}
    if isinstance(flow.get("source"), dict):
        allowed.add(flow["source"]["file_id"])
    for part in flow.get("instruction_plan", {}).get("parts", []):
        allowed.add(part["source"]["file_id"])
    limit = settings.talk_agent_context_chars
    failure, knowledge_failed, verified_search = "CallLimit", False, False
    for index in range(settings.talk_agent_max_calls):
        if len(json.dumps(payload, ensure_ascii=False)) > limit:
            failure = "ContextBudget"
            break
        try:
            action = AgentAction.model_validate(llm.agent_action(payload))
        except (ValidationError, ValueError):
            failure = "InvalidActionContract"
            payload["tool_results"].append({"error": "InvalidActionContract", "retry": "Return valid AgentAction JSON"})
            continue
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as error:
            failure = type(error).__name__
            break
        event("talk.agent_action", action=action.action, call=index + 1)
        try:
            if action.action == "search_knowledge":
                if not action.query.strip():
                    raise ValueError("EmptySearchQuery")
                result = reorder_candidates(knowledge.search_candidates(action.query), feedback)
                verified_search |= not result.get("search_degraded", False)
                knowledge_failed |= bool(result.get("search_degraded", False))
                allowed.update(item["file_id"] for item in result["candidates"])
                payload["tool_results"].append({"tool": action.action, "query": action.query, "result": result})
            elif action.action == "read_instruction":
                if action.file_id not in allowed:
                    raise ValueError("SearchBeforeReadingUnknownSource")
                if action.file_id not in loaded:
                    loaded[action.file_id] = knowledge.read(action.file_id)
                approved = loaded[action.file_id]
                content = approved.document["content"]
                expected = ranges.get(action.file_id, 0)
                if action.offset != expected or action.offset >= len(content):
                    raise ValueError("ReadAtNextOffset")
                end = min(len(content), action.offset + PAGE_CHARS)
                result = {"source": approved.document["source"], "content": content[action.offset:end],
                          "offset": action.offset, "next_offset": end, "complete": end == len(content),
                          "images": [{"number": number, "description": picture.caption}
                                     for number, picture in approved.illustrations.items()] if action.offset == 0 else []}
                source = result["source"]
                votes = feedback.get(source_key(source))
                if votes and action.offset == 0:
                    result["feedback"] = votes
                record = {"tool": action.action, "file_id": action.file_id, "result": result}
                if len(json.dumps(payload, ensure_ascii=False)) + len(json.dumps(record, ensure_ascii=False)) > limit:
                    raise ValueError("DocumentExceedsContextBudget")
                ranges[action.file_id] = end
                payload["tool_results"].append(record)
            elif action.action == "request_support":
                from app.services.chat.talk_dialogue import _profile_form_request

                previous = next((row.get("content", "") for row in reversed(recent)
                                 if row.get("role") == "assistant" and row.get("content")), "")
                affirmative = message.casefold().strip(" .!\n") in {"да", "давайте", "согласен", "согласна", "да составьте"}
                accepted = requests_support_form(message) or direct_intent(message) == "HUMAN" or (
                    affirmative and "?" in previous and requests_support_form(previous))
                if not accepted or not action.consent_evidence or action.consent_evidence not in message:
                    raise ValueError("SupportRequestNeedsUserConsent")
                return _profile_form_request(flow, context)
            else:
                complete = {key: doc for key, doc in loaded.items() if ranges.get(key) == len(doc.document["content"])}
                if action.kind == "no_knowledge" and knowledge_failed and not verified_search:
                    raise KnowledgeUnavailable("SearchCouldNotBeVerified")
                turn = _reply(action, message, flow, memory, complete, knowledge)
                if action.new_task or type(boundary) is int:
                    turn.flow["agent"]["start_sequence"] = max(next_sequence, boundary or 0) if action.new_task else boundary
                return turn
        except (ValueError, KnowledgeUnavailable) as error:
            # Never return provider bodies, document text or credentials as an exception payload.
            code = error.code if isinstance(error, KnowledgeUnavailable) else str(error)
            knowledge_failed |= isinstance(error, KnowledgeUnavailable)
            failure = code if code.isascii() and code.isalnum() and len(code) <= 100 else type(error).__name__
            if code == "InstructionChangedBeforeReply":
                loaded.clear()
                ranges.clear()
            payload["tool_results"].append({"tool": action.action, "error": code[:100]})
    event("talk.agent_exhausted", calls=index + 1, error_code=failure, knowledge_failed=knowledge_failed)
    return Turn("REPLY", [(UNAVAILABLE if knowledge_failed else MODEL_UNAVAILABLE, None)], flow=copy.deepcopy(flow))


def _reply(action, message, flow, memory, complete, knowledge):
    from app.services.chat.talk_dialogue import Turn

    if not action.blocks or not any(block.text.strip() or block.images for block in action.blocks):
        raise ValueError("EmptyReply")
    if action.new_task and action.memory is None:
        raise ValueError("NewTaskRequiresMemory")
    if action.kind == "resolved" and resolution_feedback(message) != "SUCCESS":
        raise ValueError("ResolutionRequiresExplicitUserFeedback")
    if action.kind == "instruction" and not complete:
        raise ValueError("ReadFullInstructionBeforeReply")
    blocks, sources, used = [], [], {}
    image_count = 0
    for block in action.blocks:
        if len({ref.file_id for ref in block.references}) > 1:
            raise ValueError("SplitBlockBySource")
        if action.kind == "instruction" and not block.references:
            raise ValueError("InstructionBlockNeedsReferences")
        for reference in block.references:
            approved = complete.get(reference.file_id)
            if approved is None:
                raise ValueError("UnreadReference")
            text = approved.document["content"] + "\n" + "\n".join(p.caption for p in approved.illustrations.values())
            if reference.quote not in text:
                raise ValueError("QuoteNotInOriginal")
            used[reference.file_id] = approved
        if block.text.strip():
            blocks.append((_redact(block.text.strip()), None))
            sources.append(complete[block.references[0].file_id].document["source"] if block.references else None)
        for picture in block.images:
            approved = complete.get(picture.file_id)
            if approved is None or picture.number not in approved.illustrations:
                raise ValueError("UnknownOriginalImage")
            if not any(ref.file_id == picture.file_id for ref in block.references):
                raise ValueError("ImageNeedsSourceReference")
            image_count += 1
            if image_count > 6:
                raise ValueError("TooManyImages")
            used[picture.file_id] = approved
            blocks.append(("", approved.illustrations[picture.number]))
            sources.append(approved.document["source"])
    for approved in used.values():
        if not knowledge.is_current(approved.document["source"]):
            raise KnowledgeUnavailable("InstructionChangedBeforeReply")
    next_memory = action.memory.model_dump() if action.memory is not None else memory
    if len(json.dumps(next_memory, ensure_ascii=False)) > 8000:
        raise ValueError("TaskMemoryTooLarge")
    next_memory = {key: [_redact(item) for item in value] if isinstance(value, list) else _redact(value)
                   for key, value in next_memory.items()}
    previous_sources = flow.get("agent", {}).get("sources", []) or ([flow["source"]] if flow.get("source") else [])
    if action.new_task:
        previous_sources = []
    retained_sources = [doc.document["source"] for doc in used.values()] or previous_sources
    next_flow = {"agent": {"version": 1, "memory": next_memory,
                           "sources": copy.deepcopy(retained_sources)},
                 "query": next_memory["goal"] or (None if action.new_task else flow.get("query")) or message}
    if retained_sources:
        next_flow["source"] = retained_sources[0]
    turn = Turn("AGENT_RESOLVED" if action.kind == "resolved" else "AGENT_REPLY", blocks,
                flow=next_flow, block_sources=sources)
    turn.agent_documents = list(used.values())
    return turn
