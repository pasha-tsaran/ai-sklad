"""Bounded agent loop; reads have no side effects, replies commit through the Talk outbox."""

import copy
import json
import re

import httpx
from pydantic import ValidationError

from app.core.logging import event
from app.services.chat.memory import _redact
from app.services.chat.policies import direct_intent, requests_support_form, resolution_feedback
from app.services.chat.talk_feedback import reorder_candidates, source_key
from app.services.chat.task_state import starts_new_task
from app.services.llm.support_agent import AgentAction, TaskMemory
from app.services.rag.query import normalized_query, source_compatible
from app.services.rag.talk import KnowledgeUnavailable


PAGE_CHARS = 12000
MODEL_UNAVAILABLE = ("Сейчас не удалось подготовить ответ. Ваш вопрос сохранён. "
                     "Если ждать неудобно, напишите «позвать специалиста».")
OUTLOOK_LARGE_MAIL_FILE = "e70650e7-d3bb-4708-8bfe-9f2e748d7922"
OUTLOOK_LARGE_MAIL_SHA256 = "bf09266ba3fa350cc635d70650bfbfaf8b83ded4e78276d4e2dd5ab6585b82ed"
WINDOWS_SCREEN_FILE = "410c6e3f-f500-425f-9a43-fa204ae8338e"
WINDOWS_SCREEN_SHA256 = "014397db413283a9ebc802197a65b9f6ea7abe9c0e2956fe98d2523b9ab8bfa6"
_SCREEN_REQUEST = re.compile(
    r"(?:у меня )?(?:гаснет экран|экран гаснет)(?: через какое[- ]то время)?(?: что делать)?"
)
_WINDOWS_REPLY = re.compile(
    r"(?:windows|виндовс)(?: 10| 11)?|"
    r"(?:на )?(?:рабочем )?(?:компьютере|компьютер|ноутбуке|ноутбук|пк)(?: (?:с )?(?:windows|виндовс)(?: 10| 11)?)?"
)
_LARGE_MAIL_REQUEST = re.compile(
    r"(?:надо отследить большой размер писем(?: как это сделать)?|"
    r"как (?:найти|отследить) (?:большие|крупные) письма(?: в outlook)?|"
    r"(?:как )?создать папку(?: поиска)? для писем (?:больше|более) 1024(?: кб)?(?: в outlook)?)"
)


def _approved_outlook_reply(message, knowledge, start_sequence):
    if not _LARGE_MAIL_REQUEST.fullmatch(normalized_query(message)):
        return None
    doc = knowledge.read(OUTLOOK_LARGE_MAIL_FILE)
    source = doc.document["source"]
    if source["file_id"] != OUTLOOK_LARGE_MAIL_FILE or source["sha256"] != OUTLOOK_LARGE_MAIL_SHA256:
        return None
    content = doc.document["content"]
    steps = [line.strip() for line in content.splitlines() if re.match(r"^[1-5]\. ", line.strip())]
    if len(steps) != 5 or any(number not in doc.illustrations for number in (2, 3, 4, 5)):
        return None
    groups = [
        ("В классическом Outlook:\n\n" + "\n".join(steps[:2]), 2),
        (steps[2], 3),
        (steps[3], 4),
        (steps[4] + "\n\nПапка поиска показывает существующие письма. "
         "Удаление из результатов затрагивает сами письма.", 5),
    ]
    memory = TaskMemory(goal="Найти письма больше 1024 КБ в классическом Outlook")
    action = AgentAction.model_validate({
        "action": "reply", "kind": "instruction", "memory": memory.model_dump(), "blocks": [{
            "text": text, "references": [{"file_id": source["file_id"], "quote": steps[number - 1]}],
            "images": [{"file_id": source["file_id"], "number": number}],
        } for text, number in groups],
    })
    turn = _reply(action, message, {}, {}, {source["file_id"]: doc}, knowledge)
    turn.flow["agent"]["start_sequence"] = start_sequence
    return turn


def _approved_windows_screen_reply(message, memory, flow, knowledge):
    if (memory["expected_reply"] != "Тип устройства" or not _SCREEN_REQUEST.fullmatch(normalized_query(memory["goal"]))
            or not _WINDOWS_REPLY.fullmatch(normalized_query(message))):
        return None
    doc = knowledge.read(WINDOWS_SCREEN_FILE)
    source = doc.document["source"]
    if source["file_id"] != WINDOWS_SCREEN_FILE or source["sha256"] != WINDOWS_SCREEN_SHA256:
        return None
    # Only the employee-facing diagnostic steps, before the administrator section.
    body = doc.document["content"].split("Только для ИТ", 1)[0]
    steps = [line.strip() for line in body.splitlines() if re.match(r"^[1-4]\. ", line.strip())]
    if len(steps) != 4 or 1 not in doc.illustrations:
        return None
    blocks = [
        {"text": "На рабочем компьютере с Windows:\n\n" + "\n".join(steps[:2]), "references": [
            {"file_id": source["file_id"], "quote": steps[0]}]},
        {"text": steps[2], "references": [{"file_id": source["file_id"], "quote": steps[2]}],
         "images": [{"file_id": source["file_id"], "number": 1}]},
        {"text": steps[3], "references": [{"file_id": source["file_id"], "quote": steps[3]}]},
    ]
    action = AgentAction.model_validate({"action": "reply", "kind": "instruction", "blocks": blocks,
                                         "memory": {"goal": "Проверить, почему гаснет экран Windows"}})
    turn = _reply(action, message, flow, memory, {source["file_id"]: doc}, knowledge)
    if "start_sequence" in flow.get("agent", {}):
        turn.flow["agent"]["start_sequence"] = flow["agent"]["start_sequence"]
    turn.agent_question = memory["goal"]
    return turn


def run_agent(message, flow, context, llm, knowledge, settings, *, force_new=False):
    from app.services.chat.talk_dialogue import (
        Turn, UNAVAILABLE, _printer_clarification, _problem_query, _scanner_clarification, _simple_social, _step_confirmed,
    )

    saved = copy.deepcopy(flow.get("agent", {}))
    legacy_memory = {"goal": str(flow.get("query", ""))[:1000],
                     "current_step": json.dumps(flow.get("instruction_plan", {}).get("shown", flow.get("shown", []))),
                     "next_step": str(flow.get("instruction_plan", {}).get("cursor", flow.get("next_offset", ""))),
                     "expected_reply": "Продолжение или вопрос по текущему шагу" if flow.get("source") else ""}
    memory = TaskMemory.model_validate(saved.get("memory", legacy_memory)).model_dump()
    recent = context.get("short_term", {}).get("messages", [])
    next_sequence = max((row.get("sequence", 0) for row in recent), default=0) + 1
    try:
        direct_reply = _approved_outlook_reply(message, knowledge, next_sequence)
        if direct_reply is None and not force_new:
            direct_reply = _approved_windows_screen_reply(message, memory, flow, knowledge)
    except KnowledgeUnavailable as error:
        event("talk.agent_direct_instruction_failed", error_code=error.code)
    else:
        if direct_reply is not None:
            event("talk.agent_direct_instruction")
            return direct_reply
    feedback = context.get("agent_feedback", {})
    prefetch = getattr(knowledge, "prefetch_candidates", None)
    clarification = {} if force_new else flow
    pending_device = bool(clarification.get("clarification_keys"))
    search_message = _problem_query(message, clarification) if pending_device else message
    initial_result = None
    searches = {}
    if (callable(prefetch) and normalized_query(message) and not _simple_social(message)
            and not _step_confirmed(message) and not requests_support_form(message)):
        query = " ".join(search_message.split())[:1000]
        try:
            result = reorder_candidates(prefetch(query), feedback)
            searches[normalized_query(query)] = result
            initial_result = {"tool": "search_knowledge", "query": query, "result": result, "initial": True}
        except KnowledgeUnavailable as error:
            initial_result = {"tool": "search_knowledge", "error": error.code, "initial": True}
        event("talk.agent_initial_search", candidates=len(initial_result.get("result", {}).get("candidates", [])),
              error_code=initial_result.get("error"))
    boundary = saved.get("start_sequence")
    if type(boundary) is int:
        recent = [row for row in recent if row.get("sequence", 0) >= boundary]
    classify = getattr(llm, "classify_task_relation", None)
    relation = "new" if force_new or starts_new_task(message) else "current"
    if initial_result:
        # A general request like "configure a scanner" does not authorize a
        # procedure specific to one kind of scanner or work application.
        clues = {} if relation == "new" else clarification
        clarify = _scanner_clarification(message, clues, knowledge) or _printer_clarification(message, clues)
        if clarify:
            key = clarify.flow["clarification_keys"][-1]
            if key == "SCANNER_KIND" and key not in clues.get("clarification_keys", []):
                clarify.blocks = [("Какой у вас сканер — ручной для штрихкодов или для бумажных документов?", None)]
            clarify.flow["agent"] = {
                "version": 1, "sources": [],
                "start_sequence": (boundary or next_sequence) if search_message != message else next_sequence,
                "memory": TaskMemory(goal=clarify.flow["clarification_query"][:1000],
                                     expected_reply=clarify.blocks[0][0]).model_dump(),
            }
            clarify.action = "AGENT_REPLY"
            return clarify
    has_active_task = memory["goal"] or memory["expected_reply"] or saved.get("sources") or flow.get("source")
    route_unavailable = False
    if (relation == "current" and not (pending_device and search_message != message)
            and callable(classify) and has_active_task):
        try:
            relation = classify(message, memory, recent[-4:])
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            event("talk.agent_task_relation_failed", error_code=type(error).__name__)
            if initial_result and initial_result.get("result", {}).get("candidates"):
                relation = "unclear"
                route_unavailable = True
            else:
                return Turn("REPLY", [(MODEL_UNAVAILABLE, None)], flow=copy.deepcopy(flow))
    if relation == "unclear" and initial_result is None:
        return Turn("REPLY", [("Это продолжение предыдущего вопроса или новая задача? Уточните, пожалуйста.", None)],
                    flow=copy.deepcopy(flow))
    if relation == "new":
        boundary, recent = max(next_sequence, boundary or 0), []
        flow, saved, memory = {}, {}, TaskMemory().model_dump()
        # Persist the boundary even if knowledge is unavailable for the new task.
        flow = {"agent": {"version": 1, "memory": memory, "sources": [], "start_sequence": boundary}}
    if not memory["goal"] and _SCREEN_REQUEST.fullmatch(normalized_query(message)):
        flow = {"agent": {"version": 1, "sources": [], "start_sequence": boundary or next_sequence,
                          "memory": TaskMemory(goal=message[:1000], expected_reply="Тип устройства").model_dump()},
                "query": message}
        return Turn("AGENT_REPLY", [("Где гаснет экран — на компьютере, на сканере или на телефоне?", None)],
                    flow=flow)
    event("talk.agent_task_relation", relation=relation)
    # Older conversations enter with their context and frozen progress, not a new confirmation gate.
    legacy = {key: copy.deepcopy(flow[key]) for key in (
        "query", "source", "shown", "next_offset", "has_more", "instruction_plan",
    ) if key in flow} if not saved else {}
    agent_context = ({"short_term": {"messages": recent}} if type(boundary) is int else
                     {key: value for key, value in context.items() if key != "agent_feedback"})
    payload = {"message": message, "context": agent_context, "memory": memory,
               "active_sources": saved.get("sources", []), "previous_progress": legacy,
               "tool_results": [initial_result] if initial_result else []}
    loaded, ranges = {}, {}
    allowed = {source["file_id"] for source in saved.get("sources", []) if isinstance(source, dict)
               and isinstance(source.get("file_id"), str)}
    if isinstance(flow.get("source"), dict):
        allowed.add(flow["source"]["file_id"])
    for part in flow.get("instruction_plan", {}).get("parts", []):
        allowed.add(part["source"]["file_id"])
    if route_unavailable:
        # Current-search candidates remain allowed. Previous sources require
        # rediscovery when we could not establish that the topic continued.
        allowed.clear()
        payload.update(active_sources=[], previous_progress={})
    limit = settings.talk_agent_context_chars
    first = (initial_result or {}).get("result")
    if first is not None:
        allowed.update(item["file_id"] for item in first["candidates"])
    failure = "CallLimit"
    model_failed = False
    knowledge_failed = bool(initial_result and (first is None or first.get("search_degraded")))
    verified_search = first is not None and not first.get("search_degraded", False)
    tried_reformulation = False
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
            model_failed = True
            break
        event("talk.agent_action", action=action.action, call=index + 1)
        try:
            if action.action == "search_knowledge":
                if not action.query.strip():
                    raise ValueError("EmptySearchQuery")
                key = normalized_query(action.query)
                result = searches.get(key)
                if result is None:
                    result = reorder_candidates(knowledge.search_candidates(action.query), feedback)
                    searches[key] = result
                    tried_reformulation |= not result.get("search_degraded", False) or bool(result["candidates"])
                else:
                    raise ValueError("SearchAlreadyReturnedUseCandidatesOrRephrase")
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
                if action.kind == "no_knowledge" and first is not None and not tried_reformulation:
                    raise ValueError("TrySearchByDifferentWordsOrActionsBeforeNoKnowledge")
                if action.kind == "no_knowledge" and knowledge_failed and not verified_search:
                    raise KnowledgeUnavailable("SearchCouldNotBeVerified")
                if action.kind == "instruction" and callable(prefetch):
                    used_ids = {ref.file_id for block in action.blocks for ref in block.references}
                    if len(used_ids) == 1:
                        approved = complete.get(next(iter(used_ids)))
                        if approved is not None and not source_compatible(
                                search_message, approved.candidate.title + "\n" + approved.document["content"]):
                            raise ValueError("SourceConflictsWithCurrentRequest")
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
    return Turn("REPLY", [(UNAVAILABLE if knowledge_failed and not model_failed else MODEL_UNAVAILABLE, None)],
                flow=copy.deepcopy(flow))


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
