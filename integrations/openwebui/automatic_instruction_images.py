"""
title: AI Support — фотографии к ответу автоматически
author: AI Support
description: Автоматически прикладывает фотографии утверждённой инструкции и продолжает её по шагам.
required_open_webui_version: 0.11.0
version: 0.3.0
"""

# Install as an admin Function (Filter), scoped to ai-support, not as a Tool.
# Reuses the authorized current request's instruction toolkit. No new packages.
import copy
import json
import logging
import re
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
IMAGE_TOOL = "show_instruction_images"
READ_TOOL = "read_approved_instruction"


def _uuid(value):
    try:
        return isinstance(value, str) and str(UUID(value)) == value.lower()
    except (ValueError, TypeError, AttributeError):
        return False


def _json(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        value = "".join(v.get("text", "") for v in value if isinstance(v, dict))
    if not isinstance(value, str) or len(value) > 1_000_000:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def tool_results(message):
    """Only structured, paired native calls/results, never prose pretending to be a tool."""
    output = message.get("output") or []
    calls = {item.get("call_id") or item.get("id"): item for item in output
             if isinstance(item, dict) and item.get("type") == "function_call"}
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        call = calls.get(item.get("call_id"))
        if not call or item.get("status") not in (None, "completed"):
            continue
        raw = item.get("output", "")
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
        result, arguments = _json(raw), _json(call.get("arguments", "{}"))
        if isinstance(arguments, dict):
            yield call.get("name"), arguments, result


def _allowed_file(file_id, filename):
    return (_uuid(file_id) and isinstance(filename, str) and filename.lower().endswith(".docx")
            and not filename.casefold().startswith(("ит_", "справочник_")))


def selected_source(message):
    """Prefer a real read, then a unique cited source. Search hits are not all selected."""
    read_ids = []
    for name, arguments, result in tool_results(message):
        if not isinstance(result, dict) or "error" in result:
            continue
        source = result.get("source", {}) if name == READ_TOOL else result
        file_id = source.get("file_id") if name == READ_TOOL else source.get("id")
        if (name in {READ_TOOL, "view_file", "view_knowledge_file"}
                and arguments.get("file_id") == file_id
                and _allowed_file(file_id, source.get("filename"))):
            read_ids.append(file_id)
    unique = list(dict.fromkeys(read_ids))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return None
    sources = message.get("sources") or []
    eligible = []
    cited = []
    text = message.get("content") or ""
    references = {int(value) for value in re.findall(r"\[(\d+)\]", text)}
    for index, entry in enumerate(sources, 1):
        if not isinstance(entry, dict):
            continue
        source = entry.get("source") or {}
        file_id, filename = source.get("id"), source.get("name")
        if not _allowed_file(file_id, filename):
            continue
        eligible.append(file_id)
        if index in references or filename in text:
            cited.append(file_id)
    chosen = list(dict.fromkeys(cited))
    if len(chosen) == 1:
        return chosen[0]
    return None


def _normal(text):
    return " ".join(re.findall(r"[a-zа-яё0-9]+", str(text).casefold().replace("ё", "е")))


_QUERY_NOISE = {
    "работаю", "нужно", "нужен", "хочу", "помогите", "подскажите", "пожалуйста", "уже", "установлен",
    "плохо", "разбираюсь", "настройках", "настроить", "настройка", "чего", "начать", "моего", "моем",
    "рабочего", "рабочем", "места", "месте", "версии", "версия", "драйвер", "программе", "программа",
    "документ", "документы", "инструкция", "инструкции", "показать", "покажи", "фото", "фотографии",
    "скриншот", "первые", "действия", "правильно", "почему", "который",
    "спасибо", "привет", "здравствуйте", "получилось", "готово", "дальше", "продолжить", "продолжай",
    "это", "как", "что", "где", "для", "при", "или", "мой", "мне", "меня", "есть", "его", "вот",
    "docx", "файл", "файла", "базе", "база", "знаний",
}


def search_terms(text):
    terms = set()
    for word in _normal(text).split():
        if word in _QUERY_NOISE or word.isdigit() or (len(word) < 3 and word != "1с"):
            continue
        if word.startswith(("изображ", "картин", "рисунк")):
            terms.add("image")
        elif word.startswith(("акробат", "acrobat")):
            terms.add("acrobat")
        else:
            terms.add(word[:5])
    return terms


def choose_search_source(query, chunks):
    """Use clear lexical evidence within scoped semantic hits; never take an arbitrary first hit."""
    terms = search_terms(query)
    if len(terms) < 2 or not isinstance(chunks, list):
        return None
    grouped = {}
    for chunk in chunks[:32]:
        if not isinstance(chunk, dict) or chunk.get("type") in {"external", "note"}:
            continue
        file_id, filename = chunk.get("file_id"), chunk.get("source")
        if not _allowed_file(file_id, filename):
            continue
        entry = grouped.setdefault(file_id, {"title": set(), "content": set()})
        entry["title"].update(search_terms(filename))
        entry["content"].update(search_terms(str(chunk.get("content", ""))[:50_000]))
    ranked = []
    for file_id, entry in grouped.items():
        title_hits, content_hits = terms & entry["title"], terms & entry["content"]
        if not title_hits or len(title_hits | content_hits) < 2:
            continue
        ranked.append((len(title_hits) * 3 + len(content_hits), file_id))
    ranked.sort(reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 2):
        return None
    return ranked[0][1]


def problem_query(messages, user_entries):
    index, last = user_entries[-1]
    text = str(last.get("content", ""))
    if re.fullmatch(r"(?:спасибо|благодарю|привет|здравствуйте|добрый день|все получилось|готово)[\s!.]*", text.casefold()):
        return text
    # A short answer to the immediately preceding clarification belongs to that
    # question, not to an arbitrary older incident in this chat.
    if len(_normal(text).split()) <= 4 and len(user_entries) >= 2:
        previous = next((m for m in reversed(messages[:index]) if m.get("role") == "assistant"), {})
        if "?" in str(previous.get("content", "")):
            text = str(user_entries[-2][1].get("content", ""))[-4000:] + "\n" + text
    return text[-6000:]


def has_technical_actions(text):
    return bool(re.search(r"(?m)^\s*\d+[.)]\s|```|(?:нажмите|откройте|выберите|введите|удалите|снимите|перейдите|включите)",
                          str(text).casefold()))


def missing_source_reply(body, index, unavailable=False):
    text = body["messages"][index].get("content", "")
    if not has_technical_actions(text) and str(text).count("?") == 1:
        question = re.search(r"\b(?:(?:в|на|к|с|для)\s+)?(?:какой|какая|какое|какие|какую|каком|"
                             r"что|где|когда|как|сколько|есть\s+ли|видно\s+ли)\b[^?]*\?", str(text), re.I)
        if question:
            # A clarification is allowed, an invented diagnosis preceding it is not.
            actual_question = question.group().strip()
            return body if actual_question == str(text).strip() else neutral_response(body, index, actual_question)
    message = ("Сейчас не удалось проверить базу знаний. Я не могу предложить технические действия без проверенной инструкции."
               if unavailable else "В утверждённой базе знаний не удалось найти подходящую инструкцию. Требуется помощь специалиста технической поддержки.")
    return neutral_response(body, index, message)


async def report_status(emitter, description, done):
    try:
        await emitter({"type": "status", "data": {"description": description, "done": done}})
    except Exception:
        log.debug("AI Support: status delivery unavailable")


def is_confirmation(text):
    text = _normal(text)
    if re.search(r"\b(не|нет|ошибка|трудно|проблема|почему|где|как)\b", text):
        return False
    return bool(re.fullmatch(r"(?:да[ ,]*)?(?:готово|готов|сделал|сделала|выполнил|выполнила|"
                             r"дошел|получилось|дальше|продолжай|продолжить|все сделал|все готово|да)", text))


def is_ui_question(text):
    return bool(re.search(r"(?:не\s+(?:могу\s+)?(?:найти|нахожу|вижу|видно|понимаю)|где\s+\S|"
                          r"нет\s+(?:такого|такой|этого|этой|пункта|кнопки|меню)|"
                          r"как\s+(?:найти|открыть)|что\s+нажать|какую\s+кнопку)", str(text).casefold()))


def applicable_images(document, user_history):
    """Omit explicitly decorative/admin-only pictures, not arbitrary guessed visual details."""
    mentions = re.findall(r"драйвер[^\n.!?]{0,55}?\b(?:не\s+)?установлен\b", user_history.casefold())
    driver_installed = bool(mentions and not re.search(r"\bне\s+установлен", mentions[-1]))
    selected = []
    for image in document.get("images", []):
        caption = image.get("caption", "")
        text = _normal(caption)
        if ("служит оформлением" in text or "не содержит настройки" in text
                or re.search(r"^логотип\b", text)):
            continue
        if "драйвер" in text and "установ" in text and "специалист" in text:
            continue
        if driver_installed and ("драйвер не установлен" in text or "версия не определена" in text):
            continue
        number = image.get("number")
        if type(number) is int and number > 0 and isinstance(caption, str) and caption.strip():
            selected.append(number)
    return selected


def clarification_images(document, text, fallback):
    words = {word[:7] for word in _normal(text).split() if len(word) >= 4}
    words -= {"найти", "вижу", "могу", "понимаю", "находит", "нажать", "кнопку", "кнопка"}
    scored = []
    for image in document.get("images", []):
        tokens = {word[:7] for word in _normal(image.get("caption", "")).split() if len(word) >= 4}
        scored.append((len(tokens & words), image.get("number")))
    best = max((score for score, _ in scored), default=0)
    if best:
        return [number for score, number in scored if score == best][:2]
    return fallback[:1]


def _source_notes(document):
    text = document.get("content", "").split("Фотографии и пояснения", 1)[0]
    notes = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph or re.match(r"^\d+[.)]\s", paragraph):
            continue
        if re.search(r"(?:перед |если |только |верси|процедура|выполняйте|не является|не требование)", paragraph.casefold()):
            notes.append(paragraph)
    return "\n\n".join(dict.fromkeys(notes))


def procedure_image_groups(document, images):
    """Keep original numbered actions, including unpictured entry/prerequisite steps.

    Prepared DOCX captions describe the screenshots, not necessarily every action
    (for example, Ctrl+P precedes the only Acrobat screenshot). Bind original steps
    to the closest caption in document order instead of replacing them with captions.
    """
    text = document.get("content", "").split("Фотографии и пояснения", 1)[0]
    steps = [re.sub(r"^\d+[.)]\s*", "", p.strip()) for p in text.split("\n\n")
             if re.match(r"^\d+[.)]\s", p.strip())]
    groups = {image["number"]: [] for image in images}
    if not steps:
        return [[] for image in images]
    original_images = document.get("images", [])
    if len(original_images) == 1:
        return [steps if image["number"] == original_images[0]["number"] else [] for image in images]
    stop = {"нажмите", "откройт", "выберит", "рисунке", "снимке", "иллюстр", "картинк", "окне", "окно",
            "на", "по", "из", "со", "во", "до", "за", "не", "при", "для", "это", "что", "как", "его", "ее", "или"}

    def words(value):
        return {word[:7] for word in _normal(value).split() if len(word) >= 2 or word.isdigit()} - stop

    tokens = [words(image.get("caption", "")) for image in original_images]
    frequencies = {word: sum(word in token_set for token_set in tokens) for word in set().union(*tokens)}
    candidates = [(0, 0, 1.0)]  # The first original action is an entry/prerequisite, never a later-page action.
    for step_index, step in enumerate(steps[1:], 1):
        terms = words(step)
        scores = [sum(1 / frequencies[word] for word in sorted(token_set & terms)) for token_set in tokens]
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        if ranked and scores[ranked[0]] - (scores[ranked[1]] if len(ranked) > 1 else 0) >= 0.25:
            candidates.append((step_index, ranked[0], scores[ranked[0]]))
    # Find the longest coherent sequence of source-order anchors. A misleading
    # early keyword shared with the final picture must not pull every later step
    # into that final page. Unmatched prerequisites stay with the preceding group.
    paths = []
    for index, (_, image_index, _) in enumerate(candidates):
        prior = [paths[j] for j in range(index) if candidates[j][1] <= image_index]
        chain = max(prior, key=lambda value: (len(value), sum(candidates[j][2] for j in value)), default=[])
        paths.append(chain + [index])
    chain = max(paths, key=lambda value: (len(value), sum(candidates[j][2] for j in value)), default=[])
    anchors = {candidates[i][0]: candidates[i][1] for i in chain}
    previous = 0
    for index, step in enumerate(steps):
        previous = anchors.get(index, previous)
        if original_images and original_images[previous].get("number") in groups:
            groups[original_images[previous]["number"]].append(step)
    return [groups[image["number"]] for image in images]


def render_steps(document, images, has_more, clarification=False):
    parts = []
    notes = _source_notes(document)
    if notes:
        parts.append(notes)
    groups = [[] for image in images] if clarification else procedure_image_groups(document, images)
    step_number = 1
    for index, (image, steps) in enumerate(zip(images, groups), 1):
        if not _uuid(image.get("file_id")):
            raise ValueError("An inline illustration requires its real private file UUID")
        parts.append(f"**Шаг {index}**")
        if steps:
            parts.append("\n".join(f"{number}. {step}" for number, step in enumerate(steps, step_number)))
            step_number += len(steps)
        if not any(_normal(image["caption"]) == _normal(step) for step in steps):
            parts.append(image["caption"])
        parts.append(f"![Иллюстрация {image['number']}](/api/v1/files/{image['file_id']}/content)")
    if clarification:
        parts.append("Удалось найти этот пункт?")
    elif has_more:
        parts.append("Когда выполните эти шаги, напишите «Готово» — покажу продолжение.")
    else:
        parts.append("Удалось выполнить эти действия?")
    return "\n\n".join(parts)


def replace_text(message, text):
    message["content"] = text
    # Native UI renders output before content. Replace user-facing message nodes,
    # retaining real tool call/result traces and keeping both representations equal.
    output = [item for item in message.get("output", [])
              if isinstance(item, dict) and item.get("type") != "message"]
    output.append({"type": "message", "id": "msg_" + uuid4().hex, "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": text}]})
    message["output"] = output


def record_call(message, name, arguments, result):
    call_id = "call_ai_support_" + uuid4().hex
    output = message.setdefault("output", [])
    index = next((i for i, item in enumerate(output) if isinstance(item, dict) and item.get("type") == "message"), len(output))
    output[index:index] = [
        {"type": "function_call", "id": call_id, "call_id": call_id, "name": name,
         "arguments": json.dumps(arguments, ensure_ascii=False), "status": "completed"},
        {"type": "function_call_output", "id": "fc_" + uuid4().hex, "call_id": call_id,
         "output": [{"type": "input_text", "text": json.dumps(result, ensure_ascii=False)}], "status": "completed"},
    ]


def annotate_flow(message, images, flow):
    wanted = [item.get("file_id") for item in images]
    for item in message.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        raw = item.get("output")
        result = _json(raw)
        if (isinstance(result, dict) and result.get("displayed") is True
                and [image.get("file_id") for image in result.get("images", [])] == wanted):
            result["_ai_support_flow"] = flow
            item["output"] = [{"type": "input_text", "text": json.dumps(result, ensure_ascii=False)}]
            return


def neutral_response(body, index, text):
    updated = copy.deepcopy(body)
    replace_text(updated["messages"][index], text)
    return updated


def limit_clarifications(body, index):
    text = body["messages"][index].get("content", "")
    if not isinstance(text, str) or text.count("?") <= 1:
        return body
    # Limit question-only replies. Never cut procedural steps, quoted warnings or
    # code to obtain a shorter answer; source rendering below handles procedures.
    if re.search(r"(?m)^\s*\d+[.)]\s|```|(?:нажмите|откройте|выберите|введите|удалите|снимите)", text.casefold()):
        return body
    return neutral_response(body, index, text.split("?", 1)[0].rstrip() + "?")


class _Runtime:
    def __init__(self, body, metadata, user, request, emitter, image_tool_id):
        self.body, self.metadata, self.user_context = body, metadata, user
        self.request, self.emitter, self.image_tool_id = request, emitter, image_tool_id
        self.user = None
        self.chat = None
        self.read_tool = None
        self.image_tool = None
        self.search_tool = None
        self.owner_authorized = False

    async def authorize(self):
        from open_webui.models.chats import Chats
        from open_webui.models.users import Users

        self.user = await Users.get_user_by_id(self.user_context["id"])
        if not self.user or self.user.role not in {"admin", "user"}:
            return False
        self.chat = await Chats.get_chat_by_id_and_user_id(self.metadata["chat_id"], self.user.id)
        if not self.chat:
            return False
        self.owner_authorized = True
        tools = self.metadata.get("tools") or {}
        readers, images = [], []
        for tool in tools.values():
            if not isinstance(tool, dict) or not callable(tool.get("callable")):
                continue
            if (tool.get("tool_id") == "builtin:query_knowledge_files"
                    and tool.get("spec", {}).get("name") == "query_knowledge_files"):
                self.search_tool = tool
            if self.image_tool_id and tool.get("tool_id") != self.image_tool_id:
                continue
            name = tool.get("spec", {}).get("name")
            if name == READ_TOOL:
                readers.append(tool)
            elif name == IMAGE_TOOL:
                images.append(tool)
        pairs = [(reader, image) for reader in readers for image in images
                 if reader.get("tool_id") == image.get("tool_id") and reader.get("tool_id")]
        if len(pairs) != 1:
            return False
        self.read_tool, self.image_tool = pairs[0]
        return True

    async def read(self, file_id):
        return _json(await self.read_tool["callable"](file_id=file_id))

    async def search(self, query):
        if self.search_tool is None:
            return {"status": "unavailable", "chunks": []}
        try:
            chunks = _json(await self.search_tool["callable"](query=query, count=8))
            return {"status": "ok", "chunks": chunks} if isinstance(chunks, list) else {"status": "unavailable", "chunks": []}
        except Exception:
            return {"status": "unavailable", "chunks": []}

    async def show(self, file_id, selection, offset):
        arguments = {"file_id": file_id, "image_numbers": selection, "offset": offset}
        if offset:
            # This private flag is absent from the model's public schema. Only the
            # controller reaches a nonzero offset after checking the user's reply.
            arguments["__allow_continuation__"] = True
        return _json(await self.image_tool["callable"](**arguments))

    async def verify(self, message, result):
        """A model assertion is not delivery: bind saved private PNGs to this reply/source."""
        saved = self.chat.chat.get("history", {}).get("messages", {}).get(message.get("id"), {})
        return await self._verify_saved(saved, result)

    async def _verify_saved(self, saved, result):
        from open_webui.models.files import Files

        attached = set()
        for file in saved.get("files", []):
            match = re.fullmatch(r"/api/v1/files/([a-fA-F0-9-]+)/content", file.get("url", ""))
            if match:
                attached.add(match.group(1))
        source = result.get("source", {})
        images = result.get("images", [])
        if not images or len(images) > 6 or not _uuid(source.get("file_id")):
            return False
        for image in images:
            file_id = image.get("file_id")
            if not _uuid(file_id):
                return False
            inline = (result.get("_ai_support_flow", {}).get("layout") == "inline"
                      and f"![Иллюстрация {image.get('number')}](/api/v1/files/{file_id}/content)"
                      in saved.get("content", ""))
            if file_id not in attached and not inline:
                return False
            file = await Files.get_file_by_id(file_id)
            provenance = (file.meta or {}).get("data", {}) if file else {}
            if (not file or file.user_id != self.user.id
                    or provenance.get("ai_support_tool") != "approved_instruction_images"
                    or provenance.get("source_file_id") != source.get("file_id")
                    or provenance.get("source_sha256") != source.get("sha256")
                    or provenance.get("illustration_number") != image.get("number")
                    or provenance.get("caption") != image.get("caption")):
                return False
        return True

    async def present_inline(self, message, result):
        """Keep private files, move only this verified batch from the gallery into text."""
        from open_webui.models.chats import Chats

        try:
            saved = await Chats.get_message_by_id_and_message_id(self.metadata["chat_id"], message["id"])
            if not isinstance(saved, dict) or not await self._verify_saved(saved, result):
                return False
            urls = {f"/api/v1/files/{image['file_id']}/content" for image in result["images"]}
            remaining = [file for file in saved.get("files", []) if file.get("url") not in urls]
            patch = {"content": message["content"], "output": message["output"], "files": remaining}
            updated_chat = await Chats.upsert_message_to_chat_by_id_and_message_id(
                self.metadata["chat_id"], message["id"], patch, touch=False
            )
            if updated_chat is None:
                return False
        except Exception:
            log.warning("AI Support automatic images: inline layout could not be saved")
            return False
        self.chat = updated_chat
        message["files"] = remaining
        try:
            # v0.11 renders files above all text. Persist first, then remove only
            # our gallery entries live; the returned outlet also carries files.
            await self.emitter({"type": "chat:message:files", "data": {"files": remaining}})
        except Exception:
            log.warning("AI Support automatic images: saved inline layout awaits UI refresh")
        return True


class Filter:
    class Valves(BaseModel):
        priority: int = 20
        enabled_model_ids: list[str] = Field(default_factory=lambda: ["ai-support"])
        image_tool_id: str = Field(default="", description="Обычно пусто: используется разрешённый инструмент текущего запроса.")

    def __init__(self):
        self.valves = self.Valves()

    async def outlet(self, body: dict, __user__=None, __request__=None, __metadata__=None,
                     __event_emitter__=None, __model__=None) -> dict:
        metadata = __metadata__ if isinstance(__metadata__, dict) else {}
        model_id = (__model__.get("id") if isinstance(__model__, dict) else None) or body.get("model")
        if (model_id not in self.valves.enabled_model_ids or not _uuid(metadata.get("chat_id"))
                or not metadata.get("session_id") or not metadata.get("message_id")
                or not isinstance(__user__, dict) or not __user__.get("id")
                or not __request__ or not callable(__event_emitter__)
                or metadata.get("task") or metadata.get("internal")):
            log.debug("AI Support automatic images: skipped outside enabled owned UI scope")
            return body  # Never alter backend JSON, temporary/API/channel or unrelated model responses.
        messages = body.get("messages", [])
        current_index = next((i for i in range(len(messages) - 1, -1, -1)
                              if messages[i].get("role") == "assistant"
                              and messages[i].get("id") == metadata["message_id"]), None)
        if current_index is None:
            return body
        user_entries = [(i, m) for i, m in enumerate(messages[:current_index]) if m.get("role") == "user"]
        if not user_entries:
            return body
        user_text = user_entries[-1][1].get("content", "")
        keep_current = is_ui_question(user_text) or is_confirmation(user_text)
        runtime = _Runtime(body, metadata, __user__, __request__, __event_emitter__, self.valves.image_tool_id)
        status_started = False
        completion = "Проверка инструкции завершена"
        try:
            if not await runtime.authorize():
                if getattr(runtime, "owner_authorized", False):
                    # A duplicate old completed endpoint may omit tools. Keep an
                    # already processed delivery only if private PNG proof survives.
                    for name, _, result in tool_results(messages[current_index]):
                        if (name == IMAGE_TOOL and isinstance(result, dict)
                                and result.get("_ai_support_flow", {}).get("processed")
                                and await runtime.verify(messages[current_index], result)):
                            return body
                    await report_status(__event_emitter__, "Обработчик знаний не подключён к этому чату", True)
                    return neutral_response(body, current_index,
                                            "Не удалось проверить инструкцию: обработчик знаний не подключён к этому чату. Обратитесь к администратору поддержки.")
                return body
            body = limit_clarifications(body, current_index)
            messages = body["messages"]
            current = messages[current_index]
            existing = None
            # Render a real manual delivery once; repeated outlet runs do not upload/render twice.
            for name, arguments, result in tool_results(current):
                if (name == IMAGE_TOOL and isinstance(result, dict) and result.get("displayed") is True
                        and await runtime.verify(current, result)):
                    if (result.get("_ai_support_flow", {}).get("processed")
                            and result.get("_ai_support_flow", {}).get("layout") == "inline"):
                        return body
                    existing = (arguments, result)
                    break
                if name == IMAGE_TOOL and isinstance(result, dict) and result.get("status") == "delivery_unknown":
                    return body
            query = problem_query(messages, user_entries)
            if not keep_current and not search_terms(query) and not has_technical_actions(current.get("content", "")):
                return body  # Greetings/thanks are not a request for an arbitrary procedure.
            await report_status(__event_emitter__, "Проверяю инструкцию и фотографии в базе знаний…", False)
            status_started = True
            previous = None
            previous_index = None
            for message in reversed(messages[:current_index]):
                if message.get("role") != "assistant":
                    continue
                for name, arguments, result in reversed(list(tool_results(message))):
                    if (name == IMAGE_TOOL and isinstance(result, dict) and result.get("displayed") is True
                            and await runtime.verify(message, result)):
                        previous = (message, arguments, result)
                        break
                # An intervening answer starts a new flow. Never resume an older unrelated guide.
                previous_index = messages.index(message)
                if not previous and is_ui_question(user_text):
                    prior_id = selected_source(message)
                    if prior_id:
                        previous = (message, {"image_numbers": "auto"},
                                    {"source": {"file_id": prior_id}, "images": [], "has_more": False})
                break
            clarification = is_ui_question(user_text) and previous is not None
            advance = is_confirmation(user_text) and previous is not None
            flow = previous[2].get("_ai_support_flow", previous[2]) if previous else {}
            if advance and not flow.get("has_more"):
                return body
            pinned = clarification or (advance and flow.get("has_more"))
            file_id = (previous[2].get("source", {}).get("file_id") if pinned else
                       existing[1].get("source", {}).get("file_id") if existing else selected_source(current))
            search_trace = None
            if not _uuid(file_id):
                if len(search_terms(query)) < 2:
                    return missing_source_reply(body, current_index)
                searched = await runtime.search(query)
                chunks = searched.get("chunks", []) if isinstance(searched, dict) else []
                available = isinstance(searched, dict) and searched.get("status") == "ok"
                file_id = choose_search_source(query, chunks) if available else None
                # Persist a bounded source summary, not the whole search corpus in
                # every future model context. The approved reader checks membership.
                search_trace = {"status": "ok" if available else "unavailable", "sources": [
                    {"file_id": chunk.get("file_id"), "source": chunk.get("source")}
                    for chunk in chunks[:32] if isinstance(chunk, dict)
                    and _allowed_file(chunk.get("file_id"), chunk.get("source"))
                ]}
                if not file_id:
                    completion = "Подходящая инструкция не найдена" if available else "Проверка базы знаний недоступна"
                    updated = missing_source_reply(body, current_index, unavailable=not available)
                    if updated is body:
                        updated = copy.deepcopy(body)
                    record_call(updated["messages"][current_index], "query_knowledge_files",
                                {"query": query, "count": 8}, search_trace)
                    return updated
            document = await runtime.read(file_id)
            if (not isinstance(document, dict) or document.get("status") != "ready"
                    or document.get("source", {}).get("file_id") != file_id):
                if pinned or existing:
                    return neutral_response(body, current_index,
                                            "Не удалось проверить текущую инструкцию. Я не могу заменить её другой процедурой без подтверждённых сведений.")
                completion = "Не удалось прочитать утверждённую инструкцию"
                return missing_source_reply(body, current_index, unavailable=True)
            old_hash = previous[2].get("source", {}).get("sha256") if pinned else None
            if old_hash and document["source"].get("sha256") != old_hash:
                text = "Инструкция изменилась. Не могу продолжить прежнюю последовательность без проверки актуального документа."
                updated = copy.deepcopy(body)
                replace_text(updated["messages"][current_index], text)
                return updated
            start_user_id = flow.get("start_user_id") if pinned else user_entries[-1][1].get("id")
            if pinned and not start_user_id and previous_index is not None:
                prior_users = [m for m in messages[:previous_index] if m.get("role") == "user"]
                start_user_id = prior_users[-1].get("id") if prior_users else None
            start_index = next((i for i, m in user_entries if start_user_id and m.get("id") == start_user_id),
                               user_entries[-1][0])
            user_context = "\n".join(str(m.get("content", "")) for i, m in user_entries if i >= start_index)
            eligible = applicable_images(document, user_context)
            if not eligible:
                approved_text = document.get("content", "").split("Фотографии и пояснения", 1)[0].strip()
                if not approved_text:
                    return missing_source_reply(body, current_index, unavailable=True)
                completion = "Инструкция проверена, применимых фотографий нет"
                updated = neutral_response(body, current_index, approved_text + "\n\nУдалось выполнить эти действия?")
                record_call(updated["messages"][current_index], READ_TOOL, {"file_id": file_id},
                            {"status": "ready", "source": document["source"]})
                return updated
            selection, offset = ",".join(map(str, eligible)), 0
            if advance:
                selection = flow.get("selection", previous[1].get("image_numbers", "auto"))
                offset = flow.get("next_offset")
                if type(offset) is not int or offset < 0:
                    return body
            elif clarification:
                fallback = [i["number"] for i in previous[2].get("images", []) if i.get("number") in eligible]
                selected = [i for i in clarification_images(document, user_text, fallback or eligible) if i in eligible]
                if not selected:
                    return body
                selection = ",".join(map(str, selected))
            if existing and existing[1].get("source", {}).get("file_id") == file_id:
                result = copy.deepcopy(existing[1])
                selection, offset = existing[0].get("image_numbers", "auto"), existing[0].get("offset", 0)
            else:
                existing = None
                result = await runtime.show(file_id, selection, offset)
            if not isinstance(result, dict):
                return body
            updated = copy.deepcopy(body)
            response = updated["messages"][current_index]
            if search_trace is not None:
                record_call(response, "query_knowledge_files", {"query": query, "count": 8}, search_trace)
            record_call(response, READ_TOOL, {"file_id": file_id}, {"status": "ready", "source": document["source"]})
            if result.get("status") != "displayed" or result.get("displayed") is not True:
                record_call(response, IMAGE_TOOL, {"file_id": file_id, "image_numbers": selection, "offset": offset}, result)
                replace_text(response, "Не удалось подтвердить показ фотографии из инструкции. "
                             + ("Проверьте, появилась ли она в ответе." if result.get("status") == "delivery_unknown"
                                else "Текст инструкции доступен, но изображение сейчас получить не удалось."))
                return updated
            source_images = {image["number"]: image["caption"] for image in document.get("images", [])}
            if (result.get("source", {}).get("file_id") != file_id
                    or result.get("source", {}).get("sha256") != document["source"].get("sha256")
                    or not result.get("images")
                    or any(source_images.get(image.get("number")) != image.get("caption")
                           for image in result.get("images", []))):
                return neutral_response(updated, current_index,
                                        "Не удалось подтвердить соответствие фотографий текущей инструкции. Продолжение приостановлено.")
            if clarification:
                next_flow = {"selection": flow.get("selection", previous[1].get("image_numbers", "auto")),
                             "has_more": flow.get("has_more", False), "next_offset": flow.get("next_offset")}
            else:
                next_flow = {"selection": selection, "has_more": result.get("has_more", False),
                             "next_offset": result.get("next_offset")}
            if existing and selection.strip().casefold() in {"auto", "all"}:
                # Exclude source-marked decoration/admin-only pictures from future automatic pages.
                shown = {image["number"] for image in result["images"]}
                next_position = max((i + 1 for i, number in enumerate(eligible) if number in shown), default=0)
                more = next_position < len(eligible)
                next_flow = {"selection": ",".join(map(str, eligible)), "has_more": more,
                             "next_offset": next_position if more else None}
            next_flow.update({"processed": True, "layout": "inline", "start_user_id": start_user_id})
            result["_ai_support_flow"] = next_flow
            if existing:
                annotate_flow(response, result["images"], next_flow)
            else:
                record_call(response, IMAGE_TOOL, {"file_id": file_id, "image_numbers": selection, "offset": offset}, result)
            replace_text(response, render_steps(document, result.get("images", []), next_flow["has_more"], clarification))
            if not await runtime.present_inline(response, result):
                next_flow["processed"] = False
                annotate_flow(response, result["images"], next_flow)
                completion = "Фотографии получены, оформление ответа не сохранено"
                return neutral_response(updated, current_index,
                                        "Фотографии получены, но сохранить пояснения рядом с ними не удалось. Попробуйте повторить сообщение.")
            completion = "Инструкция проверена, фотографии добавлены"
            return updated
        except Exception:
            # Never retain unsupported technical instructions after a source-check failure.
            log.warning("AI Support automatic images: source delivery could not be completed")
            if keep_current:
                return neutral_response(body, current_index,
                                        "Не удалось проверить текущую инструкцию. Я не могу предложить другую процедуру без подтверждённого источника.")
            completion = "Проверка базы знаний недоступна"
            return missing_source_reply(body, current_index, unavailable=True)
        finally:
            if status_started:
                await report_status(__event_emitter__, completion, True)
