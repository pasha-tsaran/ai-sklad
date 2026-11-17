"""Persistent, source-grounded instruction discovery and verified parse caching."""

import copy
import hashlib
import io
import json
import math
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from zipfile import ZipFile

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.models import KnowledgeIndex, KnowledgeIndexImage, now
from app.services.rag.documents import Illustration
from app.services.rag.evidence import build_evidence_registry
from app.services.rag.interface import Candidate
from app.services.rag.query import _terms, normalized_query
from integrations.openwebui.approved_instruction_images import MAX_SOURCE_BYTES, W, WP, _member, _validate_archive, _xml

PARSER_VERSION = "source-catalog-v2"
_WARNING = re.compile(r"\b(?:внимание|осторожно|предупреждение|нельзя|запрещено|"
                      r"не\s+(?:удаляйте|сканируйте|выключайте|отключайте))\b", re.I)
_PREREQUISITE = re.compile(r"^(?:перед\s+(?:началом|настройкой|работой)|предварительн\w*|"
                           r"необходим\w*|требовани\w*|требуется|убедитесь|для\s+работы\s+понадобится)\b", re.I)
_CONDITION = re.compile(r"\b(?:если|когда|при\s+(?:первом|повторном)|для\s+(?:одного|двух)|"
                        r"перед\s+(?:началом|настройкой|работой))\b", re.I)
_EQUIPMENT = re.compile(r"\b(?:сканер\w*|принтер\w*|термопринтер\w*|терминал\w*|тсд|мфу|"
                        r"компьютер\w*|считывател\w*|маршрутизатор\w*)\b", re.I)
_APPLICATION = re.compile(r"\b(?:программ\w*|приложени\w*|сервис\w*|учетн\w*\s+запис\w*|"
                          r"учётн\w*\s+запис\w*)\b", re.I)
_SEQUENTIAL_HEADING = re.compile(r"^(?:(?:шаг|step|этап)\s*(?:№\s*)?\d+\b|\d+[.)](?:\s|$))", re.I)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _source_identity(source):
    return _digest({key: source[key] for key in ("file_id", "knowledge_id", "sha256")})


def read_docx_layout(original):
    """Retain original drawing positions and their exact body or alt-text anchors."""
    if not isinstance(original, bytes) or len(original) > MAX_SOURCE_BYTES:
        raise ValueError("InvalidLayoutSource")
    with ZipFile(io.BytesIO(original)) as archive:
        _validate_archive(archive)
        root = _xml(_member(archive, "word/document.xml", 5_000_000))
        body = root.find(W + "body")
        if body is None:
            raise ValueError("InvalidLayoutSource")
        numbers = {id(drawing): number for number, drawing in enumerate(root.iter(W + "drawing"), 1)}
        rows, seen = [], set()
        previous_body = None
        for ordinal, paragraph in enumerate(body.iter(W + "p"), 1):
            text = "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()
            body_row = None
            if text:
                body_row = {"text": text, "paragraph": ordinal, "image_numbers": []}
                rows.append(body_row)
            for drawing in paragraph.iter(W + "drawing"):
                if id(drawing) in seen:
                    continue
                seen.add(id(drawing))
                number = numbers[id(drawing)]
                properties = next(drawing.iter(WP + "docPr"), None)
                caption = properties.get("descr", "").strip() if properties is not None else ""
                if caption and (not text or " ".join(caption.split()) not in " ".join(text.split())):
                    rows.append({"text": caption, "paragraph": ordinal, "image_numbers": [number]})
                else:
                    anchor = body_row if body_row is not None else previous_body
                    if anchor is not None:
                        anchor["image_numbers"].append(number)
            if body_row is not None:
                previous_body = body_row
        return {"content": "\n\n".join(row["text"] for row in rows), "paragraphs": rows}


def read_docx_headings(original):
    """Read explicit Word outline structure without inferring headings from text."""
    if not isinstance(original, bytes) or len(original) > MAX_SOURCE_BYTES:
        raise ValueError("InvalidHeadingSource")

    def outline(properties):
        node = properties.find(W + "outlineLvl") if properties is not None else None
        if node is None:
            return False, None
        value = node.get(W + "val", "")
        return True, int(value) + 1 if value.isdigit() and 0 <= int(value) <= 8 else None

    with ZipFile(io.BytesIO(original)) as archive:
        _validate_archive(archive)
        root = _xml(_member(archive, "word/document.xml", 5_000_000))
        body = root.find(W + "body")
        if body is None:
            raise ValueError("InvalidHeadingSource")
        styles = {}
        if "word/styles.xml" in archive.namelist():
            style_root = _xml(_member(archive, "word/styles.xml", 5_000_000))
            for style in style_root.findall(W + "style"):
                name, parent = style.find(W + "name"), style.find(W + "basedOn")
                styles[style.get(W + "styleId")] = {
                    "outline": outline(style.find(W + "pPr")),
                    "name": name.get(W + "val", "") if name is not None else "",
                    "parent": parent.get(W + "val") if parent is not None else None,
                }

        def style_level(style_id):
            seen = set()
            for _ in range(32):
                if not style_id or style_id in seen:
                    return None
                seen.add(style_id)
                style = styles.get(style_id, {})
                specified, level = style.get("outline", (False, None))
                if specified:
                    return level
                for name in (style.get("name", ""), style_id):
                    match = re.fullmatch(r"(?:heading|заголовок)[ _-]*([1-9])", name, re.I)
                    if match:
                        return int(match.group(1))
                style_id = style.get("parent")
            return None

        occurrences, headings = Counter(), []
        for ordinal, paragraph in enumerate(body.iter(W + "p"), 1):
            text = "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()
            if not text:
                continue
            occurrences[text] += 1
            properties = paragraph.find(W + "pPr")
            specified, level = outline(properties)
            if not specified and properties is not None:
                style = properties.find(W + "pStyle")
                level = style_level(style.get(W + "val")) if style is not None else None
            if level is not None:
                headings.append({"text": text, "occurrence": occurrences[text], "level": level, "paragraph": ordinal})
        return headings


def instruction_sections(document):
    """Preserve original paragraphs with stable, version-scoped delivery IDs."""
    source, content = document["source"], document["content"]
    identity = _source_identity(source)
    captions = {image["number"]: " ".join(image["caption"].split()) for image in document.get("images", [])}
    sections = []
    heading = ""
    group_id = "procedure-" + identity[:24]
    explicit_headings = {(item["text"], item["occurrence"]): item["level"] for item in document.get("headings", [])
                         if isinstance(item, dict) and isinstance(item.get("text"), str)
                         and type(item.get("occurrence")) is int and type(item.get("level")) is int
                         and item["occurrence"] > 0 and 1 <= item["level"] <= 9}
    positioned_headings = {(item["text"], item["paragraph"]): item["level"] for item in document.get("headings", [])
                           if isinstance(item, dict) and isinstance(item.get("text"), str)
                           and type(item.get("paragraph")) is int and type(item.get("level")) is int
                           and item["paragraph"] > 0 and 1 <= item["level"] <= 9}
    occurrences = Counter()
    mandatory_kind = None
    layout = document.get("layout")
    positioned = (isinstance(layout, list) and all(isinstance(row, dict) and isinstance(row.get("text"), str)
                  and type(row.get("paragraph")) is int and isinstance(row.get("image_numbers"), list)
                  for row in layout) and "\n\n".join(row["text"] for row in layout) == content)
    paragraphs = layout if positioned else [{"text": text} for text in content.splitlines()]
    for paragraph in paragraphs:
        text = paragraph["text"].strip()
        if not text:
            continue
        ordinal = len(sections) + 1
        normalized = " ".join(text.split())
        occurrences[text] += 1
        markdown = re.match(r"^(#{1,6})\s+\S", text)
        level = (positioned_headings.get((text, paragraph["paragraph"])) if positioned else
                 explicit_headings.get((text, occurrences[text])))
        if level is None and markdown:
            level = len(markdown.group(1))
        if positioned:
            images = [number for number in paragraph["image_numbers"] if type(number) is int and number in captions]
        else:
            images = [number for number, caption in captions.items()
                      if caption and (caption in normalized or len(normalized) >= 20 and normalized in caption)]
        kind = "TEXT"
        label = text.lstrip("# ")
        if _WARNING.search(label):
            kind = "WARNING"
        elif _PREREQUISITE.search(label):
            kind = "PREREQUISITE"
        elif images:
            kind = "IMAGE"
        is_heading = level is not None or len(text) <= 100 and text.endswith(":")
        if is_heading:
            heading = text.lstrip("# ").rstrip(":")
            mandatory_kind = kind if kind in {"WARNING", "PREREQUISITE"} else None
        elif mandatory_kind and kind not in {"WARNING", "PREREQUISITE"}:
            kind = mandatory_kind
        suffix = _digest([identity, ordinal, text])[:24]
        if level is not None and level <= 2 and not _SEQUENTIAL_HEADING.match(label):
            group_id = "procedure-" + suffix
        sections.append({"id": "section-" + suffix, "title": heading or text[:100], "text": text,
                         "kind": kind, "image_numbers": images, "ordinal": ordinal,
                         "group_id": group_id,
                         "step_ids": ["step-" + suffix],
                         "image_ids": [f"image-{identity[:24]}-{number}" for number in images]})
    return sections


@dataclass(frozen=True)
class CatalogEntry:
    source: dict
    content: str
    card: dict
    sections: tuple[dict, ...]
    registry: dict
    remote_revision: str | None = None

    @property
    def evidence(self):
        return self.registry


def build_catalog_entry(approved):
    """Build discovery excerpts without generating instructions or conditions."""
    document = approved.document
    sections = instruction_sections(document)
    source = copy.deepcopy(document["source"])

    def excerpts(selected):
        return [{"section_id": section["id"], "text": section["text"][:400]} for section in selected[:4]]

    purpose_positions = {position for index, section in enumerate(sections)
                         if re.search(r"назначени|област\w*\s+применени", section["text"], re.I)
                         for position in (index, index + 1) if position < len(sections)}
    purpose = [section for index, section in enumerate(sections) if index in purpose_positions]
    card = {"source_id": "source-" + _source_identity(source)[:24], "title": approved.candidate.title,
            "purpose": excerpts(purpose or sections[:2]),
            "equipment": excerpts([section for section in sections if _EQUIPMENT.search(section["text"])]),
            "applications": excerpts([section for section in sections if _APPLICATION.search(section["text"])]),
            "conditions": excerpts([section for section in sections if _CONDITION.search(section["text"])]),
            "section_count": len(sections), "image_count": len(approved.illustrations), "version": PARSER_VERSION}
    captions = {number: image.caption for number, image in approved.illustrations.items()}
    return CatalogEntry(source, document["content"], card, tuple(sections),
                        build_evidence_registry(document["content"], captions))


def _tokens(value):
    words = []
    for word in normalized_query(value).split():
        words.extend([word] if word.isdigit() else sorted(_terms(word)))
    return words


def bm25_rank(query, entries, limit=None):
    """Rank the whole available catalogue with source-only lexical evidence."""
    query_terms = set(_tokens(query))
    if not query_terms or not entries:
        return []
    frequencies = {key: Counter(_tokens(entry.card["title"] + "\n" + entry.card["title"] + "\n" + entry.content))
                   for key, entry in entries.items()}
    lengths = {key: sum(terms.values()) for key, terms in frequencies.items()}
    average_length = sum(lengths.values()) / len(lengths) or 1
    document_frequency = Counter(term for terms in frequencies.values() for term in terms if term in query_terms)
    scores = {}
    for key, terms in frequencies.items():
        score = 0.0
        for term in query_terms & terms.keys():
            inverse_frequency = math.log(1 + (len(entries) - document_frequency[term] + 0.5)
                                         / (document_frequency[term] + 0.5))
            frequency = terms[term]
            denominator = frequency + 1.2 * (0.25 + 0.75 * lengths[key] / average_length)
            score += inverse_frequency * frequency * 2.2 / denominator
        if score:
            scores[key] = score
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    return ranked if limit is None else ranked[:limit]


def reciprocal_rank_fusion(*rankings, k=60, limit=None):
    """Merge independent lexical and semantic rankings without mixing scores."""
    if type(k) is not int or k < 1:
        raise ValueError("InvalidRankFusionConstant")
    scores, first_seen = {}, {}
    for ranking in rankings:
        for position, item in enumerate(dict.fromkeys(ranking), 1):
            first_seen.setdefault(item, len(first_seen))
            scores[item] = scores.get(item, 0.0) + 1 / (k + position)
    ranked = sorted(scores, key=lambda item: (-scores[item], first_seen[item], item))
    return ranked if limit is None else ranked[:limit]


def remote_revision(item):
    """A discovery freshness hint, never proof of the original byte hash."""
    fields = {key: item[key] for key in ("hash", "updated_at", "size", "etag")
              if isinstance(item.get(key), (str, int, float)) and not isinstance(item.get(key), bool)}
    return _digest(fields) if fields else None


class KnowledgeCatalog:
    def __init__(self, settings, factory):
        self.settings, self.factory = settings, factory
        self.scope_hash = _digest({"base": settings.openwebui_base_url.rstrip("/"),
                                   "collections": sorted(set(settings.openwebui_knowledge_ids))})

    @staticmethod
    def _matches(row, files):
        item = files.get(row.file_id)
        return bool(item and item.get("filename") == row.filename
                    and item.get("knowledge_id") == row.knowledge_id)

    def _entry(self, row):
        return CatalogEntry(copy.deepcopy(row.source), row.content, copy.deepcopy(row.card),
                            tuple(copy.deepcopy(row.sections)), copy.deepcopy(row.evidence), row.remote_revision)

    def reconcile(self, files):
        changed = 0
        with self.factory() as session, session.begin():
            rows = session.scalars(select(KnowledgeIndex).where(KnowledgeIndex.scope_hash == self.scope_hash)).all()
            for row in rows:
                if row.active and not self._matches(row, files):
                    row.active = False
                    changed += 1
        return changed

    def entries(self, files):
        with self.factory() as session:
            rows = session.scalars(select(KnowledgeIndex).where(
                KnowledgeIndex.scope_hash == self.scope_hash, KnowledgeIndex.active.is_(True),
                KnowledgeIndex.parser_version == PARSER_VERSION, KnowledgeIndex.error_code.is_(None),
            )).all()
            return {row.file_id: self._entry(row) for row in rows if self._matches(row, files)
                    and (remote_revision(files[row.file_id]) is None
                         or row.remote_revision == remote_revision(files[row.file_id]))}

    def revision(self, files):
        entries = self.entries(files)
        return _digest({"version": PARSER_VERSION, "scope": self.scope_hash,
                        "sources": {key: entry.source for key, entry in sorted(entries.items())}})

    def load_verified(self, file_id, original, files):
        """Only current original bytes can authorize reuse of a parsed snapshot."""
        if not isinstance(original, bytes):
            raise ValueError("OriginalBytesRequired")
        digest = hashlib.sha256(original).hexdigest()
        # One SQL snapshot prevents combining an old card with replacement photos.
        with self.factory() as session:
            rows = session.execute(select(KnowledgeIndex, KnowledgeIndexImage).outerjoin(
                KnowledgeIndexImage, KnowledgeIndexImage.index_id == KnowledgeIndex.id,
            ).where(KnowledgeIndex.scope_hash == self.scope_hash, KnowledgeIndex.file_id == file_id,
                    KnowledgeIndex.active.is_(True), KnowledgeIndex.parser_version == PARSER_VERSION,
                    KnowledgeIndex.error_code.is_(None),
                    KnowledgeIndex.source_sha256 == digest).order_by(KnowledgeIndexImage.number)).all()
            if not rows or not self._matches(rows[0][0], files):
                return None
            row = rows[0][0]
            expected = {"file_id": file_id, "filename": row.filename, "knowledge_id": row.knowledge_id, "sha256": digest}
            if row.source != expected:
                return None
            images = {}
            for _, image in rows:
                if image is not None:
                    if hashlib.sha256(image.data).hexdigest() != image.sha256:
                        return None
                    images[image.number] = Illustration(image.filename, image.caption, image.content_type, image.data)
            candidate = Candidate(content=row.content, illustrations=list(images.values()), **copy.deepcopy(row.candidate_meta))
            document = {**copy.deepcopy(row.document_meta), "content": row.content}
        from app.services.rag.talk import ApprovedTalkDocument

        return ApprovedTalkDocument(document, images, candidate)

    def store(self, approved, *, remote_revision=None):
        entry = build_catalog_entry(approved)
        source = entry.source
        if source["knowledge_id"] not in self.settings.openwebui_knowledge_ids:
            raise ValueError("KnowledgeScopeMismatch")
        candidate = approved.candidate
        candidate_meta = {key: copy.deepcopy(getattr(candidate, key)) for key in
                          ("solution_id", "title", "version", "chunk_id", "steps", "score", "source")}
        for attempt in range(2):
            try:
                with self.factory() as session, session.begin():
                    row = session.scalar(select(KnowledgeIndex).where(
                        KnowledgeIndex.scope_hash == self.scope_hash, KnowledgeIndex.file_id == source["file_id"],
                    ).with_for_update())
                    if row is None:
                        row = KnowledgeIndex(scope_hash=self.scope_hash, file_id=source["file_id"])
                        session.add(row)
                    same_source = row.source_sha256 == source["sha256"]
                    row.remote_revision = (remote_revision if remote_revision is not None
                                           else row.remote_revision if same_source else None)
                    row.knowledge_id, row.filename = source["knowledge_id"], source["filename"]
                    row.source_sha256, row.parser_version = source["sha256"], PARSER_VERSION
                    row.active, row.error_code, row.checked_at = True, None, now()
                    row.content, row.source = entry.content, copy.deepcopy(source)
                    row.document_meta = {key: copy.deepcopy(value) for key, value in approved.document.items() if key != "content"}
                    row.candidate_meta = candidate_meta
                    row.card, row.sections, row.evidence = entry.card, list(entry.sections), entry.registry
                    session.flush()
                    session.execute(delete(KnowledgeIndexImage).where(KnowledgeIndexImage.index_id == row.id))
                    for number, image in approved.illustrations.items():
                        session.add(KnowledgeIndexImage(index_id=row.id, number=number, filename=image.filename,
                                                       caption=image.caption, content_type=image.content_type,
                                                       sha256=hashlib.sha256(image.data).hexdigest(), data=image.data))
                return entry
            except IntegrityError:
                if attempt:
                    raise

    def _mark_error(self, file_id, code):
        with self.factory() as session, session.begin():
            row = session.scalar(select(KnowledgeIndex).where(
                KnowledgeIndex.scope_hash == self.scope_hash, KnowledgeIndex.file_id == file_id,
            ).with_for_update())
            if row is not None:
                row.error_code, row.checked_at = code, now()

    def sync(self, knowledge, client=None, files=None, *, refresh=False):
        """Index new or changed sources using the existing bounded original reader."""
        errors, reads = {}, 0
        with nullcontext(client) if client is not None else knowledge._client() as active_client:
            current = knowledge._files(active_client) if files is None else files
            self.reconcile(current)
            known = self.entries(current)
            for file_id, item in current.items():
                if not refresh and file_id in known:
                    continue
                try:
                    approved = knowledge._read(active_client, file_id, current)
                    self.store(approved, remote_revision=remote_revision(item))
                    reads += 1
                except Exception as error:
                    from app.services.rag.talk import KnowledgeUnavailable

                    code = error.code if isinstance(error, KnowledgeUnavailable) else type(error).__name__
                    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", code):
                        code = type(error).__name__
                    errors[file_id] = code
                    self._mark_error(file_id, code)
            final_files = knowledge._files(active_client)
            self.reconcile(final_files)
            entries = self.entries(final_files)
            errors = {key: code for key, code in errors.items() if key in final_files}
        return {"status": "ready" if len(entries) == len(final_files) and not errors else "partial",
                "source_count": len(final_files), "indexed_count": len(entries), "read_count": reads,
                "complete": len(entries) == len(final_files) and not errors, "errors": errors,
                "revision": self.revision(final_files)}
