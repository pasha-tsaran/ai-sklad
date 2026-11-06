"""Scoped native retrieval and original instructions for the Talk account bot."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import quote
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

from app.core.logging import event
from app.services.rag.documents import Illustration, read_xlsx
from app.services.rag.interface import Candidate
from app.services.rag.knowledge_files import KnowledgeFilesError, list_knowledge_files
from integrations.openwebui.approved_instruction_images import MAX_SOURCE_BYTES, MAX_TEXT_CHARS, read_docx_images, read_docx_text
from integrations.openwebui.automatic_instruction_images import choose_search_source


class KnowledgeUnavailable(ValueError):
    """A source or retrieval service could not be checked safely."""

    def __init__(self, code, stage=None, http_status=None):
        self.code = code if isinstance(code, str) and code.isascii() and code.replace("_", "").isalnum() else "KnowledgeError"
        self.stage, self.http_status = stage, http_status
        super().__init__(self.code)


@contextmanager
def knowledge_stage(stage):
    try:
        yield
    except KnowledgeUnavailable as error:
        error.stage = error.stage or stage
        raise
    except httpx.HTTPStatusError as error:
        raise KnowledgeUnavailable("HTTPError", stage, error.response.status_code) from None
    except httpx.TimeoutException:
        raise KnowledgeUnavailable("RequestTimedOut", stage) from None
    except httpx.RequestError:
        raise KnowledgeUnavailable("RequestFailed", stage) from None
    except KnowledgeFilesError as error:
        raise KnowledgeUnavailable(str(error), stage) from None
    except Exception as error:
        raise KnowledgeUnavailable(type(error).__name__, stage) from None


def report_failure(error):
    try:
        event("talk.knowledge_error", stage=error.stage, error_code=error.code, http_status=error.http_status)
    except Exception:
        pass  # Observability must never expose source contents or change the reply.


@dataclass(frozen=True)
class ApprovedTalkDocument:
    document: dict
    illustrations: dict[int, Illustration]
    candidate: Candidate


def _uuid(value):
    try:
        return isinstance(value, str) and str(UUID(value)) == value.lower()
    except (ValueError, TypeError, AttributeError):
        return False


def _employee_file(item):
    name = item.get("filename", "")
    return (_uuid(item.get("id")) and isinstance(name, str)
            and not name.casefold().startswith(("ит_", "справочник_"))
            and name.casefold().endswith((".docx", ".xlsx", ".md", ".txt", ".pdf")))


class TalkKnowledge:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport
        self.base = settings.openwebui_base_url.rstrip("/")

    def _client(self):
        return httpx.Client(
            headers={"Authorization": "Bearer " + self.settings.openwebui_api_key.get_secret_value()},
            timeout=self.settings.llm_timeout_seconds, transport=self.transport, follow_redirects=False,
        )

    def _files(self, client):
        ids = self.settings.openwebui_knowledge_ids
        if not ids or len(ids) > 20 or any(not _uuid(value) for value in ids):
            raise KnowledgeUnavailable("KnowledgeScopeNotConfigured")
        files = {}
        for collection in ids:
            with knowledge_stage("knowledge_metadata"):
                response = client.get(self.base + "/api/v1/knowledge/" + quote(collection, safe=""))
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict) or body.get("id", collection) != collection:
                    raise KnowledgeUnavailable("InvalidKnowledgeResponse")
                if (body.get("meta") or {}).get("source") == "external":
                    raise KnowledgeUnavailable("ExternalKnowledgeNotApproved")
                items = body.get("files")
                if items is not None and not isinstance(items, list):
                    raise KnowledgeUnavailable("InvalidKnowledgeResponse")
            if items is None or any(isinstance(item, dict) and "filename" not in item for item in items):
                # Open WebUI 0.11 GET metadata returns files:null. Fetch its
                # scoped paginated file API, including files in subdirectories.
                with knowledge_stage("knowledge_files"):
                    items = list_knowledge_files(client, self.base, collection)
            for item in items:
                if isinstance(item, dict) and _employee_file(item):
                    files[item["id"]] = {**item, "knowledge_id": collection}
        return files

    def _original(self, client, file_id, suffix="content"):
        data = bytearray()
        with knowledge_stage("extracted_text" if suffix == "data/content" else "original_file"):
            with client.stream("GET", self.base + "/api/v1/files/" + file_id + "/" + suffix) as response:
                response.raise_for_status()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > min(self.settings.knowledge_max_file_bytes, MAX_SOURCE_BYTES):
                        raise KnowledgeUnavailable("KnowledgeFileTooLarge")
        return bytes(data)

    def _read(self, client, file_id, files):
        item = files.get(file_id)
        if item is None:
            raise KnowledgeUnavailable("SourceNotInApprovedKnowledge")
        filename = item["filename"]
        extracted_pdf = filename.casefold().endswith(".pdf")
        if extracted_pdf:
            content = json.loads(self._original(client, file_id, "data/content")).get("content")
            if not isinstance(content, str):
                raise KnowledgeUnavailable("ExtractedTextMissing")
            original = content.encode()
        else:
            original = self._original(client, file_id)
        pictures = []
        if extracted_pdf:
            pass
        elif filename.casefold().endswith(".docx"):
            with knowledge_stage("parse_document"):
                content, pictures = read_docx_text(original), read_docx_images(original)
        elif filename.casefold().endswith(".xlsx"):
            content = read_xlsx(original)
        else:
            content = original.decode("utf-8-sig")
        if not content.strip() or len(content) > min(self.settings.knowledge_max_chars, MAX_TEXT_CHARS):
            raise KnowledgeUnavailable("KnowledgeTextInvalidOrTooLarge")
        current = self._files(client).get(file_id)
        if not current or (current["filename"], current["knowledge_id"]) != (filename, item["knowledge_id"]):
            raise KnowledgeUnavailable("SourceMembershipChanged")
        digest = hashlib.sha256(original).hexdigest()
        illustrations = {
            picture.number: Illustration(picture.filename, picture.caption, "image/png", picture.data)
            for picture in pictures
        }
        source = {"file_id": file_id, "filename": filename, "knowledge_id": item["knowledge_id"], "sha256": digest}
        identity = str(uuid5(NAMESPACE_URL, self.base + "/files/" + file_id))
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        candidate = Candidate(
            identity, filename, 1, str(uuid5(NAMESPACE_URL, identity + content_hash)), content, [content], 1.0,
            {"backend": "openwebui", "base_url": self.base, "file_id": file_id,
             "knowledge_id": item["knowledge_id"],
             "collection_ids": list(self.settings.openwebui_knowledge_ids), "sha256": content_hash,
             "original_sha256": None if extracted_pdf else digest,
             "illustrations": [p.metadata() for p in illustrations.values()]},
            list(illustrations.values()),
        )
        return ApprovedTalkDocument(
            {"status": "ready", "source": source, "content": content,
             "images": [{"number": picture.number, "caption": picture.caption} for picture in pictures]},
            illustrations, candidate,
        )

    def read(self, file_id):
        if not _uuid(file_id):
            raise KnowledgeUnavailable("InvalidSourceId")
        try:
            with self._client() as client:
                return self._read(client, file_id, self._files(client))
        except KnowledgeUnavailable as error:
            report_failure(error)
            raise
        except Exception as error:
            failure = KnowledgeUnavailable(type(error).__name__, "read")
            report_failure(failure)
            raise failure from None

    def search(self, query, excluded=(), fallback_query=None):
        """Use the native scoped search, never an autonomous model/tool selection."""
        try:
            with self._client() as client:
                files = {key: item for key, item in self._files(client).items() if key not in excluded}
                if not files:
                    return None
                queries = list(dict.fromkeys(q[:6000] for q in (query, fallback_query) if isinstance(q, str) and q.strip()))
                problem = fallback_query if isinstance(fallback_query, str) and fallback_query.strip() else query
                for text in queries[:2]:
                    with knowledge_stage("retrieval"):
                        response = client.post(self.base + "/api/v1/retrieval/query/collection", json={
                            "collection_names": list(self.settings.openwebui_knowledge_ids), "query": text, "k": 8,
                            "hybrid": False,
                        })
                        response.raise_for_status()
                        body = response.json()
                    documents, metadata = body.get("documents"), body.get("metadatas")
                    if documents == [] and metadata == []:
                        continue
                    if (not isinstance(documents, list) or not documents or not isinstance(documents[0], list)
                            or not isinstance(metadata, list) or not metadata or not isinstance(metadata[0], list)
                            or len(documents[0]) != len(metadata[0])):
                        raise KnowledgeUnavailable("InvalidRetrievalResponse")
                    chunks = []
                    for content, meta in zip(documents[0][:32], metadata[0][:32]):
                        file_id = meta.get("file_id") if isinstance(meta, dict) else None
                        if file_id in files and isinstance(content, str):
                            chunks.append({"file_id": file_id, "source": files[file_id]["filename"], "content": content[:50_000]})
                    selected = choose_search_source(problem, chunks)
                    if selected:
                        return self._read(client, selected, files)
                return None
        except KnowledgeUnavailable as error:
            report_failure(error)
            raise
        except Exception as error:
            failure = KnowledgeUnavailable(type(error).__name__, "search")
            report_failure(failure)
            raise failure from None

    def is_current(self, source):
        if not isinstance(source, dict):
            return False
        try:
            current = self.read(source.get("file_id"))
            return current.document["source"] == source
        except KnowledgeUnavailable:
            return False

    def check(self, query="Настройка сканера штрихкодов в 1С АДРЕС 1.0.53"):
        """Read-only end-to-end check of scoped membership, search and original photos."""
        try:
            with self._client() as client:
                count = len(self._files(client))
            approved = self.search(query)
            return {"status": "ok", "read_only": True,
                    "knowledge_collections": len(self.settings.openwebui_knowledge_ids), "approved_files": count,
                    "instruction_found": approved is not None,
                    "illustrations": len(approved.illustrations) if approved else 0}
        except KnowledgeUnavailable as error:
            return {"status": "error", "read_only": True, "stage": error.stage,
                    "error_code": error.code, "http_status": error.http_status}
