"""Open WebUI owns retrieval and embeddings; DB rows are audit snapshots only."""

import hashlib
from uuid import NAMESPACE_URL, uuid5
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict

from app.models import KnowledgeChunk, KnowledgeDocument, Solution
from app.services.rag.interface import Candidate
from app.services.rag.documents import read_docx, read_xlsx
from app.services.llm.prompts import KNOWLEDGE_SELECTION_TASK, stage_messages


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: str | None


class OpenWebUIKnowledge:
    def __init__(self, settings, transport=None):
        self.settings = settings
        self.transport = transport
        self.base = settings.openwebui_base_url.rstrip("/")

    def identity(self, file_id):
        return str(uuid5(NAMESPACE_URL, self.base + "/files/" + file_id))

    def _files(self, client):
        files = {}
        for collection in self.settings.openwebui_knowledge_ids:
            response = client.get(self.base + "/api/v1/knowledge/" + quote(collection, safe=""))
            response.raise_for_status()
            # Fail closed on an incompatible server response, never search all knowledge.
            for item in response.json()["files"]:
                if str(item.get("filename", "")).startswith(("ИТ_", "СПРАВОЧНИК_")):
                    continue  # Reference/admin uploads are searchable in WebUI, not automatic employee solutions.
                files[item["id"]] = item
        return files

    def is_current(self, source):
        """Revalidate a historical snapshot without asking a model to generate an answer."""
        if source.get("base_url") != self.base or not self.settings.openwebui_knowledge_ids:
            return False
        file_id = source.get("file_id")
        if not isinstance(file_id, str):
            return False
        with httpx.Client(
            headers={"Authorization": "Bearer " + self.settings.openwebui_api_key.get_secret_value()},
            timeout=self.settings.llm_timeout_seconds, transport=self.transport, follow_redirects=False,
        ) as client:
            if file_id not in self._files(client):
                return False
            path = self.base + "/api/v1/files/" + quote(file_id, safe="")
            if source.get("original_sha256"):
                digest, size = hashlib.sha256(), 0
                with client.stream("GET", path + "/content") as raw:
                    raw.raise_for_status()
                    for block in raw.iter_bytes():
                        size += len(block)
                        if size > self.settings.knowledge_max_file_bytes:
                            return False
                        digest.update(block)
                matches = digest.hexdigest() == source["original_sha256"]
            else:
                response = client.get(path + "/data/content")
                response.raise_for_status()
                content = response.json()["content"]
                matches = isinstance(content, str) and hashlib.sha256(content.encode()).hexdigest() == source["sha256"]
            return matches and file_id in self._files(client)

    def search(self, query, excluded, context=None):
        if not self.settings.openwebui_knowledge_ids:
            return []
        with httpx.Client(
            headers={"Authorization": "Bearer " + self.settings.openwebui_api_key.get_secret_value()},
            timeout=self.settings.llm_timeout_seconds,
            transport=self.transport,
            follow_redirects=False,
        ) as client:
            files = self._files(client)
            allowed = {key: value for key, value in files.items() if self.identity(key) not in excluded}
            if not allowed:
                return []
            response = client.post(
                self.base + "/api/chat/completions",
                json={
                    "model": self.settings.openwebui_model,
                    "stream": False,
                    "temperature": 0,
                    "tool_ids": [],
                    "features": {"web_search": False},
                    "response_format": {"type": "json_object"},
                    "files": [{"type": "collection", "id": k} for k in self.settings.openwebui_knowledge_ids],
                    "messages": stage_messages(Selection, KNOWLEDGE_SELECTION_TASK, {
                        "query": query, "context": context or {},
                        "allowed_file_ids": list(allowed),
                    }),
                },
            )
            response.raise_for_status()
            selected = Selection.model_validate_json(response.json()["choices"][0]["message"]["content"])
            if selected.file_id is None:
                return []
            if selected.file_id not in allowed:
                raise ValueError("Unapproved knowledge reference")
            file_id = selected.file_id
            # Recheck membership after retrieval, then fetch the complete source, not generated prose.
            if file_id not in self._files(client):
                return []
            response = client.get(self.base + "/api/v1/files/" + quote(file_id, safe="") + "/data/content")
            response.raise_for_status()
            content = response.json()["content"]
            title = str(allowed[file_id].get("filename") or file_id)[:255]
            illustrations, original_hash = [], None
            if title.lower().endswith((".docx", ".xlsx")):
                original = bytearray()
                with client.stream("GET", self.base + "/api/v1/files/" + quote(file_id, safe="") + "/content") as raw:
                    raw.raise_for_status()
                    for block in raw.iter_bytes():
                        original.extend(block)
                        if len(original) > self.settings.knowledge_max_file_bytes:
                            raise ValueError("Knowledge upload too large")
                original_hash = hashlib.sha256(original).hexdigest()
                if title.lower().endswith(".docx"):
                    content, illustrations = read_docx(bytes(original))
                else:
                    content = read_xlsx(bytes(original))
                # Bind original bytes and images to a still-approved source.
                if file_id not in self._files(client):
                    return []
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty knowledge source")
            if len(content) > self.settings.knowledge_max_chars:
                raise ValueError("Source too large; split into complete instructions in Open WebUI")
            identity = self.identity(file_id)
            digest = hashlib.sha256(content.encode()).hexdigest()
            return [
                Candidate(
                    identity,
                    title,
                    1,
                    str(uuid5(NAMESPACE_URL, identity + digest)),
                    content,
                    [content],
                    1.0,
                    {
                        "backend": "openwebui",
                        "file_id": file_id,
                        "collection_ids": self.settings.openwebui_knowledge_ids,
                        "sha256": digest,
                        "base_url": self.base,
                        "original_sha256": original_hash,
                        "illustrations": [p.metadata() for p in illustrations],
                    },
                    illustrations,
                )
            ]


def save_snapshot(session, candidate):
    """No local vector search. Keep immutable source text for history and foreign keys."""
    document = session.get(KnowledgeDocument, candidate.solution_id)
    if document is None:
        document = KnowledgeDocument(id=candidate.solution_id, title=candidate.title, approved=True)
        session.add(document)
        session.flush()
    if session.get(KnowledgeChunk, candidate.chunk_id) is None:
        session.add(
            KnowledgeChunk(
                id=candidate.chunk_id,
                document_id=document.id,
                content=candidate.content,
                search_text="",
                embedding=[0.0] * 128,
                embedding_model="openwebui-reference",
            )
        )
        session.flush()
    solution = session.get(Solution, candidate.solution_id)
    if solution is None:
        solution = Solution(id=candidate.solution_id, problem_type="openwebui")
        session.add(solution)
    solution.chunk_id = candidate.chunk_id
    solution.title = candidate.title
    solution.steps = candidate.steps
    solution.version = candidate.version
    solution.active = True
    session.flush()
