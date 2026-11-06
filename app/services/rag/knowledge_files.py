"""Bounded file membership listing for the Open WebUI 0.11 paginated API.

The caller first checks the approved knowledge ID and its metadata/ACL. This helper
lists only that exact knowledge route; it never follows collection/search links.
"""

import json
from urllib.parse import quote
from uuid import UUID


PAGE_SIZE = 30  # v0.11 uses 30 for non-admins even when another limit is requested.
MAX_FILES = 10_000
MAX_PAGES = 334
MAX_PAGE_BYTES = 2_000_000


class KnowledgeFilesError(ValueError):
    """Fixed diagnostic code; never includes a response body or credentials."""


def _canonical_uuid(value):
    if not isinstance(value, str):
        raise KnowledgeFilesError("InvalidKnowledgeFileId")
    try:
        result = str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise KnowledgeFilesError("InvalidKnowledgeFileId") from None
    if result != value.lower():
        raise KnowledgeFilesError("InvalidKnowledgeFileId")
    return result


def list_knowledge_files(client, base, knowledge_id):
    """Return all exact-scope id/filename pairs or fail without a partial listing.

    HTTP failures remain HTTP failures so the caller can report a safe stage and
    status code. Metadata GET/allowlist checks stay with the caller.
    """
    knowledge_id = _canonical_uuid(knowledge_id)
    url = base.rstrip("/") + "/api/v1/knowledge/" + quote(knowledge_id, safe="") + "/files"
    expected_total, files, seen = None, [], set()
    for page in range(1, MAX_PAGES + 1):
        params = {"page": page, "limit": PAGE_SIZE, "order_by": "name", "direction": "asc"}
        content = bytearray()
        with client.stream("GET", url, params=params, follow_redirects=False) as response:
            response.raise_for_status()
            for block in response.iter_bytes(chunk_size=65_536):
                content.extend(block)
                if len(content) > MAX_PAGE_BYTES:
                    raise KnowledgeFilesError("KnowledgeFilesPageTooLarge")
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, ValueError):
            raise KnowledgeFilesError("InvalidKnowledgeFilesResponse") from None
        if not isinstance(payload, dict):
            raise KnowledgeFilesError("InvalidKnowledgeFilesResponse")
        if payload.get("knowledge_id", knowledge_id) != knowledge_id:
            raise KnowledgeFilesError("KnowledgeFilesScopeMismatch")
        items, total = payload.get("items"), payload.get("total")
        if not isinstance(items, list) or type(total) is not int or total < 0:
            raise KnowledgeFilesError("InvalidKnowledgeFilesResponse")
        if total > MAX_FILES:
            raise KnowledgeFilesError("KnowledgeFilesLimitExceeded")
        if expected_total is not None and total != expected_total:
            raise KnowledgeFilesError("KnowledgeFilesChangedDuringPagination")
        expected_total = total
        if len(items) != min(PAGE_SIZE, total - len(files)):
            raise KnowledgeFilesError("IncompleteKnowledgeFilesPage")
        for item in items:
            if not isinstance(item, dict):
                raise KnowledgeFilesError("InvalidKnowledgeFilesResponse")
            if item.get("knowledge_id", knowledge_id) != knowledge_id:
                raise KnowledgeFilesError("KnowledgeFilesScopeMismatch")
            file_id = _canonical_uuid(item.get("id"))
            filename = item.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                raise KnowledgeFilesError("InvalidKnowledgeFilesResponse")
            if file_id in seen:
                raise KnowledgeFilesError("KnowledgeFilesChangedDuringPagination")
            seen.add(file_id)
            files.append({"id": file_id, "filename": filename})
        if len(files) == total:
            return files
    raise KnowledgeFilesError("KnowledgeFilesPaginationLimitExceeded")
