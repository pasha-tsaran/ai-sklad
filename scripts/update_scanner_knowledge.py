"""Apply the owner's requested scanner-file changes to the configured collection.

Open WebUI 0.11 API contracts
https://github.com/open-webui/open-webui/blob/v0.11.0/backend/open_webui/routers/files.py
https://github.com/open-webui/open-webui/blob/v0.11.0/backend/open_webui/routers/knowledge.py
"""

import argparse
import json

import httpx

from app.core.config import get_settings
from app.services.rag.knowledge_files import list_knowledge_files
from app.services.rag.talk import TalkKnowledge

COLLECTION = "85cda3aa-b854-4c26-9948-12da742277ef"
WITHDRAWN = "33_Сканер штрихкодов в базе АДРЕС.docx"
RENAMES = {
    "ИТ_01_Настройка нового сканера штрихкодов в 1С.md": "01_Настройка нового сканера штрихкодов в 1С.md",
    "ИТ_37_Zebra DS2208 — лист настроечных штрихкодов.docx": "37_Zebra DS2208 — лист настроечных штрихкодов.docx",
    "ИТ_38_Zebra DS2208 — Caps Lock Override.docx": "38_Zebra DS2208 — Caps Lock Override.docx",
}


def current_files(client, base):
    response = client.get(base + "/api/v1/knowledge/" + COLLECTION)
    response.raise_for_status()
    metadata = response.json()
    if (metadata.get("id", COLLECTION) != COLLECTION
            or (metadata.get("meta") or {}).get("source") == "external"):
        raise ValueError("KnowledgeScopeMismatch")
    return list_knowledge_files(client, base, COLLECTION)


def update_scanner_knowledge(settings, *, apply=False, transport=None):
    if settings.openwebui_knowledge_ids != [COLLECTION]:
        raise ValueError("KnowledgeScopeMismatch")
    knowledge = TalkKnowledge(settings, transport=transport)
    with knowledge._client() as client:
        files = current_files(client, knowledge.base)
        existing_names = {item["filename"] for item in files}
        changes = []
        for item in files:
            filename = item["filename"]
            if filename == WITHDRAWN:
                changes.append({"action": "remove", "file_id": item["id"], "filename": filename})
            elif filename in RENAMES:
                new_name = RENAMES[filename]
                action = "remove_duplicate" if new_name in existing_names else "rename"
                changes.append({"action": action, "file_id": item["id"], "filename": filename, "new_name": new_name})
        if apply:
            for change in changes:
                current = {item["id"]: item for item in current_files(client, knowledge.base)}
                found = current.get(change["file_id"])
                if found is None:
                    continue
                if found["filename"] != change["filename"]:
                    raise ValueError("KnowledgeChangedDuringUpdate")
                if change["action"] == "rename":
                    if any(item["filename"] == change["new_name"] for item in current.values()):
                        raise ValueError("KnowledgeChangedDuringUpdate")
                    response = client.post(
                        knowledge.base + "/api/v1/files/" + change["file_id"] + "/rename",
                        json={"filename": change["new_name"]},
                    )
                else:
                    if change["action"] == "remove_duplicate" and not any(
                        item["filename"] == change["new_name"] for item in current.values()
                    ):
                        raise ValueError("ReplacementFileMissing")
                    response = client.post(
                        knowledge.base + "/api/v1/knowledge/" + COLLECTION + "/file/remove",
                        params={"delete_file": "false"}, json={"file_id": change["file_id"]},
                    )
                response.raise_for_status()
            files = current_files(client, knowledge.base)
        names = {item["filename"] for item in files}
        verified = (WITHDRAWN not in names and all(name not in names for name in RENAMES)
                    and all(name in names for name in RENAMES.values()))
        return {"status": "ok", "applied": apply, "verified": verified if apply else None,
                "knowledge_id": COLLECTION, "changes": changes,
                "scanner_files": [item["filename"] for item in files
                                  if "сканер" in item["filename"].casefold() or "ds2208" in item["filename"].casefold()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Update the three scanner filenames and remove instruction 33")
    parser.add_argument("--apply", action="store_true", help="Apply the requested changes; otherwise show the plan")
    args = parser.parse_args(argv)
    try:
        result = update_scanner_knowledge(get_settings(), apply=args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not args.apply or result["verified"] else 1
    except Exception as error:
        safe = {"status": "error", "error_code": type(error).__name__}
        if isinstance(error, httpx.HTTPStatusError):
            safe["http_status"] = error.response.status_code
        print(json.dumps(safe), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
