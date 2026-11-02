"""Exercise the importable Workspace Tool without an Open WebUI installation."""

import asyncio
import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image, PngImagePlugin


TOOL_PATH = Path(__file__).resolve().parents[1] / "integrations/openwebui/approved_instruction_images.py"
SOURCE_ID = "00000000-0000-4000-8000-000000000001"
KNOWLEDGE_ID = "85cda3aa-b854-4c26-9948-12da742277ef"
CHAT_ID = "00000000-0000-4000-8000-000000000002"


@pytest.fixture
def image_tool():
    name = "approved_instruction_images_under_test"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def png(color="red", metadata=False):
    output = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if metadata:
        info.add_text("Comment", "private original image metadata")
    Image.new("RGB", (20, 12), color).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def office(parts):
    result = io.BytesIO()
    with ZipFile(result, "w", ZIP_DEFLATED) as archive:
        for path, content in parts.items():
            archive.writestr(path, content)
    return result.getvalue()


def instruction(captions=("Нажмите «Дополнительно» вверху окна.",), *, external=False,
                target=None, image_data=None, extra_paragraph=""):
    """A minimal real DrawingML DOCX with pictures placed in instruction order."""
    paragraphs = ['<w:p><w:r><w:t>Утверждённая инструкция Acrobat</w:t></w:r></w:p>']
    relations = []
    parts = {}
    for number, caption in enumerate(captions, 1):
        description = escape(caption, {'"': "&quot;"})
        paragraphs.append(
            '<w:p><w:r><w:drawing><wp:inline>'
            f'<wp:docPr id="{number}" name="Фото {number}" descr="{description}"/>'
            f'<a:graphic><a:graphicData><a:blip r:embed="image{number}"/>'
            '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
        )
        relationship_target = target or ("https://untrusted.invalid/photo.png" if external
                                         else f"media/photo{number}.png")
        relations.append(
            f'<Relationship Id="image{number}" Target="{escape(relationship_target)}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
            f'TargetMode="{"External" if external else "Internal"}"/>'
        )
        parts[f"word/media/photo{number}.png"] = image_data or png("red" if number == 1 else "blue")
    paragraphs.append(extra_paragraph)
    parts["word/document.xml"] = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        '<w:body>' + "".join(paragraphs) + '</w:body></w:document>'
    )
    parts["word/_rels/document.xml.rels"] = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(relations) + '</Relationships>'
    )
    return office(parts)


def run(awaitable):
    return asyncio.run(awaitable)


def test_original_pixels_and_descriptions_preserved_in_order(image_tool):
    descriptions = ("Снимите галочку рядом с размером страницы.", "Нажмите кнопку «ОК».")
    pictures = image_tool.read_docx_images(instruction(descriptions))
    assert [picture.number for picture in pictures] == [1, 2]
    assert [picture.caption for picture in pictures] == list(descriptions)
    for picture, color in zip(pictures, ((255, 0, 0), (0, 0, 255)), strict=True):
        with Image.open(io.BytesIO(picture.data)) as decoded:
            assert decoded.format == "PNG"
            assert decoded.getpixel((0, 0)) == color


def test_image_metadata_removed_without_replacing_picture(image_tool):
    picture = image_tool.read_docx_images(instruction(image_data=png(metadata=True)))[0]
    with Image.open(io.BytesIO(picture.data)) as decoded:
        assert decoded.info == {}
        assert decoded.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("data", [
    b"not an office archive",
    instruction(("",)),
    instruction(external=True),
    instruction(target="../../outside.png"),
    instruction(image_data=b"not an image"),
])
def test_invalid_document_or_unapproved_image_cannot_be_shown(image_tool, data):
    with pytest.raises(ValueError):
        image_tool.read_docx_images(data)


def test_document_without_photographs_is_not_replaced_with_generated_content(image_tool):
    assert image_tool.read_docx_images(instruction(())) == []


def test_legacy_vml_image_is_not_silently_omitted(image_tool):
    source = instruction((), extra_paragraph='<w:p><w:r><w:pict/></w:r></w:p>')
    with pytest.raises(ValueError):
        image_tool.read_docx_images(source)


def test_docx_entities_rejected(image_tool):
    malicious = office({
        "word/document.xml": '<!DOCTYPE x [<!ENTITY value "unsafe">]><x>&value;</x>',
        "word/_rels/document.xml.rels": '<Relationships/>',
    })
    with pytest.raises(ValueError):
        image_tool.read_docx_images(malicious)


def test_archive_entry_count_bounded_before_parsing(image_tool):
    excessive = office({f"unneeded/{number}": b"" for number in range(2001)})
    with pytest.raises(ValueError):
        image_tool.read_docx_images(excessive)


def test_caption_size_is_bounded(image_tool):
    with pytest.raises(ValueError):
        image_tool.read_docx_images(instruction(("x" * 6001,)))


def test_instruction_picture_count_is_bounded(image_tool):
    with pytest.raises(ValueError):
        image_tool.read_docx_images(instruction(tuple("Кнопка." for _ in range(61))))


def test_total_normalized_picture_bytes_are_bounded(image_tool, monkeypatch):
    monkeypatch.setattr(image_tool, "MAX_TOTAL_IMAGE_BYTES", 1)
    with pytest.raises(ValueError):
        image_tool.read_docx_images(instruction())


def test_selector_uses_real_document_numbers(image_tool):
    assert image_tool.parse_image_numbers("all", 3) == [1, 2, 3]
    assert image_tool.parse_image_numbers("2", 3) == [2]


@pytest.mark.parametrize("selector", ["0", "-1", "4", "1,4", "one", "1-3", "", "https://example.com"])
def test_invalid_image_selection_is_rejected(image_tool, selector):
    with pytest.raises(ValueError):
        image_tool.parse_image_numbers(selector, 3)


def test_missing_request_context_cannot_report_display_success(image_tool):
    result = json.loads(run(image_tool.Tools().show_instruction_images("source-file")))
    assert not result.get("displayed", False)
    assert result.get("error") or result.get("status") != "ok"


class FakeRuntime:
    """Only replace the actual WebUI boundary, leaving delivery orchestration real."""

    def __init__(self, module, document=None):
        self.module = module
        self.document = document if document is not None else instruction()
        self.authorizations = []
        self.uploads = []
        self.discarded = []
        self.fail_upload_number = None
        self.revoke_on_recheck = False
        self.replace_on_recheck = False
        self.saved_files = []
        self.history_checks = []
        self.file = SimpleNamespace(id=SOURCE_ID, filename="05_Acrobat.docx", path="/storage/source.docx",
                                    meta={"content_type": "application/vnd.openxmlformats-officedocument."
                                                          "wordprocessingml.document"},
                                    updated_at=1, hash="source-hash")
        self.user = SimpleNamespace(id="employee", role="user")

    async def source(self, file_id, user_id, allowed_ids):
        self.authorizations.append((file_id, user_id, allowed_ids))
        if len(self.authorizations) > 1:
            if self.revoke_on_recheck:
                raise PermissionError("Source access revoked")
            if self.replace_on_recheck:
                changed = SimpleNamespace(**vars(self.file))
                changed.updated_at = 2
                changed.hash = "replacement-hash"
                return self.module.Source(self.user, changed, KNOWLEDGE_ID)
        return self.module.Source(self.user, self.file, KNOWLEDGE_ID)

    async def read_original(self, source):
        assert source.file.id == SOURCE_ID
        return self.document

    async def upload_image(self, request, source, picture, original_sha256):
        assert request is not None
        assert source.user.id == "employee"
        assert len(original_sha256) == 64
        if picture.number == self.fail_upload_number:
            raise OSError("Object storage unavailable")
        upload = SimpleNamespace(id=str(UUID(int=100 + picture.number)), filename=f"photo-{picture.number}.png",
                                 meta={"content_type": "image/png"})
        self.uploads.append((picture, upload))
        return upload

    async def discard(self, uploaded):
        self.discarded.append(uploaded.id)


    async def current_files(self, metadata, user_id):
        self.history_checks.append((metadata, user_id))
        return copy.deepcopy(self.saved_files)


def display(image_tool, monkeypatch, runtime, *, selection="all", emitter=None, file_id=SOURCE_ID,
            metadata=None):
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    events = []

    async def record(event):
        events.append(event)

    result = json.loads(run(image_tool.Tools().show_instruction_images(
        file_id, selection, __user__={"id": "employee"}, __request__=SimpleNamespace(),
        __event_emitter__=emitter or record, __metadata__=metadata,
    )))
    return result, events


def test_selected_original_images_delivered_as_one_real_native_event(image_tool, monkeypatch):
    captions = ("Первая кнопка.", "Вторая кнопка.", "Третья кнопка.")
    runtime = FakeRuntime(image_tool, instruction(captions))
    result, events = display(image_tool, monkeypatch, runtime, selection="1,3")
    assert result["status"] == "displayed" and result["displayed"] is True
    assert len(runtime.authorizations) == 2
    assert all(call[:2] == (SOURCE_ID, "employee") for call in runtime.authorizations)
    assert all(KNOWLEDGE_ID in call[2] for call in runtime.authorizations)
    assert [photo["number"] for photo in result["images"]] == [1, 3]
    assert [photo["caption"] for photo in result["images"]] == [captions[0], captions[2]]
    attachment_events = [event for event in events if event["type"] == "files"]
    assert len(attachment_events) == 1
    files = attachment_events[0]["data"]["files"]
    assert len(files) == 2
    assert all(file["type"] == "image" for file in files)
    assert [file["url"] for file in files] == [
        f"/api/v1/files/{upload.id}/content" for _, upload in runtime.uploads
    ]
    assert all("token" not in file["url"].lower() for file in files)
    assert runtime.discarded == []


def test_consecutive_calls_preserve_existing_images_without_duplicate_database_writes(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(("Первая кнопка.", "Вторая кнопка.")))
    metadata = {"chat_id": CHAT_ID, "message_id": "assistant-reply"}
    events, database_writes, live_files = [], [], []

    async def webui_v011_emitter(event):
        nonlocal live_files
        # v0.11.0 mutates the original event's list while saving new + previous files.
        events.append(copy.deepcopy(event))
        event_files = event["data"]["files"]
        if event["type"] == "files":
            event_files.extend(runtime.saved_files)
            runtime.saved_files = copy.deepcopy(event_files)
            database_writes.append(copy.deepcopy(event_files))
        if event["type"] in {"files", "chat:message:files"}:
            live_files = copy.deepcopy(event_files)

    first, _ = display(image_tool, monkeypatch, runtime, selection="1", emitter=webui_v011_emitter,
                       metadata=metadata)
    second, _ = display(image_tool, monkeypatch, runtime, selection="2", emitter=webui_v011_emitter,
                        metadata=metadata)
    assert first["displayed"] and second["displayed"]
    assert [event["type"] for event in events] == ["files", "files", "chat:message:files"]
    # Only new images go through the DB-appending event; cumulative files update UI only.
    assert [len(event["data"]["files"]) for event in events] == [1, 1, 2]
    assert [len(files) for files in database_writes] == [1, 2]
    expected_urls = {f"/api/v1/files/{upload.id}/content" for _, upload in runtime.uploads}
    assert len(live_files) == 2 and {file["url"] for file in live_files} == expected_urls
    assert len(runtime.saved_files) == 2
    assert {file["url"] for file in runtime.saved_files} == expected_urls
    assert runtime.history_checks == [(metadata, "employee"), (metadata, "employee")]


def test_existing_attachment_is_preserved_in_ui_resync(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool)
    previous = {"type": "file", "url": "/api/v1/files/previous/content", "name": "earlier.pdf"}
    runtime.saved_files = [previous]
    events = []

    async def emitter(event):
        events.append(copy.deepcopy(event))
        if event["type"] == "files":
            event["data"]["files"].extend(runtime.saved_files)
            runtime.saved_files = copy.deepcopy(event["data"]["files"])

    result, _ = display(image_tool, monkeypatch, runtime, emitter=emitter,
                        metadata={"chat_id": CHAT_ID, "message_id": "reply"})
    assert result["displayed"] is True
    assert [event["type"] for event in events] == ["files", "chat:message:files"]
    assert previous in events[-1]["data"]["files"]
    assert len(events[0]["data"]["files"]) == 1


def test_large_instruction_requires_photo_selection_instead_of_flooding_chat(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple(f"Кнопка {number}." for number in range(7))))
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["displayed"] is False and result["status"] == "selection_required"
    assert runtime.uploads == [] and not any(event["type"] == "files" for event in events)


def test_selection_caption_list_is_also_reauthorized_before_disclosure(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple(f"Кнопка {number}." for number in range(7))))
    runtime.revoke_on_recheck = True
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["displayed"] is False and result["status"] == "error"
    assert "images" not in result
    assert runtime.uploads == [] and not any(event["type"] == "files" for event in events)


def test_no_photos_does_not_claim_they_were_displayed(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(()))
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["displayed"] is False
    assert runtime.uploads == [] and not any(event["type"] == "files" for event in events)


@pytest.mark.parametrize("change", ["revoke_on_recheck", "replace_on_recheck"])
def test_source_revocation_or_replacement_before_delivery_cleans_private_uploads(image_tool, monkeypatch, change):
    runtime = FakeRuntime(image_tool)
    setattr(runtime, change, True)
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["displayed"] is False
    assert not any(event["type"] == "files" for event in events)
    assert runtime.discarded == [upload.id for _, upload in runtime.uploads]


def test_partial_upload_failure_cannot_produce_fake_success(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(("Первая кнопка.", "Вторая кнопка.")))
    runtime.fail_upload_number = 2
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["displayed"] is False
    assert not any(event["type"] == "files" for event in events)
    assert runtime.discarded == [upload.id for _, upload in runtime.uploads]


def test_event_delivery_uncertain_does_not_claim_success_or_delete_possible_attachment(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool)

    async def failing_emitter(event):
        if event["type"] == "files":
            raise TimeoutError("No delivery acknowledgement")

    result, _ = display(image_tool, monkeypatch, runtime, emitter=failing_emitter)
    assert result["status"] == "delivery_unknown" and result["displayed"] is False
    assert runtime.uploads and runtime.discarded == []


@pytest.mark.parametrize("untrusted_id", [
    "05_Acrobat.docx", "/api/v1/files/source/content", "https://untrusted.invalid/document.docx",
    "../../private.docx",
])
def test_filename_url_or_path_is_not_accepted_as_a_file_id(image_tool, monkeypatch, untrusted_id):
    runtime = FakeRuntime(image_tool)
    result, events = display(image_tool, monkeypatch, runtime, file_id=untrusted_id)
    assert result["displayed"] is False
    assert runtime.authorizations == [] and runtime.uploads == []
    assert not any(event["type"] == "files" for event in events)


@pytest.mark.parametrize("allowed_ids", [[], ["not-a-collection-id"], [KNOWLEDGE_ID] * 21])
def test_missing_or_invalid_collection_configuration_fails_closed(image_tool, monkeypatch, allowed_ids):
    runtime = FakeRuntime(image_tool)
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    tool = image_tool.Tools()
    tool.valves.allowed_knowledge_ids = allowed_ids

    async def emitter(event):
        raise AssertionError("No files may be emitted without a valid approved collection allowlist")

    result = json.loads(run(tool.show_instruction_images(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(), __event_emitter__=emitter,
    )))
    assert result["displayed"] is False and result["code"] == "not_configured"
    assert runtime.authorizations == [] and runtime.uploads == []


def install_modules(monkeypatch, modules):
    """Install only the versioned integration boundary, not a fake authorization helper."""
    for name, members in modules.items():
        parts = name.split(".")
        for count in range(1, len(parts)):
            package_name = ".".join(parts[:count])
            if package_name not in sys.modules:
                package = ModuleType(package_name)
                package.__path__ = []
                monkeypatch.setitem(sys.modules, package_name, package)
        module = ModuleType(name)
        for key, value in members.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)


def source_store(image_tool, monkeypatch, *, db_role="user", member=True, permitted=True,
                 filename="05_Acrobat.docx", available=True):
    calls = []
    file = SimpleNamespace(id=SOURCE_ID, filename=filename, path="authorized-storage-key", meta={},
                           updated_at=1, hash="original-hash")
    user = SimpleNamespace(id="employee", role=db_role)

    class Files:
        @staticmethod
        async def get_file_by_id(file_id):
            calls.append(("file", file_id))
            return file if available and file_id == SOURCE_ID else None

    class Users:
        @staticmethod
        async def get_user_by_id(user_id):
            calls.append(("user", user_id))
            return user

    class Knowledges:
        @staticmethod
        async def has_file(knowledge_id, file_id):
            calls.append(("membership", knowledge_id, file_id))
            return member and knowledge_id == KNOWLEDGE_ID and file_id == SOURCE_ID

        @staticmethod
        async def check_access_by_user_id(knowledge_id, user_id, permission):
            calls.append(("access", knowledge_id, user_id, permission))
            return permitted

    install_modules(monkeypatch, {
        "open_webui.models.files": {"Files": Files},
        "open_webui.models.users": {"Users": Users},
        "open_webui.models.knowledge": {"Knowledges": Knowledges},
    })
    return file, user, calls


def test_approved_collection_and_actual_user_permission_required(image_tool, monkeypatch):
    file, user, calls = source_store(image_tool, monkeypatch)
    source = run(image_tool._OpenWebUI().source(SOURCE_ID, "employee", [KNOWLEDGE_ID]))
    assert source.file is file and source.user is user and source.knowledge_id == KNOWLEDGE_ID
    assert ("membership", KNOWLEDGE_ID, SOURCE_ID) in calls
    assert ("access", KNOWLEDGE_ID, "employee", "read") in calls


@pytest.mark.parametrize("db_role,member,permitted", [
    ("user", True, False),
    ("user", False, True),
    ("admin", False, True),
    ("pending", True, True),
])
def test_admin_or_existing_file_access_cannot_bypass_approved_collection(
        image_tool, monkeypatch, db_role, member, permitted):
    source_store(image_tool, monkeypatch, db_role=db_role, member=member, permitted=permitted)
    with pytest.raises(image_tool.ToolError, match="доступ"):
        run(image_tool._OpenWebUI().source(SOURCE_ID, "employee", [KNOWLEDGE_ID]))


def test_forged_admin_role_from_tool_context_is_not_trusted(image_tool, monkeypatch):
    source_store(image_tool, monkeypatch, db_role="user", permitted=False)

    async def emitter(event):
        raise AssertionError("An unauthorized request must not emit attachments")

    result = json.loads(run(image_tool.Tools().show_instruction_images(
        SOURCE_ID, __user__={"id": "employee", "role": "admin"}, __request__=SimpleNamespace(),
        __event_emitter__=emitter,
    )))
    assert result["displayed"] is False and result["code"] == "access_denied"


@pytest.mark.parametrize("filename", ["ИТ_прошивка.docx", "СПРАВОЧНИК_администрирование.docx", "photo.png"])
def test_staff_only_documents_and_non_docx_files_are_excluded(image_tool, monkeypatch, filename):
    source_store(image_tool, monkeypatch, filename=filename)
    with pytest.raises(image_tool.ToolError):
        run(image_tool._OpenWebUI().source(SOURCE_ID, "employee", [KNOWLEDGE_ID]))


def test_collection_uuid_does_not_substitute_for_file_uuid(image_tool, monkeypatch):
    source_store(image_tool, monkeypatch)
    with pytest.raises(image_tool.ToolError):
        run(image_tool._OpenWebUI().source(KNOWLEDGE_ID, "employee", [KNOWLEDGE_ID]))


@pytest.mark.parametrize("size", [None, "12", True, 0, -1, 30_000_001])
def test_storage_not_called_for_unbounded_or_invalid_source_size(image_tool, monkeypatch, size):
    class Storage:
        @staticmethod
        def get_file(path):
            raise AssertionError("Invalid metadata must be rejected before reading object storage")

    install_modules(monkeypatch, {"open_webui.storage.provider": {"Storage": Storage}})
    source = image_tool.Source(SimpleNamespace(id="employee"),
                               SimpleNamespace(path="authorized-storage-key", meta={"size": size}), KNOWLEDGE_ID)
    with pytest.raises(image_tool.ToolError):
        run(image_tool._OpenWebUI().read_original(source))


@pytest.mark.parametrize("declared_size", [12, 11])
def test_original_read_is_bounded_and_uses_only_authorized_storage_path(image_tool, monkeypatch, declared_size):
    data = b"source bytes"
    calls = []

    class Storage:
        @staticmethod
        def get_file(path):
            calls.append(("storage", path))
            return "resolved-authorized-storage-path"

    class BoundedReader(io.BytesIO):
        def read(self, limit=-1):
            assert limit == image_tool.MAX_SOURCE_BYTES + 1
            return super().read(limit)

    def resolved_path(path):
        assert path == "resolved-authorized-storage-path"
        return SimpleNamespace(open=lambda mode: BoundedReader(data))

    install_modules(monkeypatch, {"open_webui.storage.provider": {"Storage": Storage}})
    monkeypatch.setattr(image_tool, "Path", resolved_path)
    source = image_tool.Source(SimpleNamespace(id="employee"),
                               SimpleNamespace(path="authorized-database-path", meta={"size": declared_size}),
                               KNOWLEDGE_ID)
    if declared_size == len(data):
        assert run(image_tool._OpenWebUI().read_original(source)) == data
    else:
        with pytest.raises(image_tool.ToolError, match="Размер"):
            run(image_tool._OpenWebUI().read_original(source))
    assert calls == [("storage", "authorized-database-path")]


@pytest.mark.parametrize("metadata", [
    None, "not metadata", {}, {"chat_id": "local:temporary", "message_id": "reply"},
    {"chat_id": CHAT_ID}, {"chat_id": CHAT_ID, "message_id": ""},
    {"chat_id": CHAT_ID, "message_id": 12},
])
def test_current_reply_files_ignore_missing_or_temporary_metadata(image_tool, metadata):
    # No WebUI modules installed: invalid metadata must return before importing/querying Chats.
    assert run(image_tool._OpenWebUI().current_files(metadata, "employee")) == []


@pytest.mark.parametrize("owned", [True, False])
def test_current_reply_files_are_read_only_from_callers_owned_chat(image_tool, monkeypatch, owned):
    calls = []
    files = [{"type": "image", "url": "/api/v1/files/private-picture/content"}]

    class Chats:
        @staticmethod
        async def get_chat_by_id_and_user_id(chat_id, user_id):
            calls.append(("owner", chat_id, user_id))
            return SimpleNamespace(id=chat_id, user_id=user_id) if owned else None

        @staticmethod
        async def get_message_by_id_and_message_id(chat_id, message_id):
            calls.append(("message", chat_id, message_id))
            assert owned, "Foreign chat messages must never be read"
            return {"files": files}

    install_modules(monkeypatch, {"open_webui.models.chats": {"Chats": Chats}})
    result = run(image_tool._OpenWebUI().current_files(
        {"chat_id": CHAT_ID, "message_id": "assistant-reply"}, "employee"))
    assert result == (files if owned else [])
    assert calls[0] == ("owner", CHAT_ID, "employee")
    assert calls[1:] == ([("message", CHAT_ID, "assistant-reply")] if owned else [])


@pytest.mark.parametrize("message", [None, {}, {"files": None}, {"files": {"url": "not a list"}}])
def test_current_reply_files_handle_missing_or_malformed_message_data(image_tool, monkeypatch, message):
    class Chats:
        @staticmethod
        async def get_chat_by_id_and_user_id(chat_id, user_id):
            return SimpleNamespace(id=chat_id, user_id=user_id)

        @staticmethod
        async def get_message_by_id_and_message_id(chat_id, message_id):
            return message

    install_modules(monkeypatch, {"open_webui.models.chats": {"Chats": Chats}})
    assert run(image_tool._OpenWebUI().current_files(
        {"chat_id": CHAT_ID, "message_id": "missing-reply"}, "employee")) == []


def test_upload_uses_authorized_user_without_document_ingestion(image_tool, monkeypatch):
    file = SimpleNamespace(id=SOURCE_ID)
    user = SimpleNamespace(id="employee")
    source = image_tool.Source(user, file, KNOWLEDGE_ID)
    received = []
    upload_id = str(UUID(int=101))

    async def upload_file_handler(**kwargs):
        received.append(kwargs)
        assert kwargs["user"] is user
        assert kwargs["process"] is False and kwargs["process_in_background"] is False
        assert kwargs["file"].content_type == "image/png"
        assert kwargs["metadata"]["source_file_id"] == SOURCE_ID
        assert kwargs["metadata"]["caption"] == "Нажмите «ОК»."
        assert kwargs["metadata"]["illustration_number"] == 1
        assert await kwargs["file"].read() == picture.data
        return SimpleNamespace(id=upload_id, user_id="employee")

    install_modules(monkeypatch, {"open_webui.routers.files": {"upload_file_handler": upload_file_handler}})
    picture = image_tool.Picture(1, "Нажмите «ОК».", png())
    uploaded = run(image_tool._OpenWebUI().upload_image(SimpleNamespace(), source, picture, "a" * 64))
    assert uploaded.id == upload_id and len(received) == 1
    assert received[0]["file"].file.closed


def test_upload_rejects_file_owned_by_other_account(image_tool, monkeypatch):
    async def upload_file_handler(**kwargs):
        return SimpleNamespace(id=str(UUID(int=101)), user_id="another-employee")

    install_modules(monkeypatch, {"open_webui.routers.files": {"upload_file_handler": upload_file_handler}})
    source = image_tool.Source(SimpleNamespace(id="employee"), SimpleNamespace(id=SOURCE_ID), KNOWLEDGE_ID)
    with pytest.raises(image_tool.ToolError):
        run(image_tool._OpenWebUI().upload_image(SimpleNamespace(), source,
                                               image_tool.Picture(1, "Кнопка.", png()), "a" * 64))


def test_channel_context_cannot_falsely_confirm_photo_display(image_tool):
    async def emitter(event):
        raise AssertionError("Channel emitters do not support this tool's image attachments")

    result = json.loads(run(image_tool.Tools().show_instruction_images(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
        __event_emitter__=emitter, __metadata__={"chat_id": "channel:warehouse", "message_id": "reply"},
    )))
    assert result["status"] == "error" and result["displayed"] is False
    assert result["code"] == "unsupported_context"
