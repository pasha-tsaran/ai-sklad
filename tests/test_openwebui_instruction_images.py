"""Exercise the importable Workspace Tool without an Open WebUI installation."""

import asyncio
import copy
import importlib.util
import inspect
import io
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import UUID
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

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
            # Identical instruction fixtures must keep the same source hash
            # even when two operations cross the ZIP clock's two-second tick.
            entry = ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0))
            archive.writestr(entry, content, compress_type=ZIP_DEFLATED)
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
    assert image_tool.parse_image_numbers("auto", 3) == [1, 2, 3]
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


def test_read_docx_text_preserves_body_warnings_and_does_not_duplicate_visible_caption(image_tool):
    caption = "Нажмите «ОК»."
    warning = "Если принтер не принимает задания, обратитесь в поддержку."
    data = instruction((caption,), extra_paragraph=(
        f"<w:p><w:r><w:t>{escape(caption)}</w:t></w:r></w:p>"
        f"<w:p><w:r><w:t>{warning}</w:t></w:r></w:p>"
    ))
    text = image_tool.read_docx_text(data)
    assert text == f"Утверждённая инструкция Acrobat\n\n{caption}\n\n{warning}"
    assert text.count(caption) == 1


def test_read_docx_text_fails_instead_of_truncating_conditions(image_tool):
    data = instruction(extra_paragraph=f'<w:p><w:r><w:t>{"x" * 200001}</w:t></w:r></w:p>')
    with pytest.raises(image_tool.ToolError) as exc:
        image_tool.read_docx_text(data)
    assert exc.value.code == "text_too_large"


@pytest.mark.parametrize("data", [b"not a docx", office({
    "word/document.xml": '<!DOCTYPE x [<!ENTITY value "unsafe">]><x>&value;</x>',
})])
def test_read_docx_text_rejects_bad_archive_or_xml_entities(image_tool, data):
    with pytest.raises(image_tool.ToolError):
        image_tool.read_docx_text(data)


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


def display(image_tool, monkeypatch, runtime, *, selection="auto", offset=0, emitter=None, file_id=SOURCE_ID,
            metadata=None, allow_continuation=False):
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    events = []

    async def record(event):
        events.append(event)

    result = json.loads(run(image_tool.Tools().show_instruction_images(
        file_id, selection, offset=offset, __user__={"id": "employee"}, __request__=SimpleNamespace(),
        __event_emitter__=emitter or record, __metadata__=metadata, __allow_continuation__=allow_continuation,
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


def test_inspector_returns_full_approved_text_and_captions_without_upload_or_emitter(image_tool, monkeypatch):
    caption = "Нажмите «Дополнительно» в верхней части окна печати."
    warning = "Если связи с принтером нет, нужна помощь поддержки."
    runtime = FakeRuntime(image_tool, instruction((caption,), extra_paragraph=(
        f"<w:p><w:r><w:t>{warning}</w:t></w:r></w:p>"
    )))
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "ready"
    assert result["source"]["file_id"] == SOURCE_ID and result["source"]["knowledge_id"] == KNOWLEDGE_ID
    assert len(result["source"]["sha256"]) == 64
    assert result["content"] == f"Утверждённая инструкция Acrobat\n\n{caption}\n\n{warning}"
    assert result["images"] == [{"number": 1, "caption": caption}]
    assert len(runtime.authorizations) == 2 and not runtime.uploads and not runtime.discarded


@pytest.mark.parametrize("change", ["revoke_on_recheck", "replace_on_recheck"])
def test_inspector_rechecks_access_and_source_before_disclosing_content(image_tool, monkeypatch, change):
    runtime = FakeRuntime(image_tool)
    setattr(runtime, change, True)
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error"
    assert not {"source", "content", "images"}.intersection(result)
    assert len(runtime.authorizations) == 2 and not runtime.uploads


@pytest.mark.parametrize("user,request_context", [(None, SimpleNamespace()), ({"id": "employee"}, None)])
def test_inspector_requires_authenticated_request_context(image_tool, user, request_context):
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__=user, __request__=request_context,
    )))
    assert result["status"] == "error" and result["code"] == "missing_context"
    assert "content" not in result


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


@pytest.mark.parametrize("selection", ["auto", "all", "1,2,3,4,5,6,7"])
def test_large_instruction_shows_first_group_without_requesting_photo_numbers(image_tool, monkeypatch, selection):
    runtime = FakeRuntime(image_tool, instruction(tuple(f"Кнопка {number}." for number in range(7))))
    result, events = display(image_tool, monkeypatch, runtime, selection=selection)
    assert result["displayed"] is True and result["status"] == "displayed"
    assert [picture["number"] for picture in result["images"]] == [1, 2, 3, 4, 5, 6]
    assert result["has_more"] is True and result["next_offset"] == 6
    assert result["offset"] == 0 and result["total_images"] == 7
    assert result["progress_message"] == "После этих шагов напишите «Готово», чтобы продолжить."
    assert "выберите" not in result["message"].casefold() and "6" not in result["message"]
    assert len(runtime.uploads) == 6
    assert len([event for event in events if event["type"] == "files"]) == 1


def test_large_instruction_first_group_is_reauthorized_before_disclosure(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple(f"Кнопка {number}." for number in range(7))))
    runtime.revoke_on_recheck = True
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["displayed"] is False and result["status"] == "error"
    assert "images" not in result
    assert runtime.discarded == [upload.id for _, upload in runtime.uploads]
    assert not any(event["type"] == "files" for event in events)


def test_continuation_shows_only_next_group_then_finishes(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple(f"Кнопка {number}." for number in range(14))))
    first, _ = display(image_tool, monkeypatch, runtime)
    second, _ = display(image_tool, monkeypatch, runtime, offset=first["next_offset"], allow_continuation=True)
    last, _ = display(image_tool, monkeypatch, runtime, offset=second["next_offset"], allow_continuation=True)
    assert [picture["number"] for picture in first["images"]] == [1, 2, 3, 4, 5, 6]
    assert [picture["number"] for picture in second["images"]] == [7, 8, 9, 10, 11, 12]
    assert [picture["number"] for picture in last["images"]] == [13, 14]
    assert second["has_more"] is True and second["next_offset"] == 12
    assert last["has_more"] is False and last["next_offset"] is None and last["progress_message"] is None
    assert [picture.number for picture, _ in runtime.uploads] == list(range(1, 15))


def test_explicit_subset_continuation_keeps_subset_positions(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple(f"Кнопка {number}." for number in range(14))))
    selector = "1,3,5,7,9,11,13"
    first, _ = display(image_tool, monkeypatch, runtime, selection=selector)
    last, _ = display(image_tool, monkeypatch, runtime, selection=selector, offset=first["next_offset"],
                      allow_continuation=True)
    assert [picture["number"] for picture in first["images"]] == [1, 3, 5, 7, 9, 11]
    assert [picture["number"] for picture in last["images"]] == [13]
    assert last["total_images"] == 7 and last["offset"] == 6
    assert not last["has_more"] and last["next_offset"] is None


@pytest.mark.parametrize("offset", [-1, True, False, "6", None, 1.5, 7, 100_000])
def test_invalid_progress_does_not_upload_or_repeat_photos(image_tool, monkeypatch, offset):
    runtime = FakeRuntime(image_tool, instruction(tuple("Кнопка." for _ in range(7))))
    result, events = display(image_tool, monkeypatch, runtime, offset=offset, allow_continuation=True)
    assert result["displayed"] is False and result["code"] == "invalid_offset"
    assert not runtime.uploads and not any(event["type"] == "files" for event in events)


@pytest.mark.parametrize("flag", [False, None, "true", "false", 1, 0, {}, []])
def test_public_continuation_cannot_self_authorize_with_a_truthy_value(image_tool, monkeypatch, flag):
    runtime = FakeRuntime(image_tool, instruction(tuple("Кнопка." for _ in range(14))))
    result, events = display(image_tool, monkeypatch, runtime, offset=6, allow_continuation=flag)
    assert result["status"] == "awaiting_confirmation" and result["displayed"] is False
    assert result["message"] == "Продолжение будет показано после подтверждения предыдущих шагов."
    assert not runtime.authorizations and not runtime.uploads and not events


def test_model_cannot_call_first_then_next_page_in_the_same_reply(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple("Кнопка." for _ in range(14))))
    first, first_events = display(image_tool, monkeypatch, runtime)
    authorizations_after_first = list(runtime.authorizations)
    assert first["displayed"] is True and len(runtime.uploads) == 6 and len(first_events) == 1
    next_page, next_events = display(image_tool, monkeypatch, runtime, offset=first["next_offset"])
    assert next_page["status"] == "awaiting_confirmation" and next_page["displayed"] is False
    assert runtime.authorizations == authorizations_after_first
    assert len(runtime.uploads) == 6 and not next_events


def test_specific_photo_for_ui_clarification_remains_available_without_page_advance(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool, instruction(tuple("Кнопка." for _ in range(14))))
    result, events = display(image_tool, monkeypatch, runtime, selection="7", offset=0)
    assert result["displayed"] is True and [picture["number"] for picture in result["images"]] == [7]
    assert len(runtime.uploads) == 1 and len(events) == 1


def test_shared_batch_helper_preserves_approved_subset_order_and_bytes(image_tool):
    pictures = [image_tool.Picture(number, f"Фото {number}.", bytes([number])) for number in range(1, 15)]
    batch = image_tool.select_image_batch(pictures, "13,11,9,7,5,3,1,3", offset=6)
    assert [picture.number for picture in batch["pictures"]] == [13]
    assert batch["pictures"][0] is pictures[12]
    assert batch == {"pictures": [pictures[12]], "offset": 6, "next_offset": None,
                     "has_more": False, "total_images": 7}


def test_shared_batch_helper_rejects_empty_source(image_tool):
    with pytest.raises(image_tool.ToolError, match="фотографий"):
        image_tool.select_image_batch([])


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


@pytest.mark.parametrize("member,permitted", [(False, True), (True, False)])
def test_inspector_cannot_disclose_unapproved_or_forbidden_file(image_tool, monkeypatch, member, permitted):
    source_store(image_tool, monkeypatch, member=member, permitted=permitted)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == "access_denied"
    assert not {"source", "content", "images"}.intersection(result)


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


@pytest.mark.parametrize("filename", ["ИТ_прошивка.docx", "СПРАВОЧНИК_администрирование.docx"])
def test_filename_prefix_does_not_override_approved_collection_access(image_tool, monkeypatch, filename):
    source_store(image_tool, monkeypatch, filename=filename)
    selected = run(image_tool._OpenWebUI().source(SOURCE_ID, "employee", [KNOWLEDGE_ID]))
    assert selected.file.filename == filename and selected.knowledge_id == KNOWLEDGE_ID


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


@pytest.mark.parametrize("filename", ["Инструкция.md", "Таблица.xlsx", "Инструкция.pdf"])
def test_reader_uses_complete_server_extracted_text_for_approved_non_docx(image_tool, monkeypatch, filename):
    runtime = FakeRuntime(image_tool)
    runtime.file.filename = filename
    runtime.file.data = {"content": "Утверждённая инструкция.\n\nЕсли шаг не помог, обратитесь в поддержку."}
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "ready" and result["content"] == runtime.file.data["content"]
    assert result["source"]["filename"] == filename and result["images"] == []
    assert len(result["source"]["sha256"]) == 64
    assert len(runtime.authorizations) == 2 and not runtime.uploads


@pytest.mark.parametrize("content,code", [
    (None, "unavailable_text"), ("", "unavailable_text"), ("x" * 200001, "text_too_large"),
], ids=["missing", "empty", "too-large"])
def test_non_docx_reader_never_invents_or_truncates_unavailable_text(image_tool, monkeypatch, content, code):
    runtime = FakeRuntime(image_tool)
    runtime.file.filename = "Инструкция.md"
    runtime.file.data = {"content": content}
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == code
    assert "content" not in result and "source" not in result


def test_non_docx_text_change_without_metadata_change_prevents_disclosure(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool)
    runtime.file.filename = "Инструкция.md"
    runtime.file.data = {"content": "Первоначальная инструкция."}
    original_source = runtime.source

    async def source(*args):
        result = await original_source(*args)
        if len(runtime.authorizations) == 2:
            result.file.data["content"] = "Текст изменился во время чтения."
        return result

    runtime.source = source
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == "source_changed"
    assert "content" not in result and "source" not in result


def test_non_docx_photo_request_has_no_upload_or_display_event(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool)
    runtime.file.filename = "Таблица.xlsx"
    runtime.file.data = {"content": "Текст таблицы, извлечённый сервером."}
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["status"] == "error" and result["code"] == "unsupported_document"
    assert not runtime.uploads and not events


def scoped_search_store(monkeypatch, *, role="user", permitted=True, exists=True):
    foreign_id = "00000000-0000-4000-8000-000000000099"
    internal_id = "00000000-0000-4000-8000-000000000098"
    files = {
        SOURCE_ID: SimpleNamespace(id=SOURCE_ID, filename="Сканер.docx", path="approved", updated_at=1,
                                   hash="approved-hash", meta={}),
        foreign_id: SimpleNamespace(id=foreign_id, filename="Другой бот.docx", path="foreign", updated_at=1,
                                    hash="foreign-hash", meta={}),
        internal_id: SimpleNamespace(id=internal_id, filename="ИТ_Администратору.docx", path="internal", updated_at=1,
                                     hash="internal-hash", meta={}),
    }
    state = SimpleNamespace(role=role, permitted=permitted, exists=exists, external=False, calls=[], queries=[], memberships=0,
                            revoke=False, replace=False, chunks=[
                                {"content": "Разрешённый результат.", "source": "Неверное имя из индекса", "file_id": SOURCE_ID},
                                {"content": "FOREIGN_KNOWLEDGE_CANARY", "source": "Другой бот.docx", "file_id": foreign_id},
                                {"content": "INTERNAL_KNOWLEDGE_CANARY", "source": "ИТ_Администратору.docx", "file_id": internal_id},
                                {"content": "NOTE_KNOWLEDGE_CANARY", "source": "Заметка", "file_id": SOURCE_ID, "type": "note"},
                            ])

    class Users:
        @staticmethod
        async def get_user_by_id(user_id):
            state.calls.append(("user", user_id))
            return SimpleNamespace(id=user_id, role=state.role)

    class Files:
        @staticmethod
        async def get_file_by_id(file_id):
            state.calls.append(("file", file_id))
            return files.get(file_id)

    class Knowledges:
        @staticmethod
        async def get_knowledge_by_id(knowledge_id):
            state.calls.append(("knowledge", knowledge_id))
            return (SimpleNamespace(id=knowledge_id, meta={"source": "external"} if state.external else {})
                    if state.exists and knowledge_id == KNOWLEDGE_ID else None)

        @staticmethod
        async def check_access_by_user_id(knowledge_id, user_id, permission):
            state.calls.append(("access", knowledge_id, user_id, permission))
            return state.permitted

        @staticmethod
        async def has_file(knowledge_id, file_id):
            state.calls.append(("membership", knowledge_id, file_id))
            if file_id == SOURCE_ID:
                state.memberships += 1
                if state.memberships > 1 and state.replace:
                    files[SOURCE_ID].hash = "replacement-hash"
                if state.memberships > 1 and state.revoke:
                    return False
            return knowledge_id == KNOWLEDGE_ID and file_id in {SOURCE_ID, internal_id}

    async def query_knowledge_files(**kwargs):
        state.queries.append(kwargs)
        return json.dumps(state.chunks)

    install_modules(monkeypatch, {
        "open_webui.models.users": {"Users": Users},
        "open_webui.models.files": {"Files": Files},
        "open_webui.models.knowledge": {"Knowledges": Knowledges},
        "open_webui.tools.builtin": {"query_knowledge_files": query_knowledge_files},
    })
    return state


@pytest.mark.parametrize("role", ["user", "admin"])
def test_search_forces_approved_scope_and_filters_foreign_results_even_for_admin(image_tool, monkeypatch, role):
    state = scoped_search_store(monkeypatch, role=role)
    request = SimpleNamespace(state=SimpleNamespace(metadata={"folder_knowledge": [
        {"type": "collection", "id": "foreign-collection"},
    ]}))
    result = json.loads(run(image_tool.Tools().search_approved_instructions(
        "Настройка сканера", __user__={"id": "employee", "role": "admin"}, __request__=request,
    )))
    elapsed = result.pop("timing_ms")
    assert type(elapsed) in {int, float} and math.isfinite(elapsed) and 0 <= elapsed <= 3_600_000
    assert result == {"status": "ok", "knowledge_ids": [KNOWLEDGE_ID], "chunks": [
        {"content": "Разрешённый результат.", "source": "Сканер.docx", "file_id": SOURCE_ID},
        {"content": "INTERNAL_KNOWLEDGE_CANARY", "source": "ИТ_Администратору.docx",
         "file_id": "00000000-0000-4000-8000-000000000098"},
    ]}
    assert state.queries[0]["__model_knowledge__"] == [{"type": "collection", "id": KNOWLEDGE_ID}]
    assert state.queries[0]["knowledge_ids"] == [KNOWLEDGE_ID]
    assert state.queries[0]["__user__"] == {"id": "employee", "role": role}
    assert state.queries[0]["count"] == 8 and state.queries[0]["__request__"] is request
    assert "FOREIGN_KNOWLEDGE_CANARY" not in json.dumps(result)
    assert "NOTE_KNOWLEDGE_CANARY" not in json.dumps(result)


@pytest.mark.parametrize("case", ["missing_collection", "no_permission", "pending_user"])
def test_search_checks_actual_collection_permission_before_query(image_tool, monkeypatch, case):
    state = scoped_search_store(monkeypatch, role="pending" if case == "pending_user" else "user",
                                permitted=case != "no_permission", exists=case != "missing_collection")
    result = json.loads(run(image_tool.Tools().search_approved_instructions(
        "Сканер", __user__={"id": "employee", "role": "admin"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == "access_denied"
    assert "chunks" not in result and not state.queries


@pytest.mark.parametrize("allowed", [[], ["invalid-id"], [KNOWLEDGE_ID] * 21])
def test_search_with_no_valid_allowlist_cannot_fall_back_to_all_knowledge(image_tool, monkeypatch, allowed):
    state = scoped_search_store(monkeypatch)
    tool = image_tool.Tools()
    tool.valves.allowed_knowledge_ids = allowed
    result = json.loads(run(tool.search_approved_instructions(
        "Сканер", __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == "not_configured"
    assert not state.calls and not state.queries and "chunks" not in result


@pytest.mark.parametrize("changed", ["revoke", "replace"])
def test_search_rechecks_membership_and_source_before_returning_chunks(image_tool, monkeypatch, changed):
    state = scoped_search_store(monkeypatch)
    setattr(state, changed, True)
    result = json.loads(run(image_tool.Tools().search_approved_instructions(
        "Сканер", __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and "chunks" not in result


@pytest.mark.parametrize("query,count", [("", 8), ("x" * 6001, 8), ("Сканер", 0), ("Сканер", 21), ("Сканер", True)],
                         ids=["empty", "too-large", "zero-count", "large-count", "boolean-count"])
def test_search_rejects_unbounded_or_invalid_requests(image_tool, monkeypatch, query, count):
    state = scoped_search_store(monkeypatch)
    result = json.loads(run(image_tool.Tools().search_approved_instructions(
        query, count=count, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == "invalid_query"
    assert not state.queries and not state.calls


def test_search_does_not_contact_external_knowledge_backend(image_tool, monkeypatch):
    state = scoped_search_store(monkeypatch)
    state.external = True
    result = json.loads(run(image_tool.Tools().search_approved_instructions(
        "Сканер", __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "error" and result["code"] == "unsupported_knowledge"
    assert not state.queries and "chunks" not in result


@pytest.mark.parametrize("user,request_context", [(None, SimpleNamespace()), ({"id": "employee"}, None)])
def test_search_requires_current_authenticated_request(image_tool, monkeypatch, user, request_context):
    state = scoped_search_store(monkeypatch)
    result = json.loads(run(image_tool.Tools().search_approved_instructions(
        "Сканер", __user__=user, __request__=request_context,
    )))
    assert result["status"] == "error" and result["code"] == "missing_context"
    assert not state.calls and not state.queries and "chunks" not in result


@pytest.mark.parametrize("name,parameters", [
    ("search_approved_instructions", ["self", "query", "count", "__user__", "__request__"]),
    ("read_approved_instruction", ["self", "file_id", "__user__", "__request__"]),
    ("show_instruction_images", ["self", "file_id", "image_numbers", "offset", "__user__", "__request__",
                                 "__event_emitter__", "__metadata__", "__allow_continuation__"]),
])
def test_timing_preserves_native_tool_schema_and_documentation(image_tool, name, parameters):
    method = getattr(image_tool.Tools, name)
    original = inspect.unwrap(method)
    signature = inspect.signature(method)
    assert list(signature.parameters) == parameters
    assert signature == inspect.signature(original)
    assert method.__annotations__ == original.__annotations__
    assert method.__doc__ == original.__doc__ and ":param" in method.__doc__
    assert method.__name__ == name and inspect.iscoroutinefunction(method)
    assert not any(parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
                   for parameter in signature.parameters.values())
    public_methods = [attribute for attribute in dir(image_tool.Tools())
                      if not attribute.startswith("_") and callable(getattr(image_tool.Tools(), attribute))
                      and not inspect.isclass(getattr(image_tool.Tools(), attribute))]
    assert sorted(public_methods) == ["read_approved_instruction", "search_approved_instructions",
                                      "show_instruction_images"]


def measured_operation(image_tool, monkeypatch, operation, *, decorated, denied=False):
    """Compare a real public operation with its original implementation and effects."""
    tool = image_tool.Tools()
    method = getattr(tool, operation) if decorated else getattr(image_tool.Tools, operation).__wrapped__
    if not decorated:
        method = method.__get__(tool, image_tool.Tools)
    if operation == "search_approved_instructions":
        state = scoped_search_store(monkeypatch, permitted=not denied)
        result = json.loads(run(method("Настройка сканера", __user__={"id": "employee"},
                                       __request__=SimpleNamespace())))
        return result, {"queries": len(state.queries), "calls": state.calls}

    runtime = FakeRuntime(image_tool, instruction(("Первая кнопка.", "Вторая кнопка.")))
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    events = []

    async def emit(event):
        events.append(event)

    arguments = {"file_id": SOURCE_ID, "__user__": None if denied else {"id": "employee"},
                 "__request__": SimpleNamespace()}
    if operation == "show_instruction_images":
        arguments.update(image_numbers="1,2", __event_emitter__=emit)
    result = json.loads(run(method(**arguments)))
    return result, {"authorizations": runtime.authorizations, "uploads": runtime.uploads,
                    "discarded": runtime.discarded, "events": events}


@pytest.mark.parametrize("operation", ["search_approved_instructions", "read_approved_instruction",
                                       "show_instruction_images"])
def test_timing_changes_only_metric_and_never_repeats_source_or_photo_delivery(image_tool, monkeypatch, operation):
    expected, expected_effects = measured_operation(image_tool, monkeypatch, operation, decorated=False)
    ticks = iter([10, 10.0125])
    monkeypatch.setattr(image_tool, "perf_counter", lambda: next(ticks))
    result, effects = measured_operation(image_tool, monkeypatch, operation, decorated=True)
    assert result.pop("timing_ms") == 12.5
    assert result == expected and effects == expected_effects
    if operation == "search_approved_instructions":
        assert effects["queries"] == 1
    else:
        assert len(effects["authorizations"]) == 2
        assert len(effects["uploads"]) == (2 if operation == "show_instruction_images" else 0)
        assert len(effects["events"]) == (1 if operation == "show_instruction_images" else 0)


@pytest.mark.parametrize("operation", ["search_approved_instructions", "read_approved_instruction",
                                       "show_instruction_images"])
@pytest.mark.parametrize("failure", ["start", "end"])
@pytest.mark.parametrize("denied", [False, True], ids=["success", "denied"])
def test_failed_timing_clock_preserves_real_operation_and_access_denial(image_tool, monkeypatch,
                                                                      operation, failure, denied):
    expected, expected_effects = measured_operation(image_tool, monkeypatch, operation,
                                                  decorated=False, denied=denied)
    calls = []

    def clock():
        calls.append(None)
        if failure == "start" or len(calls) == 2:
            raise OSError("Timing clock unavailable")
        return 10

    monkeypatch.setattr(image_tool, "perf_counter", clock)
    result, effects = measured_operation(image_tool, monkeypatch, operation, decorated=True, denied=denied)
    assert "timing_ms" not in result
    assert result == expected and effects == expected_effects
    assert len(calls) == (1 if failure == "start" else 2)
    if denied:
        assert result["status"] == "error" and not {"content", "source", "chunks", "images"}.intersection(result)
        assert effects.get("queries", 0) == 0 and effects.get("uploads", []) == []


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), 3_600_001])
def test_invalid_elapsed_measurement_cannot_change_access_denial(image_tool, monkeypatch, elapsed):
    expected, expected_effects = measured_operation(image_tool, monkeypatch, "search_approved_instructions",
                                                  decorated=False, denied=True)
    ticks = iter([0, elapsed / 1000])
    monkeypatch.setattr(image_tool, "perf_counter", lambda: next(ticks))
    result, effects = measured_operation(image_tool, monkeypatch, "search_approved_instructions",
                                        decorated=True, denied=True)
    assert result == expected and effects == expected_effects and "timing_ms" not in result
