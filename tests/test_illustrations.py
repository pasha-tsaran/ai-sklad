import hashlib
import io
import json
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import httpx
import pytest
from PIL import Image, PngImagePlugin
from sqlalchemy import select

from app.core.config import get_settings
from app.models import Delivery, Message, MessageAttachment, SolutionAttempt
from app.services.nextcloud.delivery import TalkClient, deliver_one
from app.services.rag.documents import image_bytes, read_docx, read_xlsx
from app.services.rag.openwebui import OpenWebUIKnowledge
from tests.test_integrations import post_talk, remote_settings, setup_talk


def office(parts):
    result = io.BytesIO()
    with ZipFile(result, "w", ZIP_DEFLATED) as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    return result.getvalue()


def docx(caption="Нажмите Дополнительно вверху окна", external=False):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 12), "red").save(buffer, format="PNG")
    return office({
        "word/document.xml": f'''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
        xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
        <w:body><w:p><w:r><w:t>Печать PDF с изображениями</w:t></w:r></w:p>
        <w:p><w:r><w:t>1. Нажмите Дополнительно.</w:t></w:r></w:p>
        <w:p><w:r><w:drawing><wp:inline><wp:docPr descr="{caption}"/>
        <a:blip r:embed="picture"/></wp:inline></w:drawing></w:r></w:p></w:body></w:document>''',
        "word/_rels/document.xml.rels": f'''<Relationships><Relationship Id="picture"
        Target="{'https://untrusted.invalid/photo' if external else 'media/photo.png'}"
        TargetMode="{'External' if external else 'Internal'}"/></Relationships>''',
        "word/media/photo.png": buffer.getvalue(),
    })


def document_transport(data, filename="PDF.docx", revoke=False):
    checks = 0

    def handler(request):
        nonlocal checks
        assert request.headers["Authorization"] == "Bearer test"
        if "/knowledge/" in request.url.path:
            checks += 1
            files = [] if revoke and checks > 2 else [{"id": "file-1", "filename": filename}]
            return httpx.Response(200, json={"files": files})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"file_id":"file-1"}'}}]})
        if request.url.path.endswith("/data/content"):
            return httpx.Response(200, json={"content": "Indexed search text without image descriptions"})
        assert request.url.path.endswith("/files/file-1/content")
        return httpx.Response(200, content=data)

    return httpx.MockTransport(handler)


def test_original_docx_preserves_caption_and_picture():
    data = docx()
    candidate = OpenWebUIKnowledge(remote_settings(), document_transport(data)).search("PDF", set())[0]
    assert "Нажмите Дополнительно вверху окна" in candidate.content
    assert len(candidate.illustrations) == 1
    assert candidate.illustrations[0].caption == "Нажмите Дополнительно вверху окна"
    assert candidate.source["original_sha256"] == hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(candidate.illustrations[0].data)) as picture:
        assert picture.getpixel((0, 0)) == (255, 0, 0)


def test_image_normalization_keeps_visible_pixels_without_metadata():
    original = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
    original.putpixel((1, 0), (0, 0, 0, 0))
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Comment", "Private document metadata")
    exif = Image.Exif()
    exif[270] = "Private EXIF description"
    buffer = io.BytesIO()
    original.save(buffer, format="PNG", pnginfo=metadata, exif=exif)
    normalized = image_bytes(buffer.getvalue(), 1, "Example", 100_000)
    with Image.open(io.BytesIO(normalized.data)) as picture:
        assert picture.getpixel((0, 0)) == (255, 0, 0)
        assert picture.getpixel((1, 0)) == (255, 255, 255)
        assert picture.mode == "RGB" and picture.info == {}


@pytest.mark.parametrize("data", [docx(caption=""), docx(external=True), b"not a zip"])
def test_rejects_unexplained_or_external_pictures(data):
    with pytest.raises(ValueError):
        read_docx(data)


def test_original_size_revocation_and_withdrawn_source():
    data = docx()
    with pytest.raises(ValueError, match="too large"):
        OpenWebUIKnowledge(remote_settings(knowledge_max_file_bytes=1000),
                           document_transport(data + b"a" * 2000)).search("PDF", set())
    assert OpenWebUIKnowledge(remote_settings(), document_transport(data, revoke=True)).search("PDF", set()) == []
    assert OpenWebUIKnowledge(remote_settings(), document_transport(
        data, "33_Сканер штрихкодов в базе АДРЕС.docx",
    )).search("PDF", set()) == []


def test_xlsx_rows_keep_cell_values_and_reject_formulas():
    def workbook(formula=""):
        return office({
            "xl/workbook.xml": '''<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
            <sheets><sheet name="Телефоны" r:id="s1"/></sheets></workbook>''',
            "xl/_rels/workbook.xml.rels": '<Relationships><Relationship Id="s1" Target="/xl/worksheets/sheet1.xml"/></Relationships>',
            "xl/worksheets/sheet1.xml": f'''<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
            <sheetData><row><c r="A1" t="inlineStr"><is><t>ИТ</t></is></c>
            <c r="B1">{formula}<v>141</v></c></row></sheetData></worksheet>''',
        })
    assert "A1: ИТ | B1: 141" in read_xlsx(workbook())
    with pytest.raises(ValueError, match="formulas"):
        read_xlsx(workbook('<f>WEBSERVICE("https://untrusted.invalid")</f>'))


def test_worker_persists_photos_and_owned_api(env, monkeypatch):
    get_settings().knowledge_backend = "openwebui"
    provider = OpenWebUIKnowledge(remote_settings(), document_transport(docx()))
    monkeypatch.setattr("app.workers.handlers.OpenWebUIKnowledge", lambda _: provider)
    response = env.post("/api/chat", {"message": "PDF с изображениями не печатается"}).json()
    env.drain()
    result = env.client.get("/api/incidents/" + response["incident_id"]).json()
    photos = [p for m in result["messages"] for p in m["attachments"]]
    assert len(photos) == 1
    image = env.client.get(photos[0]["url"])
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    assert "no-store" in image.headers["cache-control"]
    assert env.client.get(photos[0]["url"], headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert env.client.get(photos[0]["url"].replace(response["incident_id"], "other")).status_code == 404
    with env.factory() as session:
        attempt = session.scalar(select(SolutionAttempt))
        assert len(attempt.evidence["source"]["illustrations"]) == 1


def test_talk_upload_order_and_ambiguous_photo_not_retried(env, monkeypatch):
    setup_talk(monkeypatch)
    get_settings().knowledge_backend = "openwebui"
    provider = OpenWebUIKnowledge(remote_settings(), document_transport(docx()))
    monkeypatch.setattr("app.workers.handlers.OpenWebUIKnowledge", lambda _: provider)
    post_talk(env, "PDF с изображениями не печатается")
    env.drain()
    sent = []

    def send(*args):
        sent.append("text")
        return "100"

    def photo(*args):
        sent.append("photo")
        raise TimeoutError()

    sender = SimpleNamespace(send=send, send_image=photo)
    assert deliver_one(env.factory, get_settings(), sender)
    assert deliver_one(env.factory, get_settings(), sender)
    assert not deliver_one(env.factory, get_settings(), sender)
    assert sent == ["text", "photo"]
    with env.factory() as session:
        assert session.scalar(select(Delivery.status).join(Message).join(
            MessageAttachment, MessageAttachment.message_id == Message.id)) == "UNKNOWN"


def test_native_photo_is_private_room_share():
    settings = remote_settings(nextcloud_base_url="https://cloud.internal", nextcloud_allowed_rooms=["room-a"],
                               nextcloud_files_username="ai-support", nextcloud_files_app_password="app-secret")
    steps = []

    def handler(request):
        steps.append(request.method)
        assert request.headers["Authorization"].startswith("Basic ")
        if request.method == "MKCOL":
            return httpx.Response(201)
        if request.method == "PUT":
            assert request.content == b"photo bytes"
            return httpx.Response(201)
        from urllib.parse import parse_qs
        payload = parse_qs(request.content.decode())
        assert payload["shareType"] == ["10"] and payload["shareWith"] == ["room-a"]
        assert json.loads(payload["talkMetaData"][0])["caption"] == "Фото и пояснение"
        assert "app-secret" not in request.content.decode()
        return httpx.Response(200, json={"ocs": {"meta": {"statuscode": 100}, "data": {"id": 42}}})

    client = TalkClient(settings, httpx.MockTransport(handler))
    attachment = SimpleNamespace(data=b"photo bytes", content_type="image/png")
    assert client.send_image("room-a", "Фото и пояснение", "message", attachment) == "42"
    assert steps == ["MKCOL", "PUT", "POST"]
    with pytest.raises(ValueError, match="not allowed"):
        client.send_image("other", "text", "message", attachment)


def test_files_account_shares_are_ignored(env, monkeypatch):
    setup_talk(monkeypatch)
    get_settings().nextcloud_files_username = "files-bot"
    assert post_talk(env, "фото", user="files-bot").json() == {"ignored": True}
