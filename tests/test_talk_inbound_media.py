"""Employee photo evidence uses canonical Talk metadata and authenticated private DAV."""

import copy
import hashlib
import io
import struct
import zlib
from urllib.parse import quote, unquote
from xml.sax.saxutils import escape

import httpx
import pytest
from PIL import Image

from app.services.nextcloud.account import AccountError
from app.services.nextcloud.inbound_media import (
    MAX_IMAGE_BYTES, MAX_INCIDENT_BYTES, MAX_INBOUND_FILES, InvalidInboundMedia,
    parse_inbound_media, verify_image,
)
from tests.test_talk_account import EMPLOYEE, ROOM, SERVICE, Server, message, participant


def image_bytes(format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (12, 8), "red").save(output, format=format)
    return output.getvalue()


def upload(raw=None, mime="image/png"):
    raw = raw or image_bytes()
    return message(123, content="Не читается код. {file} Символ {other} оставьте без изменений.",
                   systemMessage="file_shared", messageParameters={"file": {
                       "type": "file", "id": "456", "name": "../../employee-private-name.png",
                       "path": "Talk/Фото ошибки 01.png", "mimetype": mime, "size": str(len(raw)),
                       "etag": "original-etag", "permissions": "1", "hide-download": "no",
                       "link": "https://attacker.example.invalid/never-follow-me",
                   }})


class MediaServer(Server):
    def __init__(self, format="PNG"):
        super().__init__()
        self.raw = image_bytes(format)
        self.mime = "image/png" if format == "PNG" else "image/jpeg"
        self.source = upload(self.raw, self.mime)
        self.dav_id, self.dav_etag = "456", "original-etag"
        self.dav_mime, self.dav_size = self.mime, len(self.raw)
        self.dav_href = None
        self.response_mime, self.response_etag = self.mime, '"original-etag"'
        self.after_stat, self.after_download = None, None
        self.redirect, self.chunked_oversize, self.bad_xml = False, False, False
        self.missing = False

    def request(self, request):
        path = request.url.path.removeprefix("/office")
        if path == "/ocs/v2.php/apps/spreed/api/v1/chat/" + ROOM:
            self.requests.append(request)
            assert request.method == "GET"
            assert dict(request.url.params) == {"lookIntoFuture": "1", "lastKnownMessageId": "123", "limit": "1",
                "timeout": "0", "includeLastKnown": "1", "setReadMarker": "0", "noStatusUpdate": "1",
                "markNotificationsAsRead": "0"}
            return httpx.Response(304) if self.missing else self.ocs([copy.deepcopy(self.source)])
        if path.startswith("/remote.php/dav/files/"):
            self.requests.append(request)
            expected = "/remote.php/dav/files/" + SERVICE + "/Talk/Фото ошибки 01.png"
            assert path == expected  # Never follow an attachment's arbitrary link/path input.
            if request.method == "PROPFIND":
                assert request.headers["Depth"] == "0"
                href = self.dav_href or "/office" + "/".join(quote(part, safe="") for part in expected.split("/"))
                body = ('<d:multistatus xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns"><d:response>'
                        '<d:href>' + escape(href) + '</d:href><d:propstat><d:prop>'
                        '<oc:fileid>' + self.dav_id + '</oc:fileid><d:getetag>"' + self.dav_etag + '"</d:getetag>'
                        '<d:getcontentlength>' + str(self.dav_size) + '</d:getcontentlength>'
                        '<d:getcontenttype>' + self.dav_mime + '</d:getcontenttype><d:resourcetype/>'
                        '</d:prop><d:status>HTTP/1.1 200 OK</d:status></d:propstat></d:response></d:multistatus>')
                if self.after_stat:
                    action, self.after_stat = self.after_stat, None
                    action()
                return httpx.Response(207, content=b'<!DOCTYPE x [<!ENTITY e "secret">]><x/>' if self.bad_xml else body.encode())
            assert request.method == "GET" and request.headers["If-Match"] == '"original-etag"'
            if self.redirect:
                return httpx.Response(302, headers={"Location": "https://attacker.example.invalid/photo"})
            if self.chunked_oversize:
                return httpx.Response(200, headers={"Content-Type": self.response_mime, "ETag": self.response_etag},
                                      stream=httpx.ByteStream(b"x" * (MAX_IMAGE_BYTES + 1)))
            if self.after_download:
                self.after_download()
            return httpx.Response(200, content=self.raw,
                                  headers={"Content-Type": self.response_mime, "ETag": self.response_etag})
        return super().request(request)


def download(server, **kwargs):
    return server.client().download_inbound_image(ROOM, "123", "456", expected_employee=EMPLOYEE, **kwargs)


def dav_gets(server):
    return [request for request in server.requests if request.method == "GET" and "/remote.php/dav/" in request.url.path]


def assert_read_only(server):
    assert all(request.method in {"GET", "PROPFIND"} for request in server.requests)
    assert all(request.url.host == "nextcloud.example" for request in server.requests)
    assert not any("/shares" in request.url.path or "/direct" in request.url.path for request in server.requests)


def test_caption_preserves_user_words_and_only_removes_matching_file_placeholders():
    result = parse_inbound_media(upload())
    assert result == {"files": [456], "caption": "Не читается код.  Символ {other} оставьте без изменений."}
    original = message(10, content="Обычный текст {file}", messageParameters={"file": {"type": "file"}})
    assert parse_inbound_media(original) == {"files": [], "caption": "Обычный текст {file}"}
    photo_only = upload()
    photo_only["message"] = "{file}"
    assert parse_inbound_media(photo_only) == {"files": [456], "caption": ""}


def test_limits_are_explicit_and_batch_overflow_is_not_silently_dropped():
    assert MAX_INBOUND_FILES == 6 and MAX_IMAGE_BYTES == 8_000_000 and MAX_INCIDENT_BYTES == 24_000_000
    source = upload()
    item = source["messageParameters"]["file"]
    source["messageParameters"] = {"file" + str(i): {**item, "id": str(i + 1)} for i in range(7)}
    result = parse_inbound_media(source)
    assert result["files"] == [] and result["error_code"] == "inbound_media_too_many"
    source["messageParameters"] = {"file" + str(i): {**item, "id": str(i + 1), "size": "8000000"} for i in range(4)}
    assert parse_inbound_media(source)["error_code"] == "inbound_media_too_large"


def test_missing_file_does_not_turn_generated_unavailable_notice_into_employee_description():
    source = message(123, content="*Другой пользователь поделился файлом, который больше недоступен*",
                     systemMessage="file_shared", messageParameters={})
    assert parse_inbound_media(source) == {"files": [], "caption": "", "error_code": "inbound_media_unavailable"}


@pytest.mark.parametrize("key,value", [("mimetype", "image/svg+xml"), ("mimetype", "application/pdf"),
                                     ("size", "8000001"), ("id", "0"), ("id", "9" * 5000)],
                         ids=["svg", "document", "oversize", "zero-id", "huge-id"])
def test_unsupported_or_invalid_photo_metadata_has_safe_error_and_keeps_caption(key, value):
    source = upload()
    source["messageParameters"]["file"][key] = value
    result = parse_inbound_media(source)
    assert result["files"] == [] and result.get("error_code")
    assert "Не читается код" in result["caption"] and "{file}" not in result["caption"]


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_download_preserves_exact_original_bytes_and_immutable_file_binding(format):
    server = MediaServer(format)
    result = download(server)
    assert result.data == server.raw and result.sha256 == hashlib.sha256(server.raw).hexdigest()
    assert result.content_type == server.mime and result.etag == "original-etag"
    assert result.file_id == "456" and result.remote_message_id == "123" and result.uploader_id == EMPLOYEE
    assert result.filename.startswith("photo-") and result.filename.endswith(".png" if format == "PNG" else ".jpg")
    assert "/" not in result.filename and "employee-private-name" not in result.filename
    assert len(dav_gets(server)) == 1
    assert len([request for request in server.requests if request.method == "PROPFIND"]) == 2
    assert len([request for request in server.requests if request.url.path.endswith("/participants")]) == 3
    assert all(request.headers.get("Authorization", "").startswith("Basic ") for request in server.requests)
    assert_read_only(server)


@pytest.mark.parametrize("scope", ["group", "foreign", "third-user", "other-employee", "other-author", "wrong-room",
                                  "wrong-message", "ordinary-message", "deleted", "missing"])
def test_download_never_fetches_photo_outside_exact_employee_private_upload(scope):
    server = MediaServer()
    if scope == "group":
        server.conversations[ROOM]["type"] = 2
    elif scope == "foreign":
        server.conversations[ROOM]["remoteServer"] = "elsewhere.example"
    elif scope == "third-user":
        server.participants[ROOM].append(participant("other-user"))
    elif scope == "other-employee":
        server.participants[ROOM][1] = participant("someone-else")
    elif scope == "other-author":
        server.source["actorId"] = "someone-else"
    elif scope == "wrong-room":
        server.source["token"] = "otherroom"
    elif scope == "wrong-message":
        server.source["id"] = 124
    elif scope == "ordinary-message":
        server.source["systemMessage"] = ""
    elif scope == "deleted":
        server.source["messageType"] = "comment_deleted"
    else:
        server.missing = True
    with pytest.raises(AccountError):
        download(server)
    assert not any("/remote.php/" in request.url.path for request in server.requests)
    assert_read_only(server)


@pytest.mark.parametrize("key,value", [("permissions", "0"), ("hide-download", "yes"),
                                     ("path", "../private.png"), ("path", "https://attacker.example/photo"),
                                     ("path", "Talk\\private.png"), ("id", "999")])
def test_file_permissions_and_server_path_are_checked_before_dav_access(key, value):
    server = MediaServer()
    server.source["messageParameters"]["file"][key] = value
    with pytest.raises(AccountError):
        download(server)
    assert not any("/remote.php/" in request.url.path for request in server.requests)
    assert_read_only(server)


@pytest.mark.parametrize("changed", ["id", "etag", "mime", "size", "href", "xml"])
def test_dav_identity_metadata_must_match_canonical_message_before_file_get(changed):
    server = MediaServer()
    if changed == "id":
        server.dav_id = "999"
    elif changed == "etag":
        server.dav_etag = "modified"
    elif changed == "mime":
        server.dav_mime = "image/svg+xml"
    elif changed == "size":
        server.dav_size += 1
    elif changed == "href":
        server.dav_href = "https://attacker.example.invalid/some-file"
    else:
        server.bad_xml = True
    with pytest.raises(AccountError):
        download(server)
    assert dav_gets(server) == []
    assert_read_only(server)


@pytest.mark.parametrize("changed", ["room-before", "room-after", "source-id", "source-path", "source-etag", "dav-etag"])
def test_privacy_and_original_file_binding_are_revalidated_around_download(changed):
    server = MediaServer()
    if changed == "room-before":
        server.after_stat = lambda: server.participants[ROOM].append(participant("third-user"))
    elif changed == "room-after":
        server.after_download = lambda: server.participants[ROOM].append(participant("third-user"))
    elif changed == "dav-etag":
        server.after_download = lambda: setattr(server, "dav_etag", "changed")
    else:
        key, value = {"source-id": ("id", "999"), "source-path": ("path", "Talk/other.png"),
                      "source-etag": ("etag", "changed")}[changed]
        server.after_download = lambda: server.source["messageParameters"]["file"].update({key: value})
    with pytest.raises(AccountError):
        download(server)
    assert len(dav_gets(server)) == (0 if changed == "room-before" else 1)
    assert_read_only(server)


@pytest.mark.parametrize("case", ["redirect", "stream-oversize", "mime", "response-etag", "not-image"])
def test_download_errors_or_forged_image_data_fail_with_safe_code_and_no_url_follow(case):
    server = MediaServer()
    if case == "redirect":
        server.redirect = True
    elif case == "stream-oversize":
        server.chunked_oversize = True
    elif case == "mime":
        server.response_mime = "text/html"
    elif case == "response-etag":
        server.response_etag = '"changed"'
    else:
        server.raw = b"<svg><script>not a photo</script></svg>"
        server.dav_size = len(server.raw)
        server.source["messageParameters"]["file"]["size"] = str(len(server.raw))
    with pytest.raises(AccountError) as error:
        download(server)
    assert str(error.value) == error.value.code and "http" not in str(error.value) and "password" not in str(error.value)
    assert_read_only(server)


def test_pixel_budget_rejects_huge_compressed_image_header_without_decoding_pixels():
    raw = bytearray(image_bytes())
    raw[16:24] = struct.pack("!II", 5000, 5000)
    raw[29:33] = struct.pack("!I", zlib.crc32(raw[12:29]))
    with pytest.raises(InvalidInboundMedia, match="invalid_inbound_media_image"):
        verify_image(bytes(raw), "image/png")


def test_percent_encoded_filename_is_never_decoded_into_traversal_or_followed_as_link():
    source = upload()
    source["messageParameters"]["file"]["path"] = "Talk/%2e%2e/private.png"
    from app.services.nextcloud.inbound_media import file_descriptor
    assert file_descriptor(source, 456)["path"] == "Talk/%2e%2e/private.png"
    encoded = "/".join(quote(part, safe="") for part in file_descriptor(source, 456)["path"].split("/"))
    assert "%252e%252e" in encoded and unquote(encoded) == "Talk/%2e%2e/private.png"
