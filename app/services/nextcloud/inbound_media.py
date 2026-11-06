"""Untrusted employee photo metadata and immutable original-byte evidence snapshots.

These photos are user evidence, never approved instructions or model vision input.
"""

import io
import re
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from PIL import Image

MAX_INBOUND_FILES = 6
MAX_IMAGE_BYTES = 8_000_000
MAX_INCIDENT_BYTES = 24_000_000
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/png": ("PNG", ".png"), "image/jpeg": ("JPEG", ".jpg")}
_KEY = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_DAV = "{DAV:}"
_OC = "{http://owncloud.org/ns}"


class InvalidInboundMedia(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class InboundImage:
    data: bytes
    content_type: str
    filename: str
    sha256: str
    remote_message_id: str
    file_id: str
    uploader_id: str
    etag: str


def _integer(value, maximum):
    if (isinstance(value, bool) or not isinstance(value, (str, int)) or len(str(value)) > 20
            or not str(value).isascii() or not str(value).isdigit()):
        raise InvalidInboundMedia("invalid_inbound_media")
    number = int(value)
    if not 0 <= number <= maximum:
        raise InvalidInboundMedia("invalid_inbound_media")
    return number


def normalize_path(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 2048 or "\\" in value
            or any(unicodedata.category(char) in {"Cc", "Cs"} for char in value)):
        raise InvalidInboundMedia("invalid_inbound_media_path")
    path = value[1:] if value.startswith("/") else value
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise InvalidInboundMedia("invalid_inbound_media_path")
    # A path is never interpreted as a URL or decoded a second time.
    if "://" in path:
        raise InvalidInboundMedia("invalid_inbound_media_path")
    return path


def normalized_etag(value):
    if not isinstance(value, str):
        raise InvalidInboundMedia("invalid_inbound_media_etag")
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    if (not 1 <= len(value) <= 256 or value.startswith("W/")
            or any(ord(char) < 33 or ord(char) > 126 or char in {'"', "\\"} for char in value)):
        raise InvalidInboundMedia("invalid_inbound_media_etag")
    return value


def _entries(message):
    parameters = message.get("messageParameters")
    if not isinstance(parameters, dict):
        raise InvalidInboundMedia("inbound_media_unavailable")
    entries = [(key, item) for key, item in parameters.items()
               if isinstance(item, dict) and item.get("type") == "file"]
    if not entries or len(entries) > MAX_INBOUND_FILES:
        raise InvalidInboundMedia("inbound_media_too_many" if entries else "inbound_media_unavailable")
    if any(not isinstance(key, str) or _KEY.fullmatch(key) is None for key, _ in entries):
        raise InvalidInboundMedia("invalid_inbound_media")
    return entries


def _file_id(value):
    number = _integer(value, 2**63 - 1)
    if number == 0:
        raise InvalidInboundMedia("invalid_inbound_media")
    return number


def parse_inbound_media(message):
    """Extract IDs/caption only; no URLs, paths, identity inference or network calls."""
    caption = message.get("message", "") if isinstance(message, dict) else ""
    if not isinstance(caption, str):
        caption = ""
    result = {"files": [], "caption": caption}
    if not isinstance(message, dict) or message.get("systemMessage") != "file_shared":
        return result
    try:
        entries = _entries(message)
        for key, _ in entries:
            caption = caption.replace("{" + key + "}", "")
        result["caption"] = caption.strip()
        if (message.get("actorType") != "users" or not isinstance(message.get("actorId"), str)
                or not message["actorId"] or message.get("messageType") != "comment"):
            raise InvalidInboundMedia("invalid_inbound_media")
        ids, total = [], 0
        for _, item in entries:
            if item.get("mimetype") not in ALLOWED_TYPES:
                raise InvalidInboundMedia("unsupported_inbound_media_type")
            size = _integer(item.get("size"), 2**63 - 1)
            if not 1 <= size <= MAX_IMAGE_BYTES:
                raise InvalidInboundMedia("inbound_media_too_large")
            total += size
            ids.append(_file_id(item.get("id")))
        if total > MAX_INCIDENT_BYTES:
            raise InvalidInboundMedia("inbound_media_too_large")
        if len(ids) != len(set(ids)):
            raise InvalidInboundMedia("invalid_inbound_media")
        result["files"] = ids
    except InvalidInboundMedia as error:
        if error.code == "inbound_media_unavailable":
            # With no file parameter this may be a localized server-generated
            # "file no longer available" notice, not the employee's caption.
            result["caption"] = ""
        result["error_code"] = error.code
    return result


def file_descriptor(message, file_id):
    parsed = parse_inbound_media(message)
    if parsed.get("error_code"):
        raise InvalidInboundMedia(parsed["error_code"])
    target = _file_id(file_id)
    matching = [item for _, item in _entries(message) if _file_id(item.get("id")) == target]
    if len(matching) != 1:
        raise InvalidInboundMedia("inbound_media_unavailable")
    item = matching[0]
    permissions = _integer(item.get("permissions"), 2**31 - 1)
    if item.get("hide-download", "no") != "no" or not permissions & 1:
        raise InvalidInboundMedia("inbound_media_download_forbidden")
    return {"id": str(target), "path": normalize_path(item.get("path")),
            "mimetype": item["mimetype"], "size": _integer(item["size"], MAX_IMAGE_BYTES),
            "etag": normalized_etag(item["etag"]) if item.get("etag") is not None else "",
            "permissions": permissions, "hide-download": "no"}


def dav_metadata(raw, expected_path, base):
    if not isinstance(raw, bytes) or len(raw) > 100_000 or re.search(br"<!\s*(?:DOCTYPE|ENTITY)", raw, re.I):
        raise InvalidInboundMedia("invalid_inbound_media_metadata")
    try:
        root = ET.fromstring(raw)
        responses = root.findall(_DAV + "response")
        if root.tag != _DAV + "multistatus" or len(responses) != 1:
            raise ValueError
        href = responses[0].findtext(_DAV + "href")
        parsed, trusted = urlsplit(href), urlsplit(base)
        if (not href or (parsed.scheme and parsed.scheme != trusted.scheme)
                or (parsed.netloc and (parsed.scheme, parsed.netloc) != (trusted.scheme, trusted.netloc))
                or parsed.query or parsed.fragment
                or unquote(parsed.path) != unquote(trusted.path.rstrip("/") + expected_path)):
            raise ValueError
        props = {}
        for part in responses[0].findall(_DAV + "propstat"):
            status = part.findtext(_DAV + "status", "")
            if len(status.split()) > 1 and status.split()[1] == "200":
                for node in part.find(_DAV + "prop"):
                    if node.tag in props:
                        raise ValueError
                    props[node.tag] = node
        if len(props[_DAV + "resourcetype"]) != 0:
            raise ValueError
        return {"id": str(_file_id(props[_OC + "fileid"].text)),
                "size": _integer(props[_DAV + "getcontentlength"].text, MAX_IMAGE_BYTES),
                "mimetype": props[_DAV + "getcontenttype"].text,
                "etag": normalized_etag(props[_DAV + "getetag"].text)}
    except (ET.ParseError, ValueError, KeyError, TypeError, AttributeError):
        raise InvalidInboundMedia("invalid_inbound_media_metadata") from None


def verify_image(raw, content_type):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_IMAGE_BYTES or content_type not in ALLOWED_TYPES:
        raise InvalidInboundMedia("invalid_inbound_media_image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if (image.format != ALLOWED_TYPES[content_type][0] or image.width * image.height > MAX_PIXELS
                        or getattr(image, "n_frames", 1) != 1):
                    raise ValueError
                image.verify()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise InvalidInboundMedia("invalid_inbound_media_image") from None
