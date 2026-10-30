"""Read bounded original Office uploads. Never follow document links or run macros."""

import hashlib
import io
import posixpath
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from zipfile import BadZipFile, ZipFile

from PIL import Image

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


@dataclass(frozen=True)
class Illustration:
    filename: str
    caption: str
    content_type: str
    data: bytes

    def metadata(self):
        return {
            "filename": self.filename,
            "caption": self.caption,
            "content_type": self.content_type,
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }


def xml(data):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("Document entities are not supported")
    return ET.fromstring(data)


def office_zip(data):
    try:
        archive = ZipFile(io.BytesIO(data))
        members = archive.infolist()
        if len(members) > 2000 or sum(m.file_size for m in members) > 100_000_000:
            raise ValueError("Office archive too large")
        if any(m.flag_bits & 1 for m in members) or len({m.filename for m in members}) != len(members):
            raise ValueError("Unsupported Office archive")
        return archive
    except BadZipFile as exc:
        raise ValueError("Invalid Office document") from exc


def relationship_target(base, target):
    path = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join(base, target))
    if path.startswith(("../", "/")) or "\\" in path or ":" in path:
        raise ValueError("Invalid Office relationship")
    return path


def image_bytes(data, number, caption, max_bytes):
    if len(data) > max_bytes:
        raise ValueError("Illustration too large")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as picture:
            if picture.format not in {"PNG", "JPEG", "GIF"} or picture.width * picture.height > 16_000_000:
                raise ValueError("Unsupported illustration")
            picture.load()
            # Decode and re-encode: do not serve embedded metadata or active formats.
            output = io.BytesIO()
            rgba = picture.convert("RGBA")
            opaque = Image.new("RGB", picture.size, "white")
            opaque.paste(rgba, mask=rgba.getchannel("A"))
            opaque.save(output, format="PNG")
    rendered = output.getvalue()
    if len(rendered) > max_bytes:
        raise ValueError("Illustration too large")
    return Illustration(f"illustration-{number:02d}.png", caption, "image/png", rendered)


def read_docx(data, max_images=60, max_image_bytes=8_000_000):
    paragraphs, pictures = [], []
    with office_zip(data) as archive:
        relations = {
            node.attrib["Id"]: node
            for node in xml(archive.read("word/_rels/document.xml.rels"))
        }
        document = xml(archive.read("word/document.xml"))
        for paragraph in document.iter(W + "p"):
            text = "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()
            if text:
                paragraphs.append(text)
            for drawing in paragraph.iter(W + "drawing"):
                blips = list(drawing.iter(A + "blip"))
                props = next(drawing.iter(WP + "docPr"), None)
                description = props.get("descr", "").strip() if props is not None else ""
                if len(blips) != 1 or not description:
                    raise ValueError("Each illustration needs a text description in DOCX alt text")
                if len(description) > 6000:
                    raise ValueError("Illustration description too long")
                relation = relations.get(blips[0].get(R + "embed"))
                if relation is None or relation.get("TargetMode") == "External":
                    raise ValueError("External illustrations are not supported")
                path = relationship_target("word", relation.attrib["Target"])
                if not path.startswith("word/media/") or len(pictures) >= max_images:
                    raise ValueError("Invalid illustration reference or count")
                pictures.append(image_bytes(archive.read(path), len(pictures) + 1, description, max_image_bytes))
                # Keep the description usable even if an upload parser ignored it.
                if not any(description in p for p in paragraphs):
                    paragraphs.append(description)
        if list(document.iter(W + "pict")):
            raise ValueError("Legacy VML pictures must be converted to inline illustrations")
    if sum(len(p.data) for p in pictures) > 32_000_000:
        raise ValueError("Illustration set too large; split the instruction")
    return "\n\n".join(paragraphs), pictures


def read_xlsx(data):
    """Render every populated row with its column headings; no formulas are executed."""
    with office_zip(data) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(n.text or "" for n in si.iter(S + "t"))
                       for si in xml(archive.read("xl/sharedStrings.xml"))]
        relations = {n.attrib["Id"]: n for n in xml(archive.read("xl/_rels/workbook.xml.rels"))}
        output = []
        for sheet in xml(archive.read("xl/workbook.xml")).iter(S + "sheet"):
            relation = relations[sheet.attrib[R + "id"]]
            if relation.get("TargetMode") == "External":
                raise ValueError("External worksheets are not supported")
            path = relationship_target("xl", relation.attrib["Target"])
            if not path.startswith("xl/worksheets/"):
                raise ValueError("Invalid worksheet reference")
            output.append("Лист: " + sheet.attrib["name"])
            for row in xml(archive.read(path)).iter(S + "row"):
                fields = []
                for cell in row.findall(S + "c"):
                    if cell.find(S + "f") is not None:
                        raise ValueError("Use stored values, not formulas, in knowledge workbooks")
                    if cell.get("t") == "inlineStr":
                        value = "".join(t.text or "" for t in cell.iter(S + "t"))
                    else:
                        node = cell.find(S + "v")
                        value = node.text or "" if node is not None else ""
                        if cell.get("t") == "s":
                            value = strings[int(value)]
                    if value:
                        fields.append(cell.attrib["r"] + ": " + value)
                if fields:
                    output.append(" | ".join(fields))
    return "\n".join(output)
