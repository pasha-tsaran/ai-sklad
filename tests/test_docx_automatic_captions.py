"""New ordinary DOCX uploads use original nearby text as photo context."""

import io
from zipfile import ZipFile

import pytest
from PIL import Image

from integrations.openwebui.approved_instruction_images import ToolError, read_docx_images, read_docx_text
from tests.test_openwebui_instruction_images import instruction, office, png


def without_alternative_text(data, *, context=""):
    with ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    xml = parts["word/document.xml"].decode()
    xml = xml.replace('descr="Нажмите кнопку сохранения."', 'descr=""')
    xml = xml.replace("<w:body>", "<w:body>" + context)
    parts["word/document.xml"] = xml
    return office(parts)


def test_plain_uploaded_docx_uses_its_original_preceding_paragraph_and_exact_photo():
    data = without_alternative_text(instruction(["Нажмите кнопку сохранения."], image_data=png("green")))
    with pytest.raises(ToolError):
        read_docx_images(data)
    image = read_docx_images(data, contextual_captions=True)[0]
    assert image.caption == "Утверждённая инструкция Acrobat"
    assert image.caption in read_docx_text(data)
    with Image.open(io.BytesIO(image.data)) as decoded:
        assert decoded.getpixel((0, 0)) == (0, 128, 0)


def test_explicit_author_description_stays_the_photo_caption():
    data = instruction(["Нажмите кнопку сохранения."])
    assert read_docx_images(data, contextual_captions=True)[0].caption == "Нажмите кнопку сохранения."


def test_automatic_caption_never_follows_an_external_photo_relationship():
    data = without_alternative_text(instruction(["Нажмите кнопку сохранения."], external=True))
    with pytest.raises(ToolError) as error:
        read_docx_images(data, contextual_captions=True)
    assert error.value.code == "external_image"


def test_a_photo_inside_the_action_paragraph_uses_that_original_action():
    data = without_alternative_text(instruction(["Нажмите кнопку сохранения."]))
    with ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    xml = parts["word/document.xml"].decode().replace(
        "<w:p><w:r><w:drawing>", "<w:p><w:r><w:t>Откройте настройки рабочей папки.</w:t></w:r><w:r><w:drawing>",
    )
    parts["word/document.xml"] = xml
    image = read_docx_images(office(parts), contextual_captions=True)[0]
    assert image.caption == "Откройте настройки рабочей папки."
