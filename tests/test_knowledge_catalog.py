"""All original-backed sources share one persistent, versioned discovery index."""

import copy
import hashlib
import importlib.util
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile

import httpx
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, func, select

from app.db import session_factory
from app.models import Base, KnowledgeIndex, KnowledgeIndexImage
from app.services.rag.catalog import (
    PARSER_VERSION, CatalogEntry, KnowledgeCatalog, bm25_rank, build_catalog_entry,
    instruction_sections, read_docx_headings, read_docx_layout, reciprocal_rank_fusion, remote_revision,
)
from app.services.rag.knowledge_files import list_knowledge_files
from integrations.openwebui.approved_instruction_images import W, WP, read_docx_images, read_docx_text
from tests.test_openwebui_instruction_images import instruction, png
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, KB, OTHER, provider


@pytest.fixture
def catalogue(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'catalogue.db'}")
    Base.metadata.create_all(engine)
    factory = session_factory(engine)
    knowledge, state = provider(filename="Первый вход в рабочую программу.docx", data=instruction([
        "На исходном снимке показано окно первого входа в корпоративную программу.",
    ]))
    approved = knowledge.read(DOC)
    yield KnowledgeCatalog(knowledge.settings, factory), knowledge, state, approved, factory
    engine.dispose()


def test_verified_parse_snapshot_roundtrip_preserves_original_source_candidate_and_photos(catalogue):
    catalog, _, state, approved, _ = catalogue
    before = copy.deepcopy(approved)
    entry = catalog.store(approved)
    files = {DOC: {"filename": approved.document["source"]["filename"], "knowledge_id": KB}}
    restored = catalog.load_verified(DOC, state["data"], files)
    assert restored == before and approved == before
    assert restored.illustrations[1].data == approved.illustrations[1].data
    assert entry == catalog.entries(files)[DOC]
    assert entry.card["image_count"] == 1 and entry.evidence == entry.registry
    assert all(proof["text"] in " ".join(approved.document["content"].split()) for proof in entry.registry.values())


@pytest.mark.parametrize("changed", ["bytes", "filename", "collection", "removed"])
def test_cache_requires_current_bytes_and_source_membership(catalogue, changed):
    catalog, _, state, approved, _ = catalogue
    catalog.store(approved)
    files = {DOC: {"filename": approved.document["source"]["filename"], "knowledge_id": KB}}
    original = state["data"]
    if changed == "bytes":
        original += b"changed"
    elif changed == "filename":
        files[DOC]["filename"] = "Другая инструкция.docx"
    elif changed == "collection":
        files[DOC]["knowledge_id"] = OTHER
    else:
        files = {}
    assert catalog.load_verified(DOC, original, files) is None
    if changed != "bytes":
        assert catalog.entries(files) == {}
        assert catalog.reconcile(files) == 1


def test_metadata_revision_cannot_authorize_a_parse_without_matching_current_original_bytes(catalogue):
    catalog, _, state, approved, _ = catalogue
    item = {"filename": approved.document["source"]["filename"], "knowledge_id": KB,
            "hash": approved.document["source"]["sha256"], "updated_at": 1}
    catalog.store(approved, remote_revision=remote_revision(item))
    assert catalog.load_verified(DOC, state["data"] + b"new original", {DOC: item}) is None
    item["updated_at"] = 2
    assert catalog.entries({DOC: item}) == {}
    assert catalog.load_verified(DOC, state["data"], {DOC: item}) == approved


@pytest.mark.parametrize("scope_change", [{"openwebui_base_url": "https://other.example.invalid"},
                                         {"openwebui_knowledge_ids": [OTHER]}])
def test_index_is_isolated_by_server_and_allowed_collection(catalogue, scope_change):
    catalog, knowledge, state, approved, factory = catalogue
    catalog.store(approved)
    other = KnowledgeCatalog(knowledge.settings.model_copy(update=scope_change), factory)
    files = {DOC: {"filename": approved.document["source"]["filename"], "knowledge_id": KB}}
    assert other.entries(files) == {} and other.load_verified(DOC, state["data"], files) is None


def test_replacement_updates_snapshot_and_images_atomically_and_invalidates_old_bytes(catalogue):
    catalog, knowledge, state, first, factory = catalogue
    catalog.store(first)
    files = {DOC: {"filename": first.document["source"]["filename"], "knowledge_id": KB}}
    original = state["data"]
    before = catalog.revision(files)
    state["data"] = instruction(["Новое исходное окно входа в программу после обновления."], image_data=png("blue"))
    second = knowledge.read(DOC)
    catalog.store(second)
    assert catalog.load_verified(DOC, original, files) is None
    assert catalog.load_verified(DOC, state["data"], files) == second
    assert catalog.revision(files) != before
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeIndex)) == 1
        assert session.scalar(select(func.count()).select_from(KnowledgeIndexImage)) == 1
        assert session.scalar(select(KnowledgeIndexImage.data)) == second.illustrations[1].data


def test_image_corruption_and_parser_version_change_force_reparse(catalogue):
    catalog, _, state, approved, factory = catalogue
    catalog.store(approved)
    files = {DOC: {"filename": approved.document["source"]["filename"], "knowledge_id": KB}}
    with factory() as session, session.begin():
        image = session.scalar(select(KnowledgeIndexImage))
        image.data = b"corrupt bytes"
    assert catalog.load_verified(DOC, state["data"], files) is None
    catalog.store(approved)
    with factory() as session, session.begin():
        row = session.scalar(select(KnowledgeIndex))
        row.parser_version = "older-parser"
    assert catalog.load_verified(DOC, state["data"], files) is None
    assert catalog.entries(files) == {}


def test_removed_source_stays_inactive_until_a_verified_original_is_stored_again(catalogue):
    catalog, _, state, approved, _ = catalogue
    catalog.store(approved)
    files = {DOC: {"filename": approved.document["source"]["filename"], "knowledge_id": KB}}
    assert catalog.reconcile({}) == 1 and catalog.reconcile({}) == 0
    assert catalog.load_verified(DOC, state["data"], files) is None and catalog.entries(files) == {}
    catalog.store(approved)
    assert catalog.load_verified(DOC, state["data"], files) == approved


def test_concurrent_store_retries_the_first_insert_without_duplicate_sources_or_photos(catalogue):
    catalog, _, state, approved, factory = catalogue
    with ThreadPoolExecutor(max_workers=2) as executor:
        stored = list(executor.map(lambda _: catalog.store(approved), range(2)))
    assert stored[0] == stored[1]
    files = {DOC: {"filename": approved.document["source"]["filename"], "knowledge_id": KB}}
    assert catalog.load_verified(DOC, state["data"], files) == approved
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(KnowledgeIndex)) == 1
        assert session.scalar(select(func.count()).select_from(KnowledgeIndexImage)) == 1


def test_sync_discovers_new_sources_skips_unchanged_and_periodically_refreshes_changed_bytes(catalogue):
    catalog, knowledge, state, _, _ = catalogue
    first = catalog.sync(knowledge)
    assert first["complete"] and first["read_count"] == 1
    assert catalog.sync(knowledge)["read_count"] == 0
    old_revision = first["revision"]
    state["data"] = instruction(["В обновлённом оригинале показано другое исходное окно программы."])
    assert catalog.sync(knowledge)["read_count"] == 0
    with knowledge._client() as client:
        assert catalog.load_verified(DOC, state["data"], knowledge._files(client)) is None
    fresh = catalog.sync(knowledge, refresh=True)
    assert fresh["complete"] and fresh["read_count"] == 1 and fresh["revision"] != old_revision
    state["files"] = []
    empty = catalog.sync(knowledge)
    assert empty["complete"] and empty["source_count"] == empty["indexed_count"] == 0
    assert catalog.entries({}) == {}


def test_sync_rechecks_optional_remote_revision_without_using_it_as_raw_sha(catalogue):
    catalog, knowledge, _, _, _ = catalogue
    original_files = knowledge._files
    metadata = {"hash": "processed-text-hash", "updated_at": 10}
    knowledge._files = lambda client: {key: {**item, **metadata} for key, item in original_files(client).items()}
    assert catalog.sync(knowledge)["read_count"] == 1
    assert catalog.sync(knowledge)["read_count"] == 0
    metadata["updated_at"] = 11
    assert catalog.sync(knowledge)["read_count"] == 1


def test_newly_uploaded_instruction_is_indexed_by_the_same_pipeline_after_initial_sync(catalogue):
    _, _, _, _, factory = catalogue
    first = instruction(["Первая рабочая процедура для программы Вектор."])
    knowledge, state = scanner_provider([(DOC, "Первый запуск.docx", first)])
    catalog = KnowledgeCatalog(knowledge.settings, factory)
    assert catalog.sync(knowledge)["indexed_count"] == 1
    second = instruction(["Программа Меридиан. При первом входе откройте рабочую папку."])
    state["files"].append({"id": OTHER, "filename": "Новая инструкция без прежних ключевых слов.docx"})
    state["originals"][OTHER] = second
    updated = catalog.sync(knowledge)
    assert updated["complete"] and updated["indexed_count"] == 2 and updated["read_count"] == 1
    with knowledge._client() as client:
        entries = catalog.entries(knowledge._files(client))
    assert bm25_rank("Меридиан рабочая папка", entries)[0] == OTHER
    assert entries[OTHER].source["sha256"] == hashlib.sha256(second).hexdigest()


def test_unreadable_source_makes_index_incomplete_and_recovers_after_retry(catalogue):
    catalog, knowledge, state, _, _ = catalogue
    catalog.sync(knowledge)
    state["data"] = b"not a DOCX"
    failure = catalog.sync(knowledge, refresh=True)
    assert failure["status"] == "partial" and not failure["complete"] and DOC in failure["errors"]
    with knowledge._client() as client:
        assert catalog.entries(knowledge._files(client)) == {}
    state["data"] = instruction(["Исходная иллюстрация после исправления файла инструкции."])
    recovered = catalog.sync(knowledge)
    assert recovered["complete"] and recovered["read_count"] == 1 and recovered["errors"] == {}


def document(content, *, file_id=DOC, images=()):
    return {"source": {"file_id": file_id, "filename": "Будущая инструкция.md", "knowledge_id": KB,
                       "sha256": hashlib.sha256(content.encode()).hexdigest()},
            "content": content, "images": list(images)}


def test_sections_have_version_scoped_ids_original_text_images_and_mandatory_kinds():
    caption = "Отсканируйте только необходимый программирующий код на этой иллюстрации."
    content = "# Назначение\nПеред началом сохраните параметры.\nВнимание: не сканируйте все коды.\n" + caption
    original = document(content, images=[{"number": 3, "caption": caption}])
    sections = instruction_sections(original)
    assert sections == instruction_sections(copy.deepcopy(original))
    assert [section["kind"] for section in sections] == ["TEXT", "PREREQUISITE", "WARNING", "IMAGE"]
    assert sections[-1]["image_numbers"] == [3] and all(section["text"] in content for section in sections)
    assert all(section["step_ids"] and section["group_id"] for section in sections)
    assert len({section["id"] for section in sections}) == len(sections)
    assert instruction_sections(document(content + "\nНовая версия"))[0]["id"] != sections[0]["id"]
    assert instruction_sections(document(content, file_id=OTHER))[0]["id"] != sections[0]["id"]


def test_plain_source_is_one_complete_procedure_and_explicit_headings_define_groups():
    plain = instruction_sections(document("Откройте программу.\nВведите параметры.\nПодтвердите подключение."))
    assert len({section["group_id"] for section in plain}) == 1
    structured = instruction_sections(document("# Проводное подключение\nПодключите кабель.\n## Wi-Fi\nВыберите сеть."))
    assert structured[0]["group_id"] == structured[1]["group_id"]
    assert structured[2]["group_id"] == structured[3]["group_id"] != structured[0]["group_id"]


@pytest.mark.parametrize("labels", [("Шаг 1", "Шаг 2"), ("Step 1", "Step 2"),
                                   ("Этап №1", "Этап №2"), ("1. Подключение", "2. Проверка")])
@pytest.mark.parametrize("word_headings", [False, True])
def test_sequential_step_headings_keep_the_whole_procedure_and_both_original_photos(labels, word_headings):
    first_photo = "Первая исходная фотография показывает подключение кабеля."
    second_photo = "Вторая исходная фотография показывает проверку результата."
    heading = "" if word_headings else "## "
    content = ("# Рабочая процедура\n" + heading + labels[0] + "\n" + first_photo + "\n"
               + heading + labels[1] + "\n" + second_photo + "\n## Другая самостоятельная процедура\nИные действия.")
    source = document(content, images=[{"number": 1, "caption": first_photo}, {"number": 2, "caption": second_photo}])
    if word_headings:
        source["headings"] = [{"text": label, "occurrence": 1, "level": 2} for label in labels]
    sections = instruction_sections(source)
    first_step = next(section for section in sections if section["text"] == heading + labels[0])
    selected = [section for section in sections if section["group_id"] == first_step["group_id"]]
    assert heading + labels[1] in [section["text"] for section in selected]
    assert [number for section in selected for number in section["image_numbers"]] == [1, 2]
    assert "Иные действия." not in [section["text"] for section in selected]


def styled_docx(paragraphs, styles):
    original = instruction([], extra_paragraph="".join(paragraphs))
    result = io.BytesIO()
    with ZipFile(io.BytesIO(original)) as source, ZipFile(result, "w") as destination:
        for item in source.infolist():
            destination.writestr(item, source.read(item.filename))
        destination.writestr("word/styles.xml", styles)
    return result.getvalue()


def test_actual_docx_outline_styles_and_repeated_heading_text_define_safe_procedure_groups():
    styles = ('<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:style w:styleId="BaseHeading"><w:name w:val="Heading 1"/></w:style>'
              '<w:style w:styleId="Recipe"><w:basedOn w:val="BaseHeading"/></w:style>'
              '<w:style w:styleId="Substep"><w:pPr><w:outlineLvl w:val="2"/></w:pPr></w:style>'
              '</w:styles>')
    paragraphs = [
        '<w:p><w:pPr><w:pStyle w:val="Recipe"/></w:pPr><w:r><w:t>Подключение</w:t></w:r></w:p>',
        '<w:p><w:r><w:t>Подключите кабель к устройству.</w:t></w:r></w:p>',
        '<w:p><w:pPr><w:pStyle w:val="Substep"/></w:pPr><w:r><w:t>Проверка</w:t></w:r></w:p>',
        '<w:p><w:r><w:t>Проверьте проводное подключение.</w:t></w:r></w:p>',
        '<w:p><w:pPr><w:pStyle w:val="Recipe"/></w:pPr><w:r><w:t>Подключение</w:t></w:r></w:p>',
        '<w:p><w:r><w:t>Выберите беспроводную сеть.</w:t></w:r></w:p>',
        '<w:p><w:pPr><w:pStyle w:val="Recipe"/><w:outlineLvl w:val="9"/></w:pPr>'
        '<w:r><w:t>Обычный текст со сброшенным уровнем.</w:t></w:r></w:p>',
    ]
    original = styled_docx(paragraphs, styles)
    headings = read_docx_headings(original)
    assert headings == [{"text": "Подключение", "occurrence": 1, "level": 1, "paragraph": 2},
                        {"text": "Проверка", "occurrence": 1, "level": 3, "paragraph": 4},
                        {"text": "Подключение", "occurrence": 2, "level": 1, "paragraph": 6}]
    source = document(read_docx_text(original))
    source["headings"] = headings
    sections = instruction_sections(source)
    wired = next(section for section in sections if section["text"] == "Подключите кабель к устройству.")
    check = next(section for section in sections if section["text"] == "Проверьте проводное подключение.")
    wireless = next(section for section in sections if section["text"] == "Выберите беспроводную сеть.")
    assert wired["group_id"] == check["group_id"] != wireless["group_id"]
    assert sections[-1]["group_id"] == wireless["group_id"]


def test_docx_heading_inheritance_cycles_are_bounded_and_direct_outline_is_honored():
    styles = ('<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:style w:styleId="A"><w:basedOn w:val="B"/></w:style>'
              '<w:style w:styleId="B"><w:basedOn w:val="A"/></w:style></w:styles>')
    paragraphs = [
        '<w:p><w:pPr><w:pStyle w:val="A"/></w:pPr><w:r><w:t>Стиль с циклом.</w:t></w:r></w:p>',
        '<w:p><w:pPr><w:pStyle w:val="A"/><w:outlineLvl w:val="1"/></w:pPr>'
        '<w:r><w:t>Явный заголовок</w:t></w:r></w:p>',
    ]
    assert read_docx_headings(styled_docx(paragraphs, styles)) == [
        {"text": "Явный заголовок", "occurrence": 1, "level": 2, "paragraph": 3},
    ]


def recipe_docx(caption, *, alt_text=True):
    original = instruction([caption if alt_text else "", caption if alt_text else ""])
    output = io.BytesIO()
    with ZipFile(io.BytesIO(original)) as archive, ZipFile(output, "w") as result:
        root = ET.fromstring(archive.read("word/document.xml"))
        body = root.find(W + "body")
        drawings = [paragraph for paragraph in body if paragraph.find(".//" + W + "drawing") is not None]
        if alt_text:
            for properties in root.iter(WP + "docPr"):
                properties.set("descr", caption)

        def paragraph(text, heading=False):
            element = ET.Element(W + "p")
            if heading:
                properties = ET.SubElement(element, W + "pPr")
                ET.SubElement(properties, W + "pStyle", {W + "val": "Heading1"})
            ET.SubElement(ET.SubElement(element, W + "r"), W + "t").text = text
            return element

        body.clear()
        body.extend([paragraph("Первый рецепт", True), paragraph(caption), drawings[0],
                     paragraph("Второй рецепт", True), paragraph(caption), drawings[1]])
        for item in archive.infolist():
            result.writestr(item, ET.tostring(root) if item.filename == "word/document.xml" else archive.read(item.filename))
    return output.getvalue()


@pytest.mark.parametrize("alt_text", [False, True])
def test_docx_image_anchors_keep_duplicate_captions_in_their_own_recipe(alt_text):
    caption = "Одинаковая исходная подпись двух разных фотографий."
    original = recipe_docx(caption, alt_text=alt_text)
    pictures = read_docx_images(original, contextual_captions=True)
    layout = read_docx_layout(original)
    assert read_docx_text(original).count(caption) == 2
    assert layout["content"].count(caption) == (4 if alt_text else 2)
    source = document(layout["content"], images=[{"number": image.number, "caption": image.caption} for image in pictures])
    source["layout"], source["headings"] = layout["paragraphs"], read_docx_headings(original)
    sections = instruction_sections(source)
    group = next(section["group_id"] for section in sections if section["text"] == "Второй рецепт")
    selected = [section for section in sections if section["group_id"] == group]
    assert [number for section in selected for number in section["image_numbers"]] == [2]
    assert pictures[0].data != pictures[1].data
    assert all(section["text"] in layout["content"] for section in selected)


def test_positioned_multiline_caption_remains_one_section_with_one_exact_image():
    caption = "Первая строка исходной подписи.\nВторая строка программирующего листа."
    original = recipe_docx(caption)
    layout = read_docx_layout(original)
    pictures = read_docx_images(original, contextual_captions=True)
    source = document(layout["content"], images=[{"number": image.number, "caption": image.caption} for image in pictures])
    source["layout"], source["headings"] = layout["paragraphs"], read_docx_headings(original)
    sections = instruction_sections(source)
    image_sections = [section for section in sections if section["image_numbers"]]
    assert len(image_sections) == 2
    assert image_sections[0]["image_numbers"] == [1] and image_sections[1]["image_numbers"] == [2]
    assert all(section["text"] == caption and section["kind"] == "IMAGE" for section in image_sections)


def test_explicit_prerequisite_heading_keeps_its_body_and_warnings_mandatory():
    content = "Перед началом\nСохраните параметры устройства.\nВнимание: не удаляйте настройки.\nПодключение\nВыберите сеть."
    source = document(content)
    source["headings"] = [{"text": "Перед началом", "occurrence": 1, "level": 1},
                          {"text": "Подключение", "occurrence": 1, "level": 1}]
    sections = instruction_sections(source)
    assert [section["kind"] for section in sections] == [
        "PREREQUISITE", "PREREQUISITE", "WARNING", "TEXT", "TEXT",
    ]


def test_file_listing_preserves_only_bounded_revision_hints_from_approved_membership():
    item = {"id": DOC, "filename": "Произвольная будущая инструкция.docx", "hash": "processed-text-hash",
            "updated_at": 12, "etag": "etag-value", "meta": {"size": 123},
            "untrusted_extra": "must not reach catalogue"}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "items": [item], "total": 1,
    }))) as client:
        result = list_knowledge_files(client, "https://webui.example.invalid", KB)
    assert result == [{"id": DOC, "filename": item["filename"], "hash": item["hash"],
                       "updated_at": 12, "etag": "etag-value", "size": 123}]


def test_cards_contain_only_sourced_excerpts_and_include_purpose_body(catalogue):
    _, _, _, approved, _ = catalogue
    approved = copy.deepcopy(approved)
    approved.document["content"] = "Назначение\nПервый вход в программу Вектор.\nЕсли окно уже открыто, используйте сохранённое подключение."
    entry = build_catalog_entry(approved)
    assert entry.card["version"] == PARSER_VERSION
    assert any("Первый вход" in item["text"] for item in entry.card["purpose"])
    assert entry.card["applications"] and entry.card["conditions"]
    for field in ("purpose", "equipment", "applications", "conditions"):
        for item in entry.card[field]:
            assert item["text"] in entry.content and item["section_id"] in {section["id"] for section in entry.sections}


def entry(title, content):
    return CatalogEntry({}, content, {"title": title}, (), {})


def test_generic_bm25_discovers_new_unknown_products_aliases_and_device_counts():
    entries = {"first": entry("Вектор. Два устройства Wi-Fi", "Два 2 устройства подключены по Wi-Fi."),
               "second": entry("Иной вариант Bluetooth", "Один 1 прибор подключён по Bluetooth."),
               "new": entry("ПотокДело. Первая рабочая папка", "В программе ПотокДело впервые откройте рабочую папку.")}
    assert bm25_rank("ПотокДело рабочая папка", entries)[0] == "new"
    assert bm25_rank("2 устройства по ви фи", entries)[0] == "first"
    assert bm25_rank("по блютуз", entries)[0] == "second"
    assert bm25_rank("отсутствующее выражение", entries) == []
    assert len(bm25_rank("устройства", entries, limit=1)) == 1


def test_rank_fusion_deduplicates_each_ranking_and_preserves_deterministic_order():
    assert reciprocal_rank_fusion(["a", "b", "a"], ["b", "c"]) == ["b", "a", "c"]
    assert reciprocal_rank_fusion([], ["z", "a"], limit=1) == ["z"]
    with pytest.raises(ValueError, match="InvalidRankFusionConstant"):
        reciprocal_rank_fusion(["a"], k=0)


def test_migration_0010_matches_models_and_preserves_prior_tables(tmp_path):
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0010_knowledge_index.py"
    spec = importlib.util.spec_from_file_location("knowledge_index_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.revision == "0010" and migration.down_revision == "0009"
    previous = MetaData()
    for table in Base.metadata.sorted_tables:
        if table.name not in {"knowledge_index", "knowledge_index_images"}:
            table.to_metadata(previous)
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}")
    try:
        with engine.begin() as connection:
            previous.create_all(connection)
            context = MigrationContext.configure(connection)
            with Operations.context(context):
                migration.upgrade()
            assert compare_metadata(context, Base.metadata) == []
            with Operations.context(context):
                migration.downgrade()
            assert compare_metadata(context, previous) == []
    finally:
        engine.dispose()
