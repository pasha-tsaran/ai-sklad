"""Scanning actions in an instruction can establish its equipment type."""

import pytest

from app.services.rag.query import choose_source, retrieval_query, source_compatible
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC


PROBLEM = "Настройка сканер ШК\nуменьшить громкость"
FILENAME = "Лист настроечных штрихкодов.docx"


@pytest.mark.parametrize("sentence", ["Отсканируйте код.", "Сканируйте код.",
                                     "Для сканирования нужен исходный код.", "Просканируйте код."])
def test_scanning_word_forms_keep_full_source_and_direct_selection_consistent(sentence):
    content = f"Для уменьшения громкости используйте штрихкод минимальной громкости. {sentence}"
    chunk = {"file_id": DOC, "source": FILENAME, "content": content}
    assert "сканер" in retrieval_query(content)
    assert choose_source(PROBLEM, [chunk]) == DOC
    assert source_compatible(PROBLEM, FILENAME + "\n" + content, complete=True)


def test_original_with_scanning_action_and_no_scanner_noun_keeps_its_photo():
    caption = "Для минимальной громкости отсканируйте верхний штрихкод."
    data = instruction([caption])
    knowledge, state = scanner_provider([(DOC, FILENAME, data)])
    approved = knowledge.search(PROBLEM)
    assert approved is not None
    assert approved.document["source"]["file_id"] == DOC
    assert approved.illustrations[1].caption == caption
    assert approved.illustrations[1].data == read_docx_images(data)[0].data
    assert not any(request.method == "POST" for request in state["calls"])


@pytest.mark.parametrize("problem,source", [
    (PROBLEM, "Принтер штрихкодов. Настройка громкости уведомлений."),
    ("Принтер: уменьшить громкость", "Лист настроечных штрихкодов. Отсканируйте код минимальной громкости."),
    ("Сканер документов: уменьшить громкость", "Лист настроечных штрихкодов. Отсканируйте код минимальной громкости."),
    ("Сканер ШК: уменьшить громкость", "Сканирование документов. Настройка громкости уведомлений."),
    (PROBLEM, "Лист настроечных штрихкодов. Отсканируйте код включения подсветки."),
    ("Сканер ШК по Wi-Fi: уменьшить громкость", "Лист настроечных штрихкодов USB. Отсканируйте код минимальной громкости."),
    ("Сканер штрихкодов DS2208: уменьшить громкость", "Лист настроечных штрихкодов DS9308. Отсканируйте код минимальной громкости."),
])
def test_scanning_word_forms_preserve_type_task_transport_and_model_checks(problem, source):
    assert not source_compatible(problem, source, complete=True)
