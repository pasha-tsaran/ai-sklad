"""User goals and reported scope lead discovery ahead of model-generated aliases."""

import copy
from types import SimpleNamespace

import pytest

from app.services.rag.catalog import bm25_rank
from app.services.rag.query import _terms, catalog_search_terms, normalized_query
from app.services.rag.talk import TalkKnowledge


def rank(problem, goal, rows, context):
    entries = {identity: SimpleNamespace(card={"title": title}, content=content)
               for identity, title, content in rows}
    files = {identity: {"filename": f"Материал {identity}.md"} for identity, _, _ in rows}
    original = copy.deepcopy(context)
    result = TalkKnowledge._instruction_ranked_sources(problem, {"g1": goal}, entries, files, [rows[-1][0]],
                                                        context=context)
    assert context == original
    return result


def test_short_technical_terms_and_full_model_identifiers_survive_search_tokenization():
    assert normalized_query("Сканер в 1C") == normalized_query("Сканер в 1С")
    assert {"1с", "qr", "ip", "a2", "ds2208"} <= _terms("1C QR IP A2 DS2208")
    assert _terms("DS2208") != _terms("DS2209")
    assert not (_terms("да ли на из") & {"да", "ли", "на", "из"})
    assert normalized_query("артикул a1c2") == "артикул a1c2"
    assert "qr" in _terms("QR-код") and "ip" in _terms("IP-адрес")


@pytest.mark.parametrize("term,original", [("QR", "QR-код"), ("IP", "IP-адрес"), ("1C", "1С")])
def test_bm25_distinguishes_short_technical_tokens_in_opaque_new_sources(term, original):
    entries = {"other": SimpleNamespace(card={"title": "Материал А"}, content="Общие сведения об устройстве."),
               "target": SimpleNamespace(card={"title": "Материал Б"}, content=f"Описание {original} и параметров.")}
    assert bm25_rank(term, entries)[0] == "target"


@pytest.mark.parametrize("wording", ["печать бледная", "темнота печати", "насыщенность печати"])
def test_print_contrast_search_finds_english_original_without_equating_resolution(wording):
    entries = {"resolution": SimpleNamespace(card={"title": "Печать"}, content="Set printing density to 200 DPI."),
               "contrast": SimpleNamespace(card={"title": "Печать"}, content="Set print contrast for a darker label.")}
    assert bm25_rank(wording, entries)[0] == "contrast"
    assert "printcontrast" not in catalog_search_terms(entries["resolution"].content)


def test_application_named_in_the_user_goal_precedes_a_competing_generated_hardware_alias():
    goal = "Считанный сканером товарный код должен появляться в программе 1C"
    rows = [("settings", "Параметры считывателя", "Сканер штрихкодов. Свет и громкость звукового сигнала."),
            ("application", "Проверка оборудования", "В 1С откройте тест сканера и проверьте переданный код."),
            ("other", "Правила доступа", "Общие сведения о работе сотрудника.")]
    context = {"current_message": "В 1С код не появляется.",
               "goal_search_queries": {"g1": ["сканер штрихкодов свет громкость звуковой сигнал"]},
               "task_state": {"facts": [{"key": "application", "value": "1С"}]}}
    assert rank(goal, goal, rows, context)[0] == "application"


@pytest.mark.parametrize("hardware_scope", [False, True])
def test_a_completed_application_connection_does_not_override_a_new_hardware_goal(hardware_scope):
    goal = "Уменьшить громкость ручного сканера"
    rows = [("application", "Сканер в 1С", "Настройка сканера в 1С. Добавьте оборудование в рабочую программу."),
            ("volume", "Звуковой сигнал считывателя", "Для уменьшения громкости используйте код «Минимальная громкость»."),
            ("other", "Правила работы", "Общие сведения о рабочем месте.")]
    facts = [{"key": "application", "value": "1С", "entity": "scanner"},
             {"key": "completed", "value": "Сканер уже подключён к 1С", "entity": "scanner"}]
    if hardware_scope:
        facts.append({"key": "scope", "value": "hardware", "entity": "scanner"})
    context = {"current_message": "Сканер уже подключён к 1C. Хочу уменьшить громкость.",
               "goal_search_queries": {"g1": ["Настройка сканера в 1С рабочая программа"]},
               "task_state": {"facts": facts}}
    assert rank("Сканер 1С. " + goal, goal, rows, context)[0] == "volume"


def test_current_application_clarification_can_anchor_an_unfamiliar_future_product():
    goal = "Открыть рабочую папку в установленной программе"
    rows = [("old", "Справочник", "Установленная программа. Открыть рабочую папку и начать работу."),
            ("new", "Локальный клиент", "В программе ПотокДело выберите корпоративную рабочую папку."),
            ("other", "Общие правила", "Справочник оборудования.")]
    context = {"current_message": "Это программа ПотокДело.",
               "goal_search_queries": {"g1": ["Установленная программа открыть рабочую папку начать работу"]},
               "task_state": {"facts": [{"key": "application", "value": "ПотокДело", "entity": "computer"},
                                         {"key": "scope", "value": "hardware", "entity": "unrelated-scanner"}]}}
    assert rank(goal, goal, rows, context)[0] == "new"


def test_only_the_model_of_the_requested_device_leads_the_current_goal():
    goal = "Подключить принтер Nova940 по Wi-Fi"
    rows = [("older", "Предыдущая модель", "Принтер Nova930 подключается по Wi-Fi."),
            ("current", "Текущая модель", "Принтер Nova940 подключается по Wi-Fi."),
            ("other", "Справочник", "Подключение принтера по Wi-Fi.")]
    context = {"current_message": "Теперь нужен Nova940.",
               "goal_search_queries": {"g1": ["Принтер Nova930 подключается по Wi-Fi"]},
               "task_state": {"facts": [{"key": "equipment_model", "value": "Nova930", "entity": "first"},
                                         {"key": "equipment_model", "value": "Nova940", "entity": "second"}]}}
    assert rank(goal, goal, rows, context)[0] == "current"
