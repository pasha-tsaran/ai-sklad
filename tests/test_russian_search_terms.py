"""Known Russian product aliases affect search, never the employee's raw text."""

from types import SimpleNamespace

import pytest

from app.services.rag.query import normalized_query, retrieval_query, source_compatible
from app.services.rag.success import cache_key


@pytest.mark.parametrize("alias,canonical", [
    ("ВИ ФИ", "wifi"), ("вифи", "wifi"), ("вай фай", "wifi"),
    ("вай-фай", "wifi"), ("вайфай", "wifi"), ("ВИ‑ФИ", "wifi"),
    ("WI FI", "wifi"), ("Wi-Fi", "wifi"),
    ("блютуз", "bluetooth"), ("блютус", "bluetooth"), ("блутуз", "bluetooth"),
    ("блю тус", "bluetooth"), ("блу тус", "bluetooth"), ("блютузу", "bluetooth"),
    ("блютузом", "bluetooth"), ("Blue Tooth", "bluetooth"),
    ("фортигейт", "fortigate"), ("форти гейт", "fortigate"),
    ("фортегейт", "fortigate"), ("фортигейте", "fortigate"),
    ("фортигейтом", "fortigate"), ("Forti Gate", "fortigate"),
    ("фортиклиент", "forticlient"), ("форти клиенте", "forticlient"),
    ("Forti Client", "forticlient"),
    ("некст клауд толк", "nextcloud talk"), ("некстклаудтолк", "nextcloud talk"),
    ("некстклауд талк", "nextcloud talk"), ("некст-клауд толке", "nextcloud talk"),
    ("Next Cloud Talk", "nextcloud talk"), ("некст клауд", "nextcloud"),
])
def test_known_russian_spellings_match_the_same_product(alias, canonical):
    original = f"Не работает {alias}, модель ZD420, версия 1.0.53"
    saved = original
    assert normalized_query(original) == normalized_query(
        f"Не работает {canonical}, модель ZD420, версия 1.0.53"
    )
    assert original == saved
    assert canonical in retrieval_query(original)
    assert "не" in normalized_query(original).split()
    assert "zd420" in normalized_query(original) and "1.0.53" in normalized_query(original)


@pytest.mark.parametrize("original,expected", [
    ("подключить поблютуз", "подключить по bluetooth"),
    ("подключить повайфай", "подключить по wifi"),
    ("подключить черезблютуз", "подключить через bluetooth"),
])
def test_missing_space_after_a_transport_preposition_is_understood(original, expected):
    assert normalized_query(original) == expected


@pytest.mark.parametrize("left,right", [
    ("фортигейт", "фортиклиент"), ("некстклауд", "некстклауд толк"),
    ("по ви фи", "по блютузу"), ("USB", "Ethernet"),
    ("блютуз не работает", "блютуз работает"),
    ("фортигейт 100F", "фортигейт 200F"),
    ("ZD420 1.0.53", "ZD410 1.0.52"),
])
def test_distinct_products_transports_models_and_negations_stay_distinct(left, right):
    assert normalized_query(left) != normalized_query(right)


def test_ordinary_russian_words_and_unknown_products_are_not_transliterated():
    assert normalized_query("От этого нет толка. Форти пока не трогаем.") == "от этого нет толка форти пока не трогаем"
    assert normalized_query("Блок питания, клиент и облако") == "блок питания клиент и облако"
    assert normalized_query("НезнакомаяСистема X100") == "незнакомаясистема x100"


def test_cache_fingerprints_equivalent_known_aliases_without_merging_other_context():
    settings = SimpleNamespace(openwebui_base_url="https://webui.example.invalid",
                               openwebui_knowledge_ids=["85cda3aa-b854-4c26-9948-12da742277ef"])
    context = {"incident": {}}
    a = cache_key(settings, "Не работает некст клауд толк по ВИ ФИ на ZD420", context)
    b = cache_key(settings, "Не работает Nextcloud Talk по Wi-Fi на ZD420", context)
    assert a == b
    assert a != cache_key(settings, "Работает Nextcloud Talk по Wi-Fi на ZD420", context)
    assert a != cache_key(settings, "Не работает Nextcloud Talk по Bluetooth на ZD420", context)


def test_russian_connection_words_still_enforce_the_actual_transport():
    problem = "Подключить принтер к ТСД по ВИ ФИ"
    assert not source_compatible(problem, "Подключите принтер к ТСД по блютузу.", complete=True)
    assert source_compatible(problem, "Подключите принтер к ТСД по Wi-Fi.", complete=True)


def test_similar_vendor_names_cannot_authorize_another_product():
    assert not source_compatible("Не работает фортигейт", "Настройте FortiClient на компьютере.", complete=True)
    assert not source_compatible("Не работает фортиклиент", "Настройте FortiGate.", complete=True)
    assert source_compatible("Не работает форти гейт", "Диагностика FortiGate.", complete=True)
    assert not source_compatible("Не работает некст клауд толк", "Использование Nextcloud Files.", complete=True)


@pytest.mark.parametrize("message", ["Фортигейт не работает", "Форти клиент не пускает",
                                     "Некст клауд толк не открывается", "Нужен ВИ ФИ", "По блютузу"])
def test_russian_technical_names_are_routed_as_support(message):
    from app.services.chat.talk_dialogue import _technical_message

    assert _technical_message(message)
