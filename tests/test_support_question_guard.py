"""Clarifications preserve user observations without introducing technical checks."""

import pytest

from app.services.chat.talk_dialogue import _model_clarification
from app.services.rag.talk import KnowledgeUnavailable


@pytest.mark.parametrize("question", [
    "Это ноутбук device_1 или компьютер device_2?",
    "Вы уже пробовали нажать Ctrl+Alt+End и выбрать смену пользователя?",
    "Что происходит после нажатия Win+R?",
    "Вы перезагрузили компьютер?",
    "Вы пробовали открыть панель управления?",
    "Какой адрес или имя сервера используется для подключения?",
    "Какой IP-адрес у принтера?",
    "Какой драйвер установлен?",
    "Используется ZPL или CPCL?",
    "Размер шрифта задаётся в программе конструктора. Какую программу используете?",
    "Какая модель сканера?",
    "Укажите, пожалуйста, производителя устройства?",
    "Что написано на шильдике?",
    "Какой бренд принтера?",
])
def test_new_technical_actions_parameters_and_internal_identifiers_are_rejected(question):
    with pytest.raises(KnowledgeUnavailable, match="InvalidClarification"):
        _model_clarification(question, ["На компьютере показан чужой пользователь, хочу войти под собой."])


@pytest.mark.parametrize("question,reports", [
    ("Это компьютер или телефон?", []),
    ("Что сейчас видно на экране?", []),
    ("Какое действие уже удалось выполнить?", []),
    ("Принтер сейчас включён или выключен?", []),
    ("Раньше это работало или устройство подключили впервые?", []),
    ("Какой из двух принтеров нужно подключить?", []),
    ("Какой сигнал нужен — тише или отключить его?", []),
    ("Нужно изменить громкость или сбросить настройки?", []),
    ("Какой параметр требуется изменить у модели DS2208?", ["У меня DS2208."]),
    ("Что происходит после Ctrl+Alt+End?", ["Я нажал Ctrl + Alt + End, появилось другое окно."]),
    ("Какой текст показан на экране device_1?", ["На корпусе компьютера написано device_1."]),
    ("В каком окне появляется «Неверный пароль»?", ["Пишет: Неверный пароль."]),
    ("Понял. Что сейчас видно на экране?", []),
])
def test_observable_questions_and_actual_employee_asset_names_remain_available(question, reports):
    assert _model_clarification(question, reports) == question
