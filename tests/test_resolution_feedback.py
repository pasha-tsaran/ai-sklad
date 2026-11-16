import pytest

from app.services.chat.policies import resolution_feedback
from tests.test_talk_dialogue import QUESTION, Knowledge, NoGeneration
from tests.test_talk_dialogue import talk as talk


@pytest.mark.parametrize("message", [
    "Да, удалось, спасибо. Все правильно",
    "Да, удалось, спасибо большое. Всё правильно!",
    "ДА, УДАЛОСЬ, СПАСИБО. ВСЁ ПРАВИЛЬНО 🙂",
    "Удалось, всё верно",
    "Ура, получилось, спасибо!",
    "Всё получилось",
    "Спасибо! Всё заработало!",
    "Да, помогло",
    "Получилось, всё работает!",
    "Проблема решена.",
    "Удалось устранить проблему",
    "Не работало раньше. Теперь всё работает.",
])
def test_explicit_complete_outcomes_are_successful(message):
    assert resolution_feedback(message) == "SUCCESS"


@pytest.mark.parametrize("message", [
    "Не помогло", "Нет, не помогло", "Не помогло, спасибо", "Не сработало",
    "Пока не работает", "Всё ещё не работает", "Ничего не изменилось", "Проблема осталась",
    "Не удалось решить проблему", "Не удалось. Всё ещё не работает",
])
def test_explicit_failed_outcomes_remain_failures(message):
    assert resolution_feedback(message) == "FAILED"


@pytest.mark.parametrize("message", [
    "Да", "Да, правильно", "Правильно понял", "Вы меня правильно поняли, спасибо",
    "Готово", "Получилось", "Да, получилось", "Удалось", "Да, удалось",
    "Удалось открыть настройки, спасибо", "Удалось увеличить громкость, ещё нужен сброс",
    "Помогло, но сканер всё ещё не читает", "Проблема решена частично",
    "Не всё заработало", "Получилось, но не полностью", "Не помогло сначала, но потом заработало",
    "Громкость увеличилась, но не удалось сбросить настройки",
    "Часть настроить удалось, но другая часть не работает",
    "Да, удалось выполнить первый шаг, но проблема осталась",
    "Всё работает, кроме печати", "Помогло, осталось настроить звук",
    "Если поможет, спасибо", "Надеюсь, всё заработало", "Не уверен, что помогло",
    "Кажется, помогло", "Скажи «помогло»", "Я написал «Помогло» в другом чате",
    "Теперь всё работает?", "Покажи как сделать, чтобы всё работало",
    "Сканер теперь сканирует",
])
def test_goal_consent_steps_partial_outcomes_and_hypotheses_are_not_overall_success(message):
    assert resolution_feedback(message) is None


def test_real_dialogue_closes_a_successful_report_without_searching_again(talk):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    assert talk.process(knowledge, model)
    talk.incoming("Да, удалось, спасибо. Все правильно", 2)
    assert talk.process(knowledge, model)
    result = talk.snapshot()
    assert result["incident"].state == "RESOLVED"
    assert len(knowledge.searches) == 1 and not model.calls
