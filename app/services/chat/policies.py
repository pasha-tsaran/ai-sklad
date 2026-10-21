MAX_ATTEMPTS = 5
TERMINAL_STATES = {"RESOLVED", "ESCALATED"}
FORM_ACTION = "COLLECT_ESCALATION_DETAILS"

FORM_TEXT = """Для передачи обращения заполните форму:

| Поле | Значение |
| --- | --- |
| ФИО пользователя рабочей станции | |
| Логин пользователя рабочей станции | |
| Имя компьютера | |
| Тема проблемы | |
| Дополнительное описание (необязательно) | |

После заполнения обязательных полей обращение будет передано в поддержку."""


def direct_intent(message: str) -> str | None:
    normalized = message.strip().lower().rstrip(".!?")
    if normalized in {"оператор", "специалист", "позвать специалиста"}:
        return "HUMAN"
    if normalized in {"помогло", "не помогло"}:
        return "FEEDBACK"
    return None
