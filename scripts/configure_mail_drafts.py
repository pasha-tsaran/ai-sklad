"""Enable local email drafts in the existing Compose .env, without SMTP or secret output."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.mail.draft import validate_recipient  # noqa: E402
from scripts import configure_talk as talk  # noqa: E402

MANAGED_KEYS = {"MAIL_BACKEND", "SUPPORT_EMAIL"}


@dataclass(frozen=True)
class MailDraftConfiguration:
    recipient: str

    def values(self):
        try:
            recipient = validate_recipient(self.recipient)
        except ValueError:
            raise talk.SetupError("Нужен один корректный адрес технической поддержки без имени и дополнительных получателей.") from None
        if recipient.casefold().endswith(".invalid"):
            raise talk.SetupError("Укажите настоящий адрес технической поддержки вместо тестового адреса.")
        return {"MAIL_BACKEND": "mailto", "SUPPORT_EMAIL": recipient}


def check_configuration(project_root):
    """Read only the two managed values; never show unrelated configuration or secrets."""
    _, _, info, records = talk._read_existing(project_root, replaced_keys=MANAGED_KEYS)
    if os.name == "posix" and stat.S_IMODE(info.st_mode) & 0o077:
        raise talk.SetupError(".env должен иметь права 0600; повторите настройку или исправьте права файла.")
    values = {}
    for key, _, value in records:
        if key in MANAGED_KEYS:
            if key in values:
                raise talk.SetupError("В .env повторяются параметры черновиков; запустите настройку без --check.")
            values[key] = talk._decode_value(value)
    if set(values) != MANAGED_KEYS or values["MAIL_BACKEND"] != "mailto":
        raise talk.SetupError("Режим черновиков не настроен; запустите настройку без --check.")
    configuration = MailDraftConfiguration(values["SUPPORT_EMAIL"])
    configuration.values()
    return configuration


def main(argv=None, *, project_root=None):
    parser = argparse.ArgumentParser(description="Настройка черновиков писем без SMTP и автоматической отправки.")
    parser.add_argument("--check", action="store_true", help="Только локальная проверка двух параметров .env")
    args = parser.parse_args(argv)
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    try:
        if Path.cwd().resolve() != root.resolve():
            raise talk.SetupError("Сначала перейдите в корень проекта: cd /opt/ai-support.")
        talk._read_existing(root, replaced_keys=MANAGED_KEYS)
        if args.check:
            check_configuration(root)
            print("MAIL_BACKEND=mailto. Получатель указан корректно. Выполнена только локальная проверка; письма не отправлялись.")
            return 0
        recipient = input("Адрес технической поддержки, который будет получателем письма: ").strip()
        talk.write_configuration(root, MailDraftConfiguration(recipient))
        print("Режим черновиков записан: MAIL_BACKEND=mailto. Остальные настройки сохранены. Письма автоматически не отправляются.")
        return 0
    except talk.SetupError as error:
        print(f"Настройка не выполнена: {error}", file=sys.stderr)
        return 1
    except (OSError, EOFError):
        print("Настройка не выполнена. Проверьте ввод, обычный файл .env, права и запуск из корня проекта.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Настройка отменена; файл не изменён.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
