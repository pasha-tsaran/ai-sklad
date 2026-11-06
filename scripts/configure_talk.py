"""Configure the existing Compose .env without printing credentials or using a shell."""

import argparse
from dataclasses import dataclass, field
import getpass
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from urllib.parse import unquote, urlsplit, urlunsplit
import warnings


class SetupError(ValueError):
    """A safe, credential-free message suitable for the terminal."""


FIXED_VALUES = {
    "NEXTCLOUD_BACKEND": "account",
    "NEXTCLOUD_ALLOWED_ROOMS": "[]",
    "NEXTCLOUD_POLL_INTERVAL_SECONDS": "3",
    "NEXTCLOUD_BATCH_SIZE": "100",
}
MANAGED_KEYS = {
    *FIXED_VALUES,
    "NEXTCLOUD_BASE_URL",
    "NEXTCLOUD_USERNAME",
    "NEXTCLOUD_APP_PASSWORD",
}
PRESERVED_CRITICAL_KEYS = {
    "POSTGRES_PASSWORD", "API_TOKEN", "DATABASE_URL", "LLM_BACKEND",
    "OPENWEBUI_BASE_URL", "OPENWEBUI_API_KEY", "OPENWEBUI_MODEL",
    "KNOWLEDGE_BACKEND", "OPENWEBUI_KNOWLEDGE_IDS", "MAIL_BACKEND",
    "SMTP_PASSWORD",
}
ASSIGNMENT = re.compile(
    r"^(?:\ufeff)?[ \t]*(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?:=|:)[ \t]*(.*)$",
    re.DOTALL,
)
INHERITED = re.compile(r"^(?:\ufeff)?[ \t]*(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*$")


def single_line(value, label, *, strip=False):
    if not isinstance(value, str) or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise SetupError(f"{label}: управляющие символы и переносы строк недопустимы.")
    result = value.strip() if strip else value
    if not result or not result.strip():
        raise SetupError(f"{label}: значение не должно быть пустым.")
    return result


def normalize_url(value):
    value = single_line(value, "URL", strip=True)
    if any(character.isspace() for character in value) or "\\" in value or "?" in value or "#" in value:
        raise SetupError("URL: нужен базовый HTTPS-адрес без параметров, фрагмента и пробелов.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        raise SetupError("URL: некорректный адрес или порт.") from None
    if parsed.scheme.lower() != "https" or not hostname or parsed.username is not None or parsed.password is not None:
        raise SetupError("URL: нужен абсолютный HTTPS-адрес без встроенного логина и пароля.")
    if port is not None and not 1 <= port <= 65535:
        raise SetupError("URL: некорректный порт.")
    single_line(unquote(value), "URL")
    host = f"[{hostname.lower()}]" if ":" in hostname else hostname.lower()
    netloc = host + (f":{port}" if port is not None and port != 443 else "")
    return urlunsplit(("https", netloc, parsed.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class TalkConfiguration:
    base_url: str
    username: str
    app_password: str = field(repr=False)

    def values(self):
        username = single_line(self.username, "Логин", strip=True)
        if len(username) > 255:
            raise SetupError("Логин: допустимо не более 255 символов.")
        return {
            "NEXTCLOUD_BACKEND": "account",
            "NEXTCLOUD_BASE_URL": normalize_url(self.base_url),
            "NEXTCLOUD_USERNAME": username,
            "NEXTCLOUD_APP_PASSWORD": single_line(self.app_password, "Пароль приложения"),
            "NEXTCLOUD_ALLOWED_ROOMS": "[]",
            "NEXTCLOUD_POLL_INTERVAL_SECONDS": "3",
            "NEXTCLOUD_BATCH_SIZE": "100",
        }


def dotenv_quote(value):
    """Compose dotenv: quoted escapes plus literal dollars, never interpolation."""
    single_line(value, "Значение")
    return json.dumps(value, ensure_ascii=False).replace("$", "$$")


def _closing_quote(value):
    quote, escaped = value[0], False
    for index, character in enumerate(value[1:], 1):
        if character == quote and not escaped:
            return index
        escaped = character == "\\" and not escaped
    return None


def env_records(text):
    """Keep unrelated records verbatim, including quoted multiline values."""
    lines = iter(text.splitlines(keepends=True))
    records, critical = [], set()
    for line in lines:
        raw = line
        match = ASSIGNMENT.match(line.rstrip("\r\n"))
        if not match:
            inherited = INHERITED.match(line.rstrip("\r\n"))
            key = inherited.group(1) if inherited else None
            value = None
        else:
            key, value = match.groups()
            if value.startswith(("'", '"')):
                end = _closing_quote(value)
                while end is None:
                    try:
                        continuation = next(lines)
                    except StopIteration:
                        raise SetupError(".env содержит незавершённое значение в кавычках; файл не изменён.") from None
                    value += "\n" + continuation.rstrip("\r\n")
                    raw += continuation
                    end = _closing_quote(value)
                tail = value[end + 1:].strip()
                if tail and not tail.startswith("#"):
                    raise SetupError(".env содержит неоднозначную запись после кавычек; файл не изменён.")
        if key in PRESERVED_CRITICAL_KEYS:
            if key in critical:
                raise SetupError(".env содержит повтор критичного параметра; устраните неоднозначность вручную.")
            critical.add(key)
        records.append((key, raw, value))
    return records


def _env_path(project_root):
    root = Path(project_root)
    if not root.is_dir() or root.is_symlink():
        raise SetupError("Нужен обычный каталог проекта.")
    path = root / ".env"
    if path.is_symlink() or not path.is_file():
        raise SetupError("В корне проекта нужен существующий обычный файл .env; ссылки не поддерживаются.")
    return path


def _read_existing(project_root):
    path = _env_path(project_root)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise SetupError(".env должен быть обычным файлом.")
        data = stream.read()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SetupError(".env должен иметь кодировку UTF-8; файл не изменён.") from None
    return path, data, info, env_records(text)


def write_configuration(project_root, configuration):
    updates = configuration.values()
    path, original, initial, records = _read_existing(project_root)
    newline = "\r\n" if b"\r\n" in original else "\n"
    written, output = set(), []
    for key, raw, _ in records:
        if key not in updates:
            output.append(raw)
        elif key not in written:
            prefix = "\ufeff" if raw.startswith("\ufeff") else ""
            output.append(f"{prefix}{key}={dotenv_quote(updates[key])}{newline}")
            written.add(key)
    if output and not output[-1].endswith(("\n", "\r")):
        output.append(newline)
    output.extend(f"{key}={dotenv_quote(value)}{newline}" for key, value in updates.items() if key not in written)
    replacement = "".join(output).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".talk-config-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), 0o600)
            else:
                os.chmod(temporary, 0o600)
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        current = path.lstat()
        if (stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
                != (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
                or path.read_bytes() != original):
            raise SetupError(".env изменился во время настройки; запись отменена, повторите настройку.")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode_value(value):
    if value is None:
        raise SetupError("Параметр Talk должен иметь явно указанное значение.")
    value = value.strip()
    if value.startswith('"'):
        end = _closing_quote(value)
        try:
            decoded = json.loads(value[:end + 1])
        except (TypeError, ValueError):
            raise SetupError("Не удалось проверить формат параметра Talk; запустите настройку заново.") from None
    elif value.startswith("'"):
        end = _closing_quote(value)
        return value[1:end].replace("\\'", "'")
    else:
        decoded = value.split(" #", 1)[0].rstrip()
    # Match Compose's literal-dollar quoting, without evaluating any variables.
    without_literals = decoded.replace("$$", "")
    if "$" in without_literals:
        raise SetupError("Параметр Talk содержит подстановку переменной; запустите настройку заново.")
    return decoded.replace("$$", "$")


def check_configuration(project_root):
    _, _, info, records = _read_existing(project_root)
    if os.name == "posix" and stat.S_IMODE(info.st_mode) & 0o077:
        raise SetupError(".env должен иметь права 0600; повторите настройку или исправьте права файла.")
    values = {}
    for key, _, value in records:
        if key in MANAGED_KEYS:
            if key in values:
                raise SetupError("Есть повтор параметра Talk; запустите настройку для его устранения.")
            values[key] = _decode_value(value)
    if not MANAGED_KEYS.issubset(values):
        raise SetupError("Настройки Talk заполнены не полностью; запустите настройку без --check.")
    for key, expected in FIXED_VALUES.items():
        if values[key] != expected:
            raise SetupError("Режим Talk отличается от рекомендуемой настройки; запустите настройку заново.")
    configuration = TalkConfiguration(
        values["NEXTCLOUD_BASE_URL"], values["NEXTCLOUD_USERNAME"], values["NEXTCLOUD_APP_PASSWORD"]
    )
    configuration.values()
    return configuration


def main(argv=None, *, project_root=None):
    parser = argparse.ArgumentParser(description="Настройка учётной записи Talk без вывода паролей.")
    parser.add_argument("--check", action="store_true", help="Только локальная проверка .env, без сети и записи")
    args = parser.parse_args(argv)
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    try:
        if Path.cwd().resolve() != root.resolve():
            raise SetupError("Сначала перейдите в корень проекта: cd /opt/ai-support.")
        _read_existing(root)
        if args.check:
            check_configuration(root)
            print("Настройки Talk заполнены и имеют допустимый формат. Сеть и учётные данные не проверялись.")
            return 0
        base_url = normalize_url(input("Базовый URL Nextcloud (HTTPS): "))
        username = input("Точный логин учётной записи [SkladTech-bot]: ").strip() or "SkladTech-bot"
        single_line(username, "Логин")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            password = getpass.getpass("Пароль приложения Nextcloud (ввод скрыт): ")
        write_configuration(root, TalkConfiguration(base_url, username, password))
        print("Настройки Talk записаны в .env. Остальные параметры сохранены; пароль не выводится.")
        return 0
    except SetupError as error:
        print(f"Настройка не выполнена: {error}", file=sys.stderr)
        return 1
    except (OSError, EOFError, getpass.GetPassWarning):
        print("Настройка не выполнена. Проверьте ввод, обычный файл .env, права и запуск из корня проекта.",
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Настройка отменена; пароль не выводится.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
