import os
from pathlib import Path
import stat

import pytest

from scripts import configure_talk as setup


BASE_ENV = (
    "# Existing deployment; values here are fake test fixtures.\n"
    "POSTGRES_PASSWORD=fixture-db\n"
    "API_TOKEN=fixture-api\n"
    "OPENWEBUI_BASE_URL=https://ai.example.invalid\n"
    "OPENWEBUI_API_KEY=fixture-llm\n"
    "OPENWEBUI_MODEL=ai-support\n"
    "MAIL_BACKEND=mock\n"
    "SMTP_PASSWORD='fixture-mail'\n"
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / ".env").write_bytes(BASE_ENV.encode())
    return tmp_path


def config(password="fixture-app-password"):
    return setup.TalkConfiguration("https://cloud.example.invalid/nextcloud/", "SkladTech-bot", password)


def test_update_preserves_existing_settings_comments_multiline_and_removes_only_owned_duplicates(project):
    path = project / ".env"
    preserved = BASE_ENV + "CUSTOM='line one\nNEXTCLOUD_USERNAME=inside-unrelated-value\nline three'\n"
    path.write_bytes((preserved + "NEXTCLOUD_BACKEND=mock\nexport NEXTCLOUD_BACKEND=talk\n"
                      "NEXTCLOUD_APP_PASSWORD=old-fixture\nNEXTCLOUD_APP_PASSWORD=second-fixture\n").encode())
    setup.write_configuration(project, config())
    actual = path.read_text()
    assert actual.startswith(preserved)
    assert actual.count('NEXTCLOUD_BACKEND="account"') == 1
    assert actual.count('NEXTCLOUD_APP_PASSWORD="fixture-app-password"') == 1
    assert "old-fixture" not in actual and "second-fixture" not in actual
    checked = setup.check_configuration(project)
    assert checked.base_url == "https://cloud.example.invalid/nextcloud"
    assert checked.username == "SkladTech-bot"
    assert sorted(item.name for item in project.iterdir()) == [".env"]


def test_password_special_characters_are_literal_and_never_evaluated(project, monkeypatch):
    monkeypatch.setenv("DANGEROUS", "must-not-appear")
    password = 'fake-${DANGEROUS}-$USER-"quote"-\\slash-#-\'single'
    setup.write_configuration(project, config(password))
    actual = (project / ".env").read_text()
    assert "$${DANGEROUS}" in actual and "$$USER" in actual
    assert "must-not-appear" not in actual
    assert '\\"quote\\"' in actual and "\\\\slash" in actual
    assert setup.check_configuration(project).app_password == password
    assert password not in repr(config(password))


def test_existing_crlf_records_survive_without_reformatting(project):
    path = project / ".env"
    preserved = BASE_ENV.replace("\n", "\r\n").encode()
    path.write_bytes(preserved + b"CUSTOM=no-final-newline")
    setup.write_configuration(project, config())
    actual = path.read_bytes()
    assert actual.startswith(preserved + b"CUSTOM=no-final-newline\r\n")
    assert b'NEXTCLOUD_ALLOWED_ROOMS="[]"\r\n' in actual


@pytest.mark.parametrize("url", [
    "http://cloud.example.invalid", "https://user:fixture@cloud.example.invalid", "https:///no-host",
    "https://cloud.example.invalid?x=1", "https://cloud.example.invalid/#fragment",
    "https://cloud.example.invalid:65536", "https://cloud.example.invalid/\r\nKEY=bad",
    "https://cloud.example.invalid/%0d%0a", "https://cloud.example.invalid\\bad", "https://cloud example.invalid",
])
def test_invalid_url_does_not_change_file(project, url):
    before = (project / ".env").read_bytes()
    with pytest.raises(setup.SetupError):
        setup.write_configuration(project, setup.TalkConfiguration(url, "SkladTech-bot", "fixture"))
    assert (project / ".env").read_bytes() == before


@pytest.mark.parametrize("username,password", [
    ("bad\nAPI_TOKEN=bad", "fixture"), ("user", "bad\r\nNEXTCLOUD_BACKEND=mock"),
    ("", "fixture"), ("user", ""), ("user", "bad\x00value"), ("u" * 256, "fixture"),
])
def test_invalid_credentials_cannot_inject_env_lines(project, username, password):
    before = (project / ".env").read_bytes()
    with pytest.raises(setup.SetupError):
        setup.write_configuration(project, setup.TalkConfiguration("https://cloud.example.invalid", username, password))
    assert (project / ".env").read_bytes() == before


def test_preserved_critical_duplicate_is_rejected_without_revealing_value(project):
    path = project / ".env"
    path.write_text(BASE_ENV + "API_TOKEN=second-hidden-fixture\n")
    before = path.read_bytes()
    with pytest.raises(setup.SetupError) as error:
        setup.write_configuration(project, config())
    assert "second-hidden-fixture" not in str(error.value)
    assert path.read_bytes() == before


def test_unterminated_unrelated_multiline_value_is_not_rewritten(project):
    path = project / ".env"
    path.write_text(BASE_ENV + "CUSTOM='unterminated\nNEXTCLOUD_BACKEND=mock\n")
    before = path.read_bytes()
    with pytest.raises(setup.SetupError):
        setup.write_configuration(project, config())
    assert path.read_bytes() == before


def test_atomic_replace_failure_keeps_original_and_removes_private_temporary(project, monkeypatch):
    original = (project / ".env").read_bytes()

    def fail_replace(source, destination):
        assert Path(destination) == project / ".env"
        if os.name == "posix":
            assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
        raise OSError("fixture replacement failure")

    monkeypatch.setattr(setup.os, "replace", fail_replace)
    with pytest.raises(OSError):
        setup.write_configuration(project, config())
    assert (project / ".env").read_bytes() == original
    assert sorted(item.name for item in project.iterdir()) == [".env"]


def test_concurrent_edit_is_preserved_and_not_overwritten(project, monkeypatch):
    original_mkstemp = setup.tempfile.mkstemp
    changed = BASE_ENV + "# edited concurrently\n"

    def edit_before_replacement(*args, **kwargs):
        (project / ".env").write_text(changed)
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(setup.tempfile, "mkstemp", edit_before_replacement)
    with pytest.raises(setup.SetupError):
        setup.write_configuration(project, config())
    assert (project / ".env").read_text() == changed
    assert sorted(item.name for item in project.iterdir()) == [".env"]


@pytest.mark.skipif(os.name != "posix", reason="Linux filesystem permissions and symlinks")
def test_posix_private_mode_and_symlink_refusal(project, tmp_path):
    setup.write_configuration(project, config())
    assert stat.S_IMODE((project / ".env").stat().st_mode) == 0o600
    target = project / "other-config"
    target.write_text("leave unchanged")
    (project / ".env").unlink()
    (project / ".env").symlink_to(target)
    with pytest.raises(setup.SetupError):
        setup.write_configuration(project, config())
    assert target.read_text() == "leave unchanged"


def test_missing_env_and_wrong_working_directory_are_rejected(project, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(project.parent)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not prompt from wrong directory"))
    assert setup.main([], project_root=project) == 1
    assert "cd /opt/ai-support" in capsys.readouterr().err
    (project / ".env").unlink()
    monkeypatch.chdir(project)
    assert setup.main(["--check"], project_root=project) == 1


def test_interactive_defaults_and_check_have_no_credential_output(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    answers = iter(["HTTPS://Cloud.Example.Invalid:443/nextcloud/", ""])
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return next(answers)

    def hidden_password(prompt):
        prompts.append(prompt)
        return "unique-hidden-fixture"

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(setup.getpass, "getpass", hidden_password)
    assert setup.main([], project_root=project) == 0
    assert len(prompts) == 3
    checked = setup.check_configuration(project)
    assert checked.username == "SkladTech-bot"
    assert checked.base_url == "https://cloud.example.invalid/nextcloud"
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("check must not prompt"))
    monkeypatch.setattr(setup.getpass, "getpass", lambda _: pytest.fail("check must not prompt"))
    before = (project / ".env").read_bytes()
    assert setup.main(["--check"], project_root=project) == 0
    output = capsys.readouterr()
    assert "unique-hidden-fixture" not in output.out + output.err
    assert "fixture-api" not in output.out + output.err
    assert (project / ".env").read_bytes() == before


def test_unavailable_masked_password_input_aborts_before_writing(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    answers = iter(["https://cloud.example.invalid", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    def cannot_hide(prompt):
        raise setup.getpass.GetPassWarning("fixture terminal cannot hide input")

    monkeypatch.setattr(setup.getpass, "getpass", cannot_hide)
    before = (project / ".env").read_bytes()
    assert setup.main([], project_root=project) == 1
    assert (project / ".env").read_bytes() == before
    assert "fixture terminal" not in capsys.readouterr().err


@pytest.mark.parametrize("replacement", [
    'NEXTCLOUD_ALLOWED_ROOMS="[\\"group-room\\"]"',
    'NEXTCLOUD_APP_PASSWORD="${MISSING_PASSWORD}"',
    'NEXTCLOUD_USERNAME=""',
])
def test_check_rejects_unsafe_or_incomplete_account_shape(project, replacement):
    setup.write_configuration(project, config())
    key = replacement.partition("=")[0]
    path = project / ".env"
    path.write_text("\n".join(replacement if line.startswith(key + "=") else line
                              for line in path.read_text().splitlines()) + "\n")
    with pytest.raises(setup.SetupError):
        setup.check_configuration(project)
