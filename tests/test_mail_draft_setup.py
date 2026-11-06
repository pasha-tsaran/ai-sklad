"""The draft wizard changes only its two owned settings and never outputs credentials."""

import os
import stat

import pytest

from scripts import configure_mail_drafts as setup

BASE = (
    "# Existing fake deployment settings; preserve bytes and quoting.\r\n"
    "POSTGRES_PASSWORD='db-fixture-$literal'\r\n"
    "API_TOKEN=api-fixture-hidden\r\n"
    'OPENWEBUI_API_KEY="webui-fixture-hidden"\r\n'
    "OPENWEBUI_BASE_URL=https://ai.collant.ru\r\n"
    "OPENWEBUI_MODEL=ai-support\r\n"
    'OPENWEBUI_KNOWLEDGE_IDS=["85cda3aa-b854-4c26-9948-12da742277ef"]\r\n'
    "NEXTCLOUD_BACKEND=account\r\n"
    "NEXTCLOUD_USERNAME=SkladTech-bot\r\n"
    "NEXTCLOUD_APP_PASSWORD='cloud-fixture-hidden'\r\n"
    "SMTP_PASSWORD='old-smtp-fixture-hidden'\r\n"
    "CUSTOM='two lines\r\nMAIL_BACKEND=inside-unrelated-value'\r\n"
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / ".env").write_bytes((BASE + "MAIL_BACKEND=mock\r\nSUPPORT_EMAIL=support@example.invalid\r\n").encode())
    return tmp_path


def test_only_managed_mail_keys_change_and_their_duplicates_are_removed(project):
    path = project / ".env"
    path.write_bytes((BASE + "MAIL_BACKEND=mock\r\nexport MAIL_BACKEND=smtp\r\n"
                      "SUPPORT_EMAIL=old@example.invalid\r\nSUPPORT_EMAIL=second@example.invalid\r\n").encode())
    setup.talk.write_configuration(project, setup.MailDraftConfiguration("tp@collant.ru"))
    content = path.read_bytes()
    assert content.startswith(BASE.encode())
    _, _, _, records = setup.talk._read_existing(project, replaced_keys=setup.MANAGED_KEYS)
    managed = [(key, setup.talk._decode_value(value)) for key, _, value in records if key in setup.MANAGED_KEYS]
    assert managed == [("MAIL_BACKEND", "mailto"), ("SUPPORT_EMAIL", "tp@collant.ru")]
    assert setup.check_configuration(project).recipient == "tp@collant.ru"
    assert sorted(item.name for item in project.iterdir()) == [".env"]


def test_dollar_mailbox_is_literal_compose_value_not_environment_interpolation(project, monkeypatch):
    monkeypatch.setenv("desk", "must-not-be-inserted")
    recipient = "support$desk@collant.ru"
    setup.talk.write_configuration(project, setup.MailDraftConfiguration(recipient))
    content = (project / ".env").read_bytes()
    assert b'SUPPORT_EMAIL="support$$desk@collant.ru"' in content
    assert b"must-not-be-inserted" not in content
    assert setup.check_configuration(project).recipient == recipient


@pytest.mark.parametrize("recipient", ["", "Support <tp@collant.ru>", "tp@collant.ru,bcc@example.com",
                                      "tp@collant.ru\r\nAPI_TOKEN=changed", "tp@example.INVALID",
                                      "tp@collant.ru\x00", "user@local"])
def test_invalid_recipient_cannot_modify_file_or_inject_settings(project, recipient):
    before = (project / ".env").read_bytes()
    with pytest.raises(setup.talk.SetupError):
        setup.talk.write_configuration(project, setup.MailDraftConfiguration(recipient))
    assert (project / ".env").read_bytes() == before


def test_unrelated_critical_duplicates_are_not_rewritten_or_disclosed(project):
    path = project / ".env"
    path.write_bytes(path.read_bytes() + b"API_TOKEN=other-hidden-fixture\r\n")
    before = path.read_bytes()
    with pytest.raises(setup.talk.SetupError) as error:
        setup.talk.write_configuration(project, setup.MailDraftConfiguration("tp@collant.ru"))
    assert "hidden-fixture" not in str(error.value) and path.read_bytes() == before


def test_interactive_setup_and_check_never_print_existing_credentials_or_send_mail(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return "tp@collant.ru"

    monkeypatch.setattr("builtins.input", answer)
    assert setup.main([], project_root=project) == 0 and len(prompts) == 1
    before = (project / ".env").read_bytes()
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Local check must not prompt"))
    assert setup.main(["--check"], project_root=project) == 0
    output = capsys.readouterr()
    for fixture in ("db-fixture", "api-fixture", "webui-fixture", "cloud-fixture", "smtp-fixture"):
        assert fixture not in output.out + output.err
    assert "MAIL_BACKEND=mailto" in output.out and "не отправлялись" in output.out
    assert (project / ".env").read_bytes() == before


def test_check_rejects_mock_mode_and_duplicate_owned_keys_without_writing(project):
    path = project / ".env"
    before = path.read_bytes()
    with pytest.raises(setup.talk.SetupError, match="не настроен"):
        setup.check_configuration(project)
    assert path.read_bytes() == before
    setup.talk.write_configuration(project, setup.MailDraftConfiguration("tp@collant.ru"))
    path.write_bytes(path.read_bytes() + b"MAIL_BACKEND=mock\r\n")
    with pytest.raises(setup.talk.SetupError, match="повторяются"):
        setup.check_configuration(project)


def test_check_rejects_variable_recipient_without_expanding_it(project, monkeypatch):
    setup.talk.write_configuration(project, setup.MailDraftConfiguration("tp@collant.ru"))
    path = project / ".env"
    path.write_bytes(path.read_bytes().replace(b'SUPPORT_EMAIL="tp@collant.ru"', b'SUPPORT_EMAIL="${PRIVATE_EMAIL}"'))
    monkeypatch.setenv("PRIVATE_EMAIL", "do-not-print@example.com")
    with pytest.raises(setup.talk.SetupError) as error:
        setup.check_configuration(project)
    assert "do-not-print" not in str(error.value)


def test_wrong_directory_and_missing_env_are_rejected_before_prompt(project, monkeypatch, capsys):
    monkeypatch.chdir(project.parent)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Do not prompt here"))
    assert setup.main([], project_root=project) == 1
    assert "cd /opt/ai-support" in capsys.readouterr().err
    monkeypatch.chdir(project)
    (project / ".env").unlink()
    assert setup.main([], project_root=project) == 1


def test_write_failure_is_safe_and_does_not_output_exception_contents(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    monkeypatch.setattr("builtins.input", lambda _: "tp@collant.ru")
    before = (project / ".env").read_bytes()

    def fail_replace(source, target):
        raise OSError("secret-upstream-fixture")

    monkeypatch.setattr(setup.talk.os, "replace", fail_replace)
    assert setup.main([], project_root=project) == 1
    assert "secret-upstream-fixture" not in capsys.readouterr().err
    assert (project / ".env").read_bytes() == before
    assert sorted(item.name for item in project.iterdir()) == [".env"]


@pytest.mark.skipif(os.name != "posix", reason="Linux .env file permissions")
def test_setup_writes_0600_and_check_refuses_other_user_access(project):
    setup.talk.write_configuration(project, setup.MailDraftConfiguration("tp@collant.ru"))
    assert stat.S_IMODE((project / ".env").stat().st_mode) == 0o600
    (project / ".env").chmod(0o644)
    with pytest.raises(setup.talk.SetupError, match="0600"):
        setup.check_configuration(project)
