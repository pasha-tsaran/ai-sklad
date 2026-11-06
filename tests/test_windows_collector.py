"""Parse the deployment package and reject hostile launch arguments without installation."""

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOST = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
pytestmark = pytest.mark.skipif(os.name != "nt" or not HOST.is_file(), reason="Windows PowerShell 5.1 parser required")


@pytest.mark.parametrize("name", ["Collect-Workstation.ps1", "Install-WorkstationCollector.ps1", "Uninstall-WorkstationCollector.ps1"])
def test_windows_package_parses_in_the_actual_builtin_powershell_host(name):
    path = str(ROOT / "integrations/windows" / name).replace("'", "''")
    command = ("$errorsFound=$null; $tokensFound=$null; "
               + "[void][System.Management.Automation.Language.Parser]::ParseFile('" + path
               + "',[ref]$tokensFound,[ref]$errorsFound); if($errorsFound.Count){exit 1}; exit 0")
    result = subprocess.run([str(HOST), "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


@pytest.mark.parametrize("suffix", ["?target=https://example.invalid", "/../other", '" -Command exit 77 #'])
def test_uri_cannot_override_endpoint_or_turn_a_protocol_argument_into_a_command(suffix):
    script = ROOT / "integrations/windows/Collect-Workstation.ps1"
    uri = "ai-support://collect/" + "A" * 43 + suffix
    result = subprocess.run([str(HOST), "-NoProfile", "-NonInteractive", "-File", str(script), "-RequestUri", uri],
                            capture_output=True, timeout=15)
    if b"PSSecurityException" in result.stderr or b"UnauthorizedAccess" in result.stderr:
        pytest.skip("Corporate execution policy requires signing before functional script checks")
    assert result.returncode == 2, result.stderr.decode(errors="replace")
    assert result.stdout == b"" and result.stderr == b""
