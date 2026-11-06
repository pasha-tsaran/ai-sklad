#requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium', PositionalBinding = $false)]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$installOwner = 'AI-Support Workstation Collector'
$configPath = 'SOFTWARE\AI-Support\WorkstationCollector'
$protocolPath = 'SOFTWARE\Classes\ai-support'

function Assert-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    try {
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw 'Run uninstallation as an administrator or a computer startup GPO.'
        }
    } finally { $identity.Dispose() }
}

function Assert-PlainPath([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        if (((Get-Item -LiteralPath $Path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Refusing to remove a symbolic link or junction.'
        }
    }
}

if ([Environment]::Is64BitOperatingSystem -and -not [Environment]::Is64BitProcess) {
    throw 'Use the 64-bit Windows PowerShell host on this computer.'
}
$programFiles = [IO.Path]::GetFullPath([Environment]::GetFolderPath('ProgramFiles'))
$parentPath = Join-Path $programFiles 'AI-Support'
$installPath = [IO.Path]::GetFullPath((Join-Path $parentPath 'WorkstationCollector'))
$expectedPath = $programFiles.TrimEnd('\') + '\AI-Support\WorkstationCollector'
if (-not [string]::Equals($installPath, $expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Unexpected installation directory.'
}
$destination = Join-Path $installPath 'Collect-Workstation.ps1'
foreach ($path in @($programFiles, $parentPath, $installPath, $destination)) { Assert-PlainPath $path }
$view = [Microsoft.Win32.RegistryView]::Registry32
if ([Environment]::Is64BitOperatingSystem) { $view = [Microsoft.Win32.RegistryView]::Registry64 }
$machine = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, $view)
try {
    $ownedConfig = $false
    foreach ($path in @($configPath, $protocolPath)) {
        $existing = $machine.OpenSubKey($path, $false)
        if ($null -ne $existing) {
            try {
                if ($existing.GetValue('InstallOwner') -cne $installOwner) {
                    throw 'Refusing to remove configuration or a protocol owned by another package.'
                }
                if ($path -eq $configPath) { $ownedConfig = $true }
            } finally { $existing.Dispose() }
        }
    }
    if ((Test-Path -LiteralPath $destination) -and -not $ownedConfig) {
        throw 'The file has no matching installation ownership marker; remove it only after manual review.'
    }
    if ($PSCmdlet.ShouldProcess($installPath, 'Remove only the owned AI Support collector, settings and URI handler')) {
        Assert-Admin
        foreach ($path in @($configPath, $protocolPath)) { $machine.DeleteSubKeyTree($path, $false) }
        # No recursive filesystem deletion: preserve unrelated files and parent keys.
        if (Test-Path -LiteralPath $destination -PathType Leaf) { Remove-Item -LiteralPath $destination -Force }
        if ((Test-Path -LiteralPath $installPath -PathType Container) -and
            @(Get-ChildItem -LiteralPath $installPath -Force).Count -eq 0) {
            Remove-Item -LiteralPath $installPath
        }
        if ((Test-Path -LiteralPath $parentPath -PathType Container) -and
            @(Get-ChildItem -LiteralPath $parentPath -Force).Count -eq 0) {
            Remove-Item -LiteralPath $parentPath
        }
        Write-Output 'AI Support workstation collector removed.'
    }
} finally { $machine.Dispose() }
