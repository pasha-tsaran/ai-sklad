#requires -Version 5.1
<#
Administrator/GPO installation. Sign this source and the collector using the
organization's trusted code-signing certificate before an AllSigned rollout.
No execution policy or certificate trust settings are modified by this script.
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium', PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ServiceUrl,
    [string]$CollectorSource = (Join-Path $PSScriptRoot 'Collect-Workstation.ps1')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$installOwner = 'AI-Support Workstation Collector'
$configPath = 'SOFTWARE\AI-Support\WorkstationCollector'
$protocolPath = 'SOFTWARE\Classes\ai-support'

function Assert-ServiceUrl([string]$Value) {
    $uri = $null
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Length -gt 2048 -or
        $Value -match '[\s\p{Cc}\\]' -or
        -not [Uri]::TryCreate($Value, [UriKind]::Absolute, [ref]$uri) -or
        $uri.Scheme -cne 'https' -or $uri.Host -eq '' -or $uri.UserInfo -ne '' -or
        $uri.Query -ne '' -or $uri.Fragment -ne '' -or
        -not $uri.AbsolutePath.EndsWith('/api/device-collection', [StringComparison]::Ordinal) -or
        -not $uri.IsWellFormedOriginalString()) {
        throw 'ServiceUrl must be an HTTPS callback ending in /api/device-collection, without credentials, query or fragment.'
    }
}

function Assert-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    try {
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw 'Run installation as an administrator or a computer startup GPO.'
        }
    } finally { $identity.Dispose() }
}

function Assert-PlainPath([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Installation path must not contain a symbolic link or junction.'
        }
    }
}

function Set-DirectoryProtection([string]$Path) {
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $administrators = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')
    $system = New-Object Security.Principal.SecurityIdentifier('S-1-5-18')
    $users = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')
    $acl.SetOwner($administrators)
    $inherit = [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
    foreach ($sid in @($administrators, $system)) {
        $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
            $sid, [Security.AccessControl.FileSystemRights]::FullControl, $inherit,
            [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)))
    }
    $acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule(
        $users, [Security.AccessControl.FileSystemRights]::ReadAndExecute, $inherit,
        [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)))
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Set-RegistryProtection([Microsoft.Win32.RegistryKey]$Key) {
    $acl = New-Object Security.AccessControl.RegistrySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $administrators = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')
    $system = New-Object Security.Principal.SecurityIdentifier('S-1-5-18')
    $users = New-Object Security.Principal.SecurityIdentifier('S-1-5-32-545')
    $acl.SetOwner($administrators)
    foreach ($sid in @($administrators, $system)) {
        $acl.AddAccessRule((New-Object Security.AccessControl.RegistryAccessRule(
            $sid, [Security.AccessControl.RegistryRights]::FullControl,
            [Security.AccessControl.InheritanceFlags]::ContainerInherit,
            [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)))
    }
    $acl.AddAccessRule((New-Object Security.AccessControl.RegistryAccessRule(
        $users, [Security.AccessControl.RegistryRights]::ReadKey,
        [Security.AccessControl.InheritanceFlags]::ContainerInherit,
        [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)))
    $Key.SetAccessControl($acl)
}

Assert-ServiceUrl $ServiceUrl
if ([Environment]::Is64BitOperatingSystem -and -not [Environment]::Is64BitProcess) {
    throw 'Use the 64-bit Windows PowerShell host on this computer.'
}
$source = Get-Item -LiteralPath $CollectorSource -Force
if ($source.PSIsContainer -or $source.Extension -ine '.ps1') { throw 'CollectorSource must be a PS1 file.' }
$programFiles = [IO.Path]::GetFullPath([Environment]::GetFolderPath('ProgramFiles'))
$parentPath = Join-Path $programFiles 'AI-Support'
$installPath = [IO.Path]::GetFullPath((Join-Path $parentPath 'WorkstationCollector'))
foreach ($path in @($programFiles, $parentPath, $installPath)) { Assert-PlainPath $path }
$destination = Join-Path $installPath 'Collect-Workstation.ps1'
Assert-PlainPath $destination
$powershell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $powershell -PathType Leaf)) { throw 'Windows PowerShell 5.1 is required.' }
$command = '"' + $powershell + '" -NoProfile -NonInteractive -WindowStyle Hidden -File "' +
    $destination + '" -RequestUri "%1"'
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
                    throw 'An existing configuration or protocol is not owned by this package.'
                }
                if ($path -eq $configPath) { $ownedConfig = $true }
            } finally { $existing.Dispose() }
        }
    }
    if ((Test-Path -LiteralPath $installPath -PathType Container) -and -not $ownedConfig -and
        @(Get-ChildItem -LiteralPath $installPath -Force).Count -ne 0) {
        throw 'The nonempty installation directory has no matching ownership marker; review it before installation.'
    }
    if ($PSCmdlet.ShouldProcess($installPath, 'Install protected AI Support collector and machine URI handler')) {
        Assert-Admin
        $null = New-Item -ItemType Directory -Path $installPath -Force
        foreach ($path in @($programFiles, $parentPath, $installPath)) { Assert-PlainPath $path }
        Set-DirectoryProtection $installPath
        if ($source.FullName -ine $destination) {
            Copy-Item -LiteralPath $source.FullName -Destination $destination -Force
        }
        # The copied file inherits the protected directory ACL, even on updates.
        $fileAcl = Get-Acl -LiteralPath $destination
        $fileAcl.SetAccessRuleProtection($false, $false)
        foreach ($rule in @($fileAcl.GetAccessRules($true, $false, [Security.Principal.SecurityIdentifier]))) {
            $null = $fileAcl.RemoveAccessRuleSpecific($rule)
        }
        $fileAcl.SetOwner((New-Object Security.Principal.SecurityIdentifier('S-1-5-32-544')))
        Set-Acl -LiteralPath $destination -AclObject $fileAcl
        $config = $machine.CreateSubKey($configPath)
        try {
            Set-RegistryProtection $config
            $config.SetValue('InstallOwner', $installOwner, [Microsoft.Win32.RegistryValueKind]::String)
            $config.SetValue('ServiceUrl', $ServiceUrl, [Microsoft.Win32.RegistryValueKind]::String)
        } finally { $config.Dispose() }
        $protocol = $machine.CreateSubKey($protocolPath)
        try {
            Set-RegistryProtection $protocol
            $protocol.SetValue('', 'URL:AI Support workstation collection', [Microsoft.Win32.RegistryValueKind]::String)
            $protocol.SetValue('URL Protocol', '', [Microsoft.Win32.RegistryValueKind]::String)
            $protocol.SetValue('InstallOwner', $installOwner, [Microsoft.Win32.RegistryValueKind]::String)
            $open = $protocol.CreateSubKey('shell\open\command')
            try { $open.SetValue('', $command, [Microsoft.Win32.RegistryValueKind]::String) }
            finally { $open.Dispose() }
        } finally { $protocol.Dispose() }
        Write-Output 'AI Support workstation collector installed.'
    }
} finally { $machine.Dispose() }
