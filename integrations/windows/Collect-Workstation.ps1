#requires -Version 5.1
<#
Runs only through the centrally installed ai-support: protocol handler.
The callback is machine configuration, never a URL supplied by the incoming URI.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RequestUri
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

function Test-ServiceUrl([string]$Value) {
    $uri = $null
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Length -gt 2048 -or
        $Value -match '[\s\p{Cc}\\]' -or
        -not [Uri]::TryCreate($Value, [UriKind]::Absolute, [ref]$uri)) {
        return $false
    }
    return ($uri.Scheme -ceq 'https' -and $uri.Host -ne '' -and
        $uri.UserInfo -eq '' -and $uri.Query -eq '' -and $uri.Fragment -eq '' -and
        $uri.AbsolutePath.EndsWith('/api/device-collection', [StringComparison]::Ordinal) -and
        $uri.IsWellFormedOriginalString())
}

try {
    # Common PowerShell parameters must not widen the one-argument protocol.
    if ($PSBoundParameters.Count -ne 1 -or -not $PSBoundParameters.ContainsKey('RequestUri') -or
        -not [regex]::IsMatch($RequestUri, '\Aai-support://collect/[A-Za-z0-9_-]{43}\z',
            [Text.RegularExpressions.RegexOptions]::CultureInvariant)) {
        exit 2
    }
    $token = $RequestUri.Substring('ai-support://collect/'.Length)
    $view = [Microsoft.Win32.RegistryView]::Registry32
    if ([Environment]::Is64BitOperatingSystem) {
        $view = [Microsoft.Win32.RegistryView]::Registry64
    }
    $machine = [Microsoft.Win32.RegistryKey]::OpenBaseKey(
        [Microsoft.Win32.RegistryHive]::LocalMachine, $view)
    try {
        $config = $machine.OpenSubKey('SOFTWARE\AI-Support\WorkstationCollector', $false)
        if ($null -eq $config) { exit 3 }
        try {
            $serviceUrl = $config.GetValue('ServiceUrl', $null,
                [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
            $owner = $config.GetValue('InstallOwner', $null,
                [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        } finally {
            $config.Dispose()
        }
    } finally {
        $machine.Dispose()
    }
    if ($owner -cne 'AI-Support Workstation Collector' -or
        $serviceUrl -isnot [string] -or -not (Test-ServiceUrl $serviceUrl)) {
        exit 3
    }
    $computerName = [Environment]::MachineName
    if (-not [regex]::IsMatch($computerName, '\A[A-Za-z0-9_.-]{1,63}\z') -or
        -not [regex]::IsMatch($computerName, '[A-Za-z0-9]')) {
        exit 4
    }
    $json = @{ computer_name = $computerName } | ConvertTo-Json -Compress
    $payload = [Text.Encoding]::UTF8.GetBytes($json)
    # Process-local TLS 1.2; normal Windows certificate validation stays enabled.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $request = [Net.HttpWebRequest]::Create([Uri]$serviceUrl)
    $request.Method = 'POST'
    $request.ContentType = 'application/json; charset=utf-8'
    $request.Accept = 'application/json'
    $request.AllowAutoRedirect = $false
    $request.Timeout = 10000
    $request.ReadWriteTimeout = 10000
    $request.ContentLength = $payload.Length
    $request.Headers['Authorization'] = 'Bearer ' + $token
    $stream = $request.GetRequestStream()
    try { $stream.Write($payload, 0, $payload.Length) } finally { $stream.Dispose() }
    $response = $request.GetResponse()
    try {
        if ([int]$response.StatusCode -ne 200) { exit 5 }
    } finally {
        $response.Dispose()
    }
    exit 0
} catch {
    # Never print the URI, token, callback response, credentials or exception body.
    [Console]::Error.WriteLine('AI Support: device collection failed.')
    exit 1
}
