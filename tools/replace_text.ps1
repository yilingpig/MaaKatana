[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Path,
    [Parameter(Mandatory)]
    [string]$Old,
    [Parameter(Mandatory)]
    [string]$New,
    [int]$ExpectedCount = 1
)

$ErrorActionPreference = 'Stop'
$resolvedPath = [System.IO.Path]::GetFullPath($Path)
if (-not [System.IO.File]::Exists($resolvedPath)) {
    throw "File not found: $resolvedPath"
}

$text = [System.IO.File]::ReadAllText($resolvedPath)
$count = 0
$offset = 0
while (($index = $text.IndexOf($Old, $offset, [System.StringComparison]::Ordinal)) -ge 0) {
    $count++
    $offset = $index + $Old.Length
}
if ($count -ne $ExpectedCount) {
    throw "Expected $ExpectedCount exact match(es), found $count in $resolvedPath"
}

$updated = $text.Replace($Old, $New)
$tempPath = "$resolvedPath.$([guid]::NewGuid().ToString('N')).tmp"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
try {
    [System.IO.File]::WriteAllText($tempPath, $updated, $utf8NoBom)
    Move-Item -LiteralPath $tempPath -Destination $resolvedPath -Force
} finally {
    if (Test-Path -LiteralPath $tempPath) {
        Remove-Item -LiteralPath $tempPath -Force
    }
}
Write-Output "Replaced $count exact match(es): $resolvedPath"