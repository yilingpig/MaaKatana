[CmdletBinding()]
param(
    [switch]$SkipSchema,
    [switch]$SkipGit
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'

function Invoke-Checked {
    param(
        [Parameter(Mandatory)]
        [string]$Command,
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

Get-Content -Raw -Encoding UTF8 -LiteralPath '.\assets\interface.json' | ConvertFrom-Json | Out-Null
Get-Content -Raw -Encoding UTF8 -LiteralPath '.\assets\resource\pipeline\Pipeline7.json' | ConvertFrom-Json | Out-Null

$pythonFiles = @(Get-ChildItem -LiteralPath '.\agent' -Filter '*.py' -File | ForEach-Object FullName)
if ($pythonFiles.Count -eq 0) {
    throw 'No agent Python files found'
}
Invoke-Checked -Command 'python' -Arguments (@('-m', 'py_compile') + $pythonFiles)
Write-Output 'Python compile: PASS'

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $projectRoot 'agent'
try {
    Invoke-Checked -Command 'python' -Arguments @('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}
Write-Output 'Python tests: PASS'

if (-not $SkipSchema) {
    Invoke-Checked -Command 'python' -Arguments @('.\tools\validate_schema.py', '--schema-dir', '.\deps\tools', '--resource-dirs', '.\assets\resource')
    Write-Output 'Schema validation: PASS'
}

$pipelinePaths = @(
    (Join-Path $projectRoot 'assets\resource\pipeline\Pipeline7.json'),
    (Join-Path $projectRoot 'install\resource\pipeline\Pipeline7.json'),
    'F:\JUSTFORFUN\MFAAvalonia-v2.13.0\resource\pipeline\Pipeline7.json'
)
$pipelineHashes = Get-FileHash -Algorithm SHA256 -LiteralPath $pipelinePaths
if (($pipelineHashes.Hash | Select-Object -Unique).Count -ne 1) {
    throw 'Pipeline7.json copies are not identical'
}
Write-Output 'Pipeline synchronization: PASS'

$agentCopyNames = @('main.py', 'building_router.py', 'condition_router.py', 'dynamic_swipe.py', 'dynamic_bidirectional_swipe.py', 'nonogram_solver.py', 'color_nonogram_solver.py', 'color_nonogram_core.py', 'color_nonogram_model.py', 'color_nonogram_vision.py', 'color_nonogram_digits.py', 'color_nonogram_truth.py', 'color_nonogram_disambiguation.py', 'color_nonogram_test.py', 'color_nonogram_paint_test.py')
foreach ($agentName in $agentCopyNames) {
    $source = Join-Path $projectRoot "agent\$agentName"
    $copy = Join-Path $projectRoot "install\agent\$agentName"
    if ((Test-Path -LiteralPath $source -PathType Leaf) -and (Test-Path -LiteralPath $copy -PathType Leaf)) {
        $hashes = Get-FileHash -Algorithm SHA256 -LiteralPath @($source, $copy)
        if (($hashes.Hash | Select-Object -Unique).Count -ne 1) {
            throw "Agent copy is not synchronized: $agentName"
        }
    }
}
Write-Output 'Agent synchronization: PASS'

if (-not $SkipGit) {
    & git -C $projectRoot diff --check
    if ($LASTEXITCODE -ne 0) {
        throw "git diff --check failed with exit code $LASTEXITCODE"
    }
    Write-Output 'git diff --check: PASS'
}

Write-Output 'Project checks completed successfully.'
