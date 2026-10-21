param([int]$Port = 8081)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    & (Join-Path $PSScriptRoot 'start_triton.ps1')
    & '.\.venv\Scripts\python.exe' -X utf8 '.\scripts\public_access.py' start --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'External access startup failed. Read the output above.' }
} finally { Pop-Location }
