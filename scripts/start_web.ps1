param([int]$Port = 8080, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Project .venv was not found.' }
    $env:PYTHONUTF8 = '1'
    $env:YOLO_CONFIG_DIR = Join-Path $projectRoot '.tools\ultralytics'
    $env:MPLCONFIGDIR = Join-Path $projectRoot '.tools\matplotlib'
    & $projectPython -c 'import flask, waitress'
    if ($LASTEXITCODE -ne 0) {
        throw 'Install web dependencies first: .\.venv\Scripts\python.exe -m pip install -r requirements-web.txt'
    }
    & (Join-Path $PSScriptRoot 'start_triton.ps1')
    $browserUrl = "http://127.0.0.1:$Port"
    $existing = $null
    try {
        $existing = Invoke-RestMethod "$browserUrl/api/health" -TimeoutSec 3
    } catch { }
    if ($existing.app -eq 'fall-detection-web') {
        Write-Host "Browser detection is already running: $browserUrl"
        if (-not $NoBrowser) { Start-Process $browserUrl }
    } else {
        $webArgs = @('-X', 'utf8', 'web_app.py', '--port', "$Port")
        if (-not $NoBrowser) { $webArgs += '--open-browser' }
        Write-Host "Keep this window open. Browser detection: $browserUrl"
        & $projectPython @webArgs
        if ($LASTEXITCODE -ne 0) { throw 'Browser server stopped with an error. Read the output above.' }
    }
} finally { Pop-Location }
