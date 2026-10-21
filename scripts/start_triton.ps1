param([switch]$Export)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeFile = Join-Path $projectRoot 'deploy\triton\compose.yaml'
Push-Location -LiteralPath $projectRoot
try {
    if ($Export) {
        & '.\.venv\Scripts\python.exe' -X utf8 scripts\export_triton.py
        if ($LASTEXITCODE -ne 0) { throw 'ONNX export failed.' }
    }
    if (-not (Test-Path -LiteralPath 'deploy\triton\model_repository\yolo26s_pose\1\model.onnx')) {
        throw 'Missing ONNX files. Run this script with -Export first.'
    }
    & docker compose -f $composeFile up -d
    if ($LASTEXITCODE -ne 0) { throw 'Triton startup failed. Ensure Docker Desktop is running.' }
    if ($Export) {
        & docker compose -f $composeFile restart triton
        if ($LASTEXITCODE -ne 0) { throw 'Triton restart after export failed.' }
    }
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $response = Invoke-WebRequest 'http://127.0.0.1:8000/v2/health/ready' -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) {
        & docker compose -f $composeFile logs --tail 60
        throw 'Triton did not become ready. Check the model load logs above.'
    }
    Write-Host 'Triton ready: http://127.0.0.1:8000'
} finally { Pop-Location }
