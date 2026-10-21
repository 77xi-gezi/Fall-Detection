$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $env:PYTHONUTF8 = '1'
    $env:YOLO_CONFIG_DIR = Join-Path $projectRoot '.tools\ultralytics'
    $env:MPLCONFIGDIR = Join-Path $projectRoot '.tools\matplotlib'
    New-Item -ItemType Directory -Path (Join-Path $projectRoot 'input\images') -Force | Out-Null
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    $projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
    & $projectPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip installation failed.' }
    & $projectPython -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    $projectRequirements = if (Test-Path -LiteralPath 'requirements-windows.lock.txt') {
        'requirements-windows.lock.txt'
    } else {
        'requirements-windows.txt'
    }
    & $projectPython -m pip install -r $projectRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
    & $projectPython -c "from ultralytics import YOLO; YOLO('yolo26s-pose.pt')"
    if ($LASTEXITCODE -ne 0) { throw 'YOLO model download failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
    Write-Host 'Setup completed. Open the VS Code workspace and press F5.'
} finally {
    Pop-Location
}
