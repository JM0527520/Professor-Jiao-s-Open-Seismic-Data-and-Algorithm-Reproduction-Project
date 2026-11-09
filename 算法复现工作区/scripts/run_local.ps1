param(
    [ValidateSet('audit','predict','all','verify')]
    [string]$Stage = 'all'
)
$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '请先按 README.md 安装本工作区的 Python 环境。'
}
Push-Location -LiteralPath $workspaceRoot
try {
    New-Item -ItemType Directory -Path 'outputs\logs' -Force | Out-Null
    if ($Stage -eq 'verify') {
        & $pythonPath -X utf8 -m pytest -q -p no:cacheprovider 2>&1 | Tee-Object -FilePath 'outputs\logs\tests.log'
        if ($LASTEXITCODE -ne 0) { throw '回归测试未通过' }
        & $pythonPath -m ruff check fault_workflow tests
        if ($LASTEXITCODE -ne 0) { throw '代码检查未通过' }
    } else {
        & $pythonPath -X utf8 -u -m fault_workflow $Stage 2>&1 | Tee-Object -FilePath "outputs\logs\$Stage.log"
        if ($LASTEXITCODE -ne 0) { throw "$Stage 阶段失败，请查看日志" }
    }
} finally {
    Pop-Location
}
