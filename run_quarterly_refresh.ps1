param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $projectDir "generated\logs"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "quarterly_refresh_$timestamp.log"

Set-Location $projectDir
& $PythonExe "refresh_dashboard_site.py" 2>&1 | Tee-Object -FilePath $logPath
