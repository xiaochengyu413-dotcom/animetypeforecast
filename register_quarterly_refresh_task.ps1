param(
    [string]$TaskName = "BangumiThemeForecastQuarterlyRefresh",
    [string]$RunAt = "03:00",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runnerPath = Join-Path $scriptDir "run_quarterly_refresh.ps1"
$powerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$taskCommand = "`"$powerShellExe`" -NoProfile -ExecutionPolicy Bypass -File `"$runnerPath`" -PythonExe `"$PythonExe`""

schtasks.exe /Create `
    /F `
    /TN $TaskName `
    /SC MONTHLY `
    /MO 3 `
    /M JAN,APR,JUL,OCT `
    /D 1 `
    /ST $RunAt `
    /TR $taskCommand

Write-Host "Registered quarterly refresh task: $TaskName"
