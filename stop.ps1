$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $root "run-logs"

function Stop-FromPidFile {
    param([Parameter(Mandatory = $true)][string]$Name)

    $pidFile = Join-Path $logDir "$Name.pid"
    if (-not (Test-Path $pidFile)) {
        return
    }

    $savedProcessId = Get-Content $pidFile | Select-Object -First 1
    if ($savedProcessId) {
        try {
            Stop-Process -Id ([int]$savedProcessId) -Force
            Write-Host "Stopped $Name ($savedProcessId)"
        } catch {
            Write-Host "$Name already stopped"
        }
    }

    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

Stop-FromPidFile -Name "dashboard"
Stop-FromPidFile -Name "sentiment"
Stop-FromPidFile -Name "alpha"
Stop-FromPidFile -Name "backend"

Write-Host "Stopping Postgres container..."
docker compose down | Out-Host
