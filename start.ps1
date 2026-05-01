param(
    [bool]$RebuildDashboard = $true,
    [bool]$OpenBrowser = $true
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $root "run-logs"
$backendDir = Join-Path $root "execution-backend"
$alphaDir = Join-Path $root "alpha-engine"
$dashboardDir = Join-Path $root "telemetry_dashboard"
$sentimentDir = Join-Path $root "sentiment-oracle"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path (Join-Path $root ".env"))) {
    Copy-Item (Join-Path $root ".env.example") (Join-Path $root ".env")
}

$envFile = Join-Path $root ".env"
$envMap = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -notmatch '=') {
        return
    }
    $parts = $_ -split '=', 2
    $envMap[$parts[0].Trim()] = $parts[1].Trim()
}

function Wait-HttpOk {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
        }
        Start-Sleep -Seconds 2
    }

    throw "Timed out waiting for $Url"
}

function Save-Pid {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$ProcessId
    )

    Set-Content -Path (Join-Path $logDir "$Name.pid") -Value $ProcessId
}

Write-Host "Starting Postgres..."
docker compose up -d | Out-Host

Write-Host "Starting backend..."
$backend = Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "set JAVA_TOOL_OPTIONS=-Duser.timezone=UTC&& .\gradlew.bat bootRun" `
    -WorkingDirectory $backendDir `
    -RedirectStandardOutput (Join-Path $logDir "backend.out.log") `
    -RedirectStandardError (Join-Path $logDir "backend.err.log") `
    -PassThru
Save-Pid -Name "backend" -ProcessId $backend.Id
Wait-HttpOk -Url "http://localhost:8080/actuator/health" -TimeoutSeconds 180

Write-Host "Installing Python dependencies..."
python -m pip install -r (Join-Path $alphaDir "requirements.txt") | Out-Host

Write-Host "Starting alpha engine..."
$alpha = Start-Process -FilePath "python.exe" `
    -ArgumentList "-u", "main.py" `
    -WorkingDirectory $alphaDir `
    -RedirectStandardOutput (Join-Path $logDir "alpha.out.log") `
    -RedirectStandardError (Join-Path $logDir "alpha.err.log") `
    -PassThru
Save-Pid -Name "alpha" -ProcessId $alpha.Id

if ($envMap["TBOT_ENABLE_SENTIMENT_ORACLE"] -eq "true") {
    Write-Host "Installing sentiment oracle dependencies..."
    python -m pip install -r (Join-Path $sentimentDir "requirements.txt") | Out-Host

    Write-Host "Starting sentiment oracle..."
    $oracle = Start-Process -FilePath "python.exe" `
        -ArgumentList "-u", "main.py" `
        -WorkingDirectory $sentimentDir `
        -RedirectStandardOutput (Join-Path $logDir "sentiment.out.log") `
        -RedirectStandardError (Join-Path $logDir "sentiment.err.log") `
        -PassThru
    Save-Pid -Name "sentiment" -ProcessId $oracle.Id
}

if ($RebuildDashboard) {
    Write-Host "Building dashboard..."
    Push-Location $dashboardDir
    try {
        flutter pub get | Out-Host
        flutter build web --pwa-strategy=none --dart-define=TBOT_BACKEND_URL=http://localhost:8080 --dart-define=TBOT_ENABLE_SUPABASE_STREAMS=false --dart-define=SUPABASE_URL=$($envMap["SUPABASE_URL"]) --dart-define=SUPABASE_ANON_KEY=$($envMap["SUPABASE_ANON_KEY"]) | Out-Host
    } finally {
        Pop-Location
    }
}

Write-Host "Starting dashboard server..."
$dashboard = Start-Process -FilePath "python.exe" `
    -ArgumentList "-m", "http.server", "3000", "-d", "build\web" `
    -WorkingDirectory $dashboardDir `
    -RedirectStandardOutput (Join-Path $logDir "dashboard.out.log") `
    -RedirectStandardError (Join-Path $logDir "dashboard.err.log") `
    -PassThru
Save-Pid -Name "dashboard" -ProcessId $dashboard.Id
Wait-HttpOk -Url "http://localhost:3000" -TimeoutSeconds 30

if ($OpenBrowser) {
    Start-Process "http://localhost:3000"
}

Write-Host ""
Write-Host "Stack is up:"
Write-Host "  Backend:   http://localhost:8080/actuator/health"
Write-Host "  Dashboard: http://localhost:3000"
Write-Host ""
Write-Host "Logs:"
Write-Host "  $logDir\backend.out.log"
Write-Host "  $logDir\alpha.out.log"
if ($envMap["TBOT_ENABLE_SENTIMENT_ORACLE"] -eq "true") {
    Write-Host "  $logDir\sentiment.out.log"
}
Write-Host "  $logDir\dashboard.out.log"
