# Run the Water Segmentation API. Frees the port if it's in use by a previous run, then starts the server.
$Port = 8000

# Find PID listening on $Port (works on all Windows)
$Line = (netstat -ano | findstr "LISTENING" | findstr ":$Port ") | Select-Object -First 1
if ($Line -and ($Line -match '\s+(\d+)\s*$')) {
    $OldPid = $Matches[1]
    Write-Host "Stopping existing process on port $Port (PID $OldPid)..."
    taskkill /PID $OldPid /F 2>$null
    Start-Sleep -Seconds 2
}

$ModelPath = Join-Path $PSScriptRoot "..\Task 5\best_model.pth"
if (-not $env:MODEL_PATH -and (Test-Path $ModelPath)) {
    $env:MODEL_PATH = (Resolve-Path $ModelPath).Path
}
Set-Location $PSScriptRoot
Write-Host "Starting API on http://localhost:$Port"
uvicorn main:app --host 0.0.0.0 --port $Port
