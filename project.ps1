Write-Host '--- Checking Docker Desktop ---'

$dockerPath = (Get-Command docker -ErrorAction SilentlyContinue).Path
if (-not $dockerPath) {
    Write-Host 'Docker is not installed. Please install Docker Desktop first.'
    Read-Host 'Press Enter to exit'
    exit 1
}

# Check if Docker daemon is running
try {
    docker info | Out-Null
    Write-Host 'Docker is running.'
} catch {
    Write-Host 'Starting Docker Desktop...'
    $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerDesktop) {
        Start-Process $dockerDesktop
        Start-Sleep -Seconds 15
    } else {
        Write-Host "Docker Desktop not found at $dockerDesktop"
        Read-Host 'Press Enter to exit'
        exit 1
    }
}

Write-Host '--- Cleaning docker environment ---'
docker-compose -f docker\docker-compose.yml down -v --remove-orphans

Write-Host '--- Deleting db and artifacts folders ---'
if (Test-Path .\db) { Remove-Item .\db -Recurse -Force }
if (Test-Path .\artifacts) { Remove-Item .\artifacts -Recurse -Force }

Write-Host '--- Building docker containers (no cache) ---'
docker-compose -f docker\docker-compose.yml build --no-cache

Write-Host '--- Starting docker containers ---'
docker-compose -f docker\docker-compose.yml up -d

Write-Host '--- Opening http://localhost:8501 ---'
Start-Process "http://localhost:8501"

Write-Host '--- Done ---'
Read-Host 'Press Enter to exit'
