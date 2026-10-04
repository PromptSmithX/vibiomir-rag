$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $ProjectRoot ".uv-cache"
    uv sync --extra dev
    uv lock
    uv run pytest
}
finally {
    Pop-Location
}
