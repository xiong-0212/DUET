param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable not found on PATH: $Python"
}

$env:PYTHONIOENCODING = "utf-8"

& $Python (Join-Path $Root "scripts\run_all.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python -m unittest discover -s (Join-Path $Root "tests") -v
exit $LASTEXITCODE
