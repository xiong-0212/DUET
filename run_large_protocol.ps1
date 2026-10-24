param(
    [string]$Python = "python",
    [int]$Jobs = 2,
    [int]$CVEpochs = 40
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable not found on PATH: $Python"
}

$env:PYTHONIOENCODING = "utf-8"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

& $Python (Join-Path $Root "scripts\20_build_large_protocol.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\21_materialize_large_graphs.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\26_run_large_matrix.py") `
    --stage cv --epochs $CVEpochs --jobs $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\29_build_large_ensemble.py") --stage cv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\26_run_large_matrix.py") `
    --stage final --jobs $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\29_build_large_ensemble.py") --stage final
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\27_build_smartpocket.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\28_build_smartpocket_graphs.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\30_eval_large_external.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\31_analyze_large_results.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\32_generate_large_summary.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\33_validate_large_results.py")
exit $LASTEXITCODE
