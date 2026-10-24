param(
    [string]$Python = "python",
    [int]$Jobs = 2,
    [int]$GridJobs = 4
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable not found on PATH: $Python"
}

$env:PYTHONIOENCODING = "utf-8"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

& $Python (Join-Path $Root "scripts\36_all_metric_stats.py") --n-resamples 2000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\37_build_large_shuffled3d.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\24_train_large_duet.py") `
    --method graph_seq3d_shuffle --stage cv --epochs 40
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\25_select_large_cv.py") `
    --methods graph_seq3d_shuffle --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\24_train_large_duet.py") `
    --method graph_seq3d_shuffle --stage final
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\41_analyze_shuffled3d.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\38_build_followup_protocols.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$sizeMethods = @(
    "cobra", "duet_graph", "duet_tcn", "duet_dense",
    "graph_seq3d", "grassp_rnafm"
)

foreach ($Size in @(80, 160)) {
    $ResultsRoot = "results/followup/size_curve/size_$Size"
    $CheckpointRoot = "checkpoints/followup/size_curve/size_$Size"
    & $Python (Join-Path $Root "scripts\40_run_protocol_matrix.py") `
        --manifest "followup_size_${Size}_cv5" `
        --results-root $ResultsRoot `
        --checkpoint-root $CheckpointRoot `
        --run-prefix "followup_size$Size" `
        --methods @sizeMethods `
        --stage all --jobs $Jobs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $Python (Join-Path $Root "scripts\39_build_followup_ensembles.py") `
        --results-root $ResultsRoot --stage all
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python (Join-Path $Root "scripts\42_analyze_size_curve.py") `
    --bootstrap-resamples 1000
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\49_run_repeated_cv.py") --jobs 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\43_analyze_repeated_cv.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\44_run_grasp_grid.py") `
    --stage all --jobs $GridJobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\47_analyze_grasp_grid.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\45_run_duplicate_ablation.py") --jobs 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\46_analyze_duplicate_ablation.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\51_generate_followup_summary.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\48_followup_acceptance.py")
exit $LASTEXITCODE
