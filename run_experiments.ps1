param(
    [string]$Python = "python",
    [int]$Jobs = 3
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python executable not found on PATH: $Python"
}

$env:PYTHONIOENCODING = "utf-8"

& $Python (Join-Path $Root "scripts\04_eval_cobra_released.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\05_materialize_graphs.py") `
    --manifests exact_sequence_unique homology_p80
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\10_train_cobra.py") --mode all
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\11_train_duet.py") `
    --manifest exact_sequence_unique --jobs $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\11_train_duet.py") `
    --manifest homology_p80 --jobs $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\12_compare_duet_cobra.py") `
    --manifest exact_sequence_unique
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\12_compare_duet_cobra.py") `
    --manifest homology_p80
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\06_build_structure_graphs.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\16_train_structure_towers.py") `
    --variants seq3d seqss3d seq3d_shuffle --jobs $Jobs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\17_compare_structure_tower.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\14_train_grassp.py") `
    --manifest homology_p80
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\14_train_grassp.py") `
    --manifest exact_sequence_unique
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\15_compare_grassp.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\15_compare_grassp.py") `
    --manifest exact_sequence_unique
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $Root "scripts\13_final_summary.py")
exit $LASTEXITCODE
