param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$Python = "C:\Users\bum\PycharmProjects\glaucoma\.venv\Scripts\python.exe",
    [int]$Epochs = 15,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [int]$BootstrapReplicates = 2000,
    [switch]$SkipTrain,
    [switch]$SkipMetrics
)

$ErrorActionPreference = "Stop"
$Seed = 42
$OutputRoot = "D:\output\seed_sawp"
$SummaryXlsx = Join-Path $ProjectRoot "experiment_summary.xlsx"
$Csv = "C:\Users\bum\PycharmProjects\RGS\JustRAIGS_processed.csv"
$ImageDir = "C:\justRAIGS_cache"
$CacheDir = "C:\justRAIGS_cache"
$SplitManifest = "C:\Users\bum\PycharmProjects\RGS\checkpoints\ablation_shared\split_manifest_seed42_70_10_20.csv"

$BaselineDir = Join-Path $OutputRoot "baseline_mtl_convnext128_batch32_seed42"
$BaselineCheckpoint = Join-Path $BaselineDir "best.pt"

function Invoke-Step {
    param([string]$Label, [string[]]$Arguments)
    Push-Location $ProjectRoot
    try {
        Write-Host "==== $Label ===="
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed: $Label" }
    }
    finally { Pop-Location }
}

function Train-GraphVariant {
    param(
        [string]$Name,
        [string]$Method,
        [string]$OutputDir,
        [switch]$FreezeBackbone,
        [switch]$WarmStart,
        [int]$BackboneLrMicro = 0
    )

    $checkpoint = Join-Path $OutputDir "best.pt"
    $metrics = Join-Path $OutputDir "test_threshold_metrics.xlsx"
    $args = @(
        "train_graph_refine_mtl.py", "--model-name", "convnext_tiny",
        "--model-variant", "graph_refine_mtl", "--aux-rg-valid-only",
        "--graph-warmup-epochs", "2",
        "--csv", $Csv, "--image-dir", $ImageDir, "--cache-dir", $CacheDir,
        "--split-manifest", $SplitManifest, "--output-dir", $OutputDir, "--checkpoint", $checkpoint,
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--num-workers", "$NumWorkers",
        "--moe-dim", "128", "--backbone-lr", "0", "--head-lr", "3e-4", "--seed", "$Seed"
    )
    if ($BackboneLrMicro -gt 0) {
        $args[$args.IndexOf("--backbone-lr") + 1] = "1e-5"
    }
    if ($FreezeBackbone) {
        $args += "--freeze-baseline"
    }
    if ($WarmStart) {
        $args += @("--warm-start-checkpoint", $BaselineCheckpoint)
    }

    if (-not $SkipTrain) {
        Invoke-Step "Train $Name seed=42" $args
    }
    elseif (-not (Test-Path -LiteralPath $checkpoint)) {
        throw "Missing checkpoint: $checkpoint"
    }

    if (-not $SkipMetrics) {
        Invoke-Step "Export $Name metrics" @(
            "export_test_threshold_metrics.py", "--checkpoint", $checkpoint,
            "--output-xlsx", $metrics, "--bootstrap-seed", "$Seed",
            "--bootstrap-replicates", "$BootstrapReplicates", "--num-workers", "0"
        )
        Invoke-Step "Append $Name summary" @(
            "append_experiment_summary.py", "--metrics-xlsx", $metrics,
            "--output-xlsx", $SummaryXlsx, "--method", $Method,
            "--seed", "$Seed", "--replace"
        )
    }
}

if (-not $SkipTrain -or -not (Test-Path -LiteralPath $BaselineCheckpoint)) {
    $baselineMetrics = Join-Path $BaselineDir "test_threshold_metrics.xlsx"
    if (-not $SkipTrain) {
        Invoke-Step "Train baseline MTL ConvNeXt 128 batch 32 seed=42" @(
            "train_baseline_mtl.py", "--model-name", "convnext_tiny",
            "--csv", $Csv, "--image-dir", $ImageDir, "--cache-dir", $CacheDir,
            "--split-manifest", $SplitManifest, "--output-dir", $BaselineDir, "--checkpoint", $BaselineCheckpoint,
            "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--num-workers", "$NumWorkers",
            "--moe-dim", "128", "--backbone-lr", "1e-5", "--head-lr", "3e-4", "--seed", "$Seed"
        )
    }
    if (-not $SkipMetrics) {
        Invoke-Step "Export baseline MTL metrics" @(
            "export_test_threshold_metrics.py", "--checkpoint", $BaselineCheckpoint,
            "--output-xlsx", $baselineMetrics, "--bootstrap-seed", "$Seed",
            "--bootstrap-replicates", "$BootstrapReplicates", "--num-workers", "0"
        )
        Invoke-Step "Append baseline MTL summary" @(
            "append_experiment_summary.py", "--metrics-xlsx", $baselineMetrics,
            "--output-xlsx", $SummaryXlsx, "--method", "baseline_mtl_convnext128_batch32",
            "--seed", "$Seed", "--replace"
        )
    }
}

if (-not (Test-Path -LiteralPath $BaselineCheckpoint)) {
    throw "Baseline checkpoint is required before warm-start variants: $BaselineCheckpoint"
}

Train-GraphVariant -Name "rgonly_frozen" -Method "rgonly_frozen" `
    -OutputDir (Join-Path $OutputRoot "rgonly_frozen_convnext128_batch32_seed42") `
    -FreezeBackbone -WarmStart

Train-GraphVariant -Name "rgonly_unfrozen" -Method "rgonly_unfrozen" `
    -OutputDir (Join-Path $OutputRoot "rgonly_unfrozen_convnext128_batch32_seed42") `
    -WarmStart -BackboneLrMicro 1

Train-GraphVariant -Name "rgonly_fromscratch" -Method "rgonly_fromscratch" `
    -OutputDir (Join-Path $OutputRoot "rgonly_fromscratch_convnext128_batch32_seed42") `
    -BackboneLrMicro 1

Write-Host "Done."
Write-Host "Summary: $SummaryXlsx"
