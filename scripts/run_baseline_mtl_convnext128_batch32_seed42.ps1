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
$OutputDir = "D:\output\seed_sawp\baseline_mtl_convnext128_batch32_seed42"
$Checkpoint = Join-Path $OutputDir "best.pt"
$MetricsXlsx = Join-Path $OutputDir "test_threshold_metrics.xlsx"
$SummaryXlsx = Join-Path $ProjectRoot "experiment_summary.xlsx"
$SplitManifest = "C:\Users\bum\PycharmProjects\RGS\checkpoints\ablation_shared\split_manifest_seed42_70_10_20.csv"

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

if (-not $SkipTrain) {
    Invoke-Step "Train ConvNeXt moe_dim=128 batch=32 baseline seed=42" @(
        "train_baseline_mtl.py",
        "--model-name", "convnext_tiny",
        "--csv", "C:\Users\bum\PycharmProjects\RGS\JustRAIGS_processed.csv",
        "--image-dir", "C:\justRAIGS_cache", "--cache-dir", "C:\justRAIGS_cache",
        "--split-manifest", $SplitManifest, "--output-dir", $OutputDir, "--checkpoint", $Checkpoint,
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--num-workers", "$NumWorkers",
        "--moe-dim", "128", "--backbone-lr", "1e-5", "--head-lr", "3e-4", "--seed", "$Seed"
    )
}

if (-not $SkipMetrics) {
    Invoke-Step "Export ConvNeXt 128 baseline metrics" @(
        "export_test_threshold_metrics.py", "--checkpoint", $Checkpoint,
        "--output-xlsx", $MetricsXlsx, "--bootstrap-seed", "$Seed",
        "--bootstrap-replicates", "$BootstrapReplicates", "--num-workers", "0"
    )
    Invoke-Step "Append ConvNeXt 128 baseline summary" @(
        "append_experiment_summary.py", "--metrics-xlsx", $MetricsXlsx,
        "--output-xlsx", $SummaryXlsx, "--method", "baseline_mtl_convnext128_batch32",
        "--seed", "$Seed", "--replace"
    )
}

Write-Host "Done."
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Metrics: $MetricsXlsx"
