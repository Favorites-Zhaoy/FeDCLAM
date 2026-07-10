param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$ProcessedData = "D:\MLPractice\datasets\processed",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [ValidateSet("all", "fundus", "prostate_mri")]
    [string]$Dataset = "all",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$datasets = if ($Dataset -eq "all") { @("fundus", "prostate_mri") } else { @($Dataset) }
$runNotes = "paper_seed0"

function Invoke-FedCLAMPaperRun {
    param([string]$DataName)

    $configDataset = $DataName
    $runtimeDataset = if ($DataName -eq "prostate_mri") { "prostate" } else { "fundus" }
    $dataPath = Join-Path (Join-Path $ProcessedData $runtimeDataset) "official"
    $marker = Join-Path $OutputRoot "$runtimeDataset\official\fedclam_${runNotes}\log\metrics.npy"
    if ((Test-Path -LiteralPath $marker) -and -not $Resume) {
        Write-Host "[skip] $DataName FedCLAM official paper result already exists"
        return
    }

    $loss = if ($DataName -eq "fundus") { "diceFIMfundus" } else { "diceFIMce" }
    $arguments = @(
        "main.py", "--config", "config.$configDataset.fedclam",
        "--exp_name", "fedclam", "--run_notes", $runNotes,
        "--seed", "0", "--norm", "in", "--protocol", "official",
        "--data-dir", $dataPath, "--output-dir", $OutputRoot,
        "--loss", $loss, "--alpha", "1", "--beta", "1",
        "--l_fim", "0.01", "--fim_warmup", "10"
    )
    if ($Resume) { $arguments += "--resume" }

    $consoleDir = Join-Path $OutputRoot "console"
    New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null
    $consoleLog = Join-Path $consoleDir "${DataName}_official_fedclam_${runNotes}.log"

    Write-Host "[run] $DataName FedCLAM official paper"
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    $env:PYTHONWARNINGS = "ignore"
    & $Python @arguments 2>&1 | Tee-Object -FilePath $consoleLog
    if ($LASTEXITCODE -ne 0) {
        throw "FedCLAM paper run failed for $DataName with exit code $LASTEXITCODE"
    }
}

Push-Location $repo
try {
    foreach ($dataName in $datasets) {
        Invoke-FedCLAMPaperRun -DataName $dataName
    }
} finally {
    Pop-Location
}
