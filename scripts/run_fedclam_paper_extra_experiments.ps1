param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$ProcessedData = "D:\MLPractice\datasets\processed",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [ValidateSet("all", "ablation", "sensitivity")]
    [string]$Experiment = "all",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$paperRoot = Join-Path $OutputRoot "paper_extra"
$consoleDir = Join-Path $paperRoot "console"
$runNotes = "paper_seed0"

function Join-CommandLine {
    param([string[]]$Arguments)

    return ($Arguments | ForEach-Object {
        if ($_ -match '[\s"`]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Name not found: $Path"
    }
}

function Get-DatasetSpec {
    param([ValidateSet("fundus", "prostate_mri")][string]$Dataset)

    if ($Dataset -eq "fundus") {
        return @{
            ConfigDataset = "fundus"
            RuntimeDataset = "fundus"
            DataPath = Join-Path (Join-Path $ProcessedData "fundus") "official"
            Lr = "1e-3"
            BatchSize = "8"
            Rounds = "200"
            BaseLoss = "dicefundus"
            FimLoss = "diceFIMfundus"
        }
    }

    return @{
        ConfigDataset = "prostate_mri"
        RuntimeDataset = "prostate"
        DataPath = Join-Path (Join-Path $ProcessedData "prostate") "official"
        Lr = "1e-4"
        BatchSize = "16"
        Rounds = "500"
        BaseLoss = "joint"
        FimLoss = "diceFIMce"
    }
}

function Invoke-PaperExtraRun {
    param(
        [ValidateSet("ablation", "sensitivity")][string]$Group,
        [ValidateSet("fundus", "prostate_mri")][string]$Dataset,
        [string]$ExpName,
        [string]$Config,
        [string]$Loss,
        [AllowNull()][string]$LFim,
        [string]$Alpha = "1",
        [string]$Beta = "1"
    )

    $spec = Get-DatasetSpec -Dataset $Dataset
    Assert-PathExists -Path $spec.DataPath -Name "$Dataset official data"

    $runDir = Join-Path $paperRoot (Join-Path $spec.RuntimeDataset (Join-Path "official" "${ExpName}_${runNotes}"))
    $doneMarker = Join-Path $runDir "log\metrics.npy"
    $consoleLog = Join-Path $consoleDir "$($Group)_$($spec.RuntimeDataset)_$($ExpName).log"

    if (Test-Path -LiteralPath $doneMarker) {
        Write-Host "[skip] completed: $runDir"
        return
    }

    $arguments = @(
        "main.py",
        "--config", $Config,
        "--exp_name", $ExpName,
        "--run_notes", $runNotes,
        "--seed", "0",
        "--norm", "in",
        "--protocol", "official",
        "--data-dir", $spec.DataPath,
        "--output-dir", $paperRoot,
        "--loss", $Loss,
        "--lr", $spec.Lr,
        "--batch_size", $spec.BatchSize,
        "--local_epochs", "1",
        "--weight_decay", "1e-4",
        "--alpha", $Alpha,
        "--beta", $Beta,
        "--fim_warmup", "10"
    )

    if ($null -ne $LFim -and $LFim -ne "") {
        $arguments += @("--l_fim", $LFim)
    }
    if ($Resume) {
        $arguments += "--resume"
    }

    Write-Host ""
    Write-Host "[run] group=$Group dataset=$($spec.RuntimeDataset) exp=$ExpName"
    Write-Host "[paper] rounds=$($spec.Rounds) lr=$($spec.Lr) batch=$($spec.BatchSize) local_epochs=1 weight_decay=1e-4"
    Write-Host "[out] $runDir"
    Write-Host "[log] $consoleLog"

    if ($DryRun) {
        Write-Host (Join-CommandLine -Arguments (@($Python) + $arguments))
        return
    }

    New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    $env:PYTHONWARNINGS = "ignore"

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @arguments 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference

    if ($exitCode -ne 0) {
        throw "Paper extra experiment failed: $ExpName, exit code $exitCode"
    }
}

function Invoke-AblationRuns {
    foreach ($dataset in @("fundus", "prostate_mri")) {
        $spec = Get-DatasetSpec -Dataset $dataset
        $configPrefix = "config.$($spec.ConfigDataset)"

        Invoke-PaperExtraRun `
            -Group "ablation" `
            -Dataset $dataset `
            -ExpName "ablation_none" `
            -Config "$configPrefix.fedavg" `
            -Loss $spec.BaseLoss `
            -LFim $null

        Invoke-PaperExtraRun `
            -Group "ablation" `
            -Dataset $dataset `
            -ExpName "ablation_fim" `
            -Config "$configPrefix.fedavg" `
            -Loss $spec.FimLoss `
            -LFim "0.01"

        Invoke-PaperExtraRun `
            -Group "ablation" `
            -Dataset $dataset `
            -ExpName "ablation_clam" `
            -Config "$configPrefix.fedclam" `
            -Loss $spec.FimLoss `
            -LFim "0"

        Invoke-PaperExtraRun `
            -Group "ablation" `
            -Dataset $dataset `
            -ExpName "ablation_fim_clam" `
            -Config "$configPrefix.fedclam" `
            -Loss $spec.FimLoss `
            -LFim "0.01"
    }
}

function Invoke-SensitivityRuns {
    $dataset = "fundus"
    $config = "config.fundus.fedclam"
    $loss = "diceFIMfundus"

    foreach ($k in @("1", "2", "5", "10", "20")) {
        Invoke-PaperExtraRun `
            -Group "sensitivity" `
            -Dataset $dataset `
            -ExpName "sensitivity_k_$k" `
            -Config $config `
            -Loss $loss `
            -LFim "0.01" `
            -Alpha "1" `
            -Beta $k
    }

    foreach ($lambda in @("1e-4", "1e-3", "1e-2", "1e-1", "1")) {
        Invoke-PaperExtraRun `
            -Group "sensitivity" `
            -Dataset $dataset `
            -ExpName "sensitivity_lfim_$lambda" `
            -Config $config `
            -Loss $loss `
            -LFim $lambda `
            -Alpha "1" `
            -Beta "1"
    }
}

Assert-PathExists -Path $Python -Name "Python executable"
Assert-PathExists -Path $ProcessedData -Name "Processed data root"
New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null

Write-Host "[paper-extra] repo: $repo"
Write-Host "[paper-extra] output root: $paperRoot"
Write-Host "[paper-extra] experiment: $Experiment; resume: $($Resume.IsPresent); dry-run: $($DryRun.IsPresent)"
Write-Host "[paper-extra] CUDA_VISIBLE_DEVICES: $($env:CUDA_VISIBLE_DEVICES)"

Push-Location $repo
try {
    if ($Experiment -eq "all" -or $Experiment -eq "ablation") {
        Invoke-AblationRuns
    }

    if ($Experiment -eq "all" -or $Experiment -eq "sensitivity") {
        Invoke-SensitivityRuns
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "[done] FedCLAM paper extra experiments finished."
