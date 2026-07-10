param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$ProcessedData = "D:\MLPractice\datasets\processed",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$PaperRoot = Join-Path $OutputRoot "paper_extra"
$ConsoleDir = Join-Path $PaperRoot "console"
$StatusCsv = Join-Path $ConsoleDir "remaining_ablation_sensitivity_status.csv"

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

function Add-StatusRow {
    param(
        [string]$Name,
        [string]$RunDir,
        [string]$Status,
        [int]$ExitCode,
        [string]$LogPath
    )

    $row = [pscustomobject]@{
        time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        name = $Name
        status = $Status
        exit_code = $ExitCode
        run_dir = $RunDir
        log = $LogPath
    }

    if (-not (Test-Path -LiteralPath $StatusCsv)) {
        $row | Export-Csv -LiteralPath $StatusCsv -NoTypeInformation -Encoding UTF8
    } else {
        $row | Export-Csv -LiteralPath $StatusCsv -NoTypeInformation -Encoding UTF8 -Append
    }
}

function Invoke-FedclamRun {
    param(
        [string]$Name,
        [string]$RuntimeDataset,
        [string]$DataPath,
        [string]$ExpName,
        [string]$RunNotes,
        [string]$Config,
        [string]$Loss,
        [string]$Lfim,
        [string]$FimWarmup,
        [string]$Lr,
        [string]$BatchSize,
        [string]$Alpha = "1",
        [string]$Beta = "1"
    )

    $RunDir = Join-Path $PaperRoot (Join-Path $RuntimeDataset (Join-Path "official" "${ExpName}_${RunNotes}"))
    $MetricsPath = Join-Path $RunDir "log\metrics.npy"
    $LatestCkpt = Join-Path $RunDir "ckpt\model_latest.pth"
    $BestCkpt = Join-Path $RunDir "ckpt\model_best.pth"
    $LogPath = Join-Path $ConsoleDir "$Name.log"

    if (Test-Path -LiteralPath $MetricsPath) {
        Write-Host "[skip] complete: $Name"
        Add-StatusRow -Name $Name -RunDir $RunDir -Status "skipped_complete" -ExitCode 0 -LogPath $LogPath
        return
    }

    $arguments = @(
        "main.py",
        "--config", $Config,
        "--data-dir", $DataPath,
        "--output-dir", $PaperRoot,
        "--protocol", "official",
        "--seed", "0",
        "--norm", "in",
        "--loss", $Loss,
        "--l_fim", $Lfim,
        "--fim_warmup", $FimWarmup,
        "--alpha", $Alpha,
        "--beta", $Beta,
        "--lr", $Lr,
        "--batch_size", $BatchSize,
        "--local_epochs", "1",
        "--weight_decay", "1e-4",
        "--exp_name", $ExpName,
        "--run_notes", $RunNotes
    )

    if ((Test-Path -LiteralPath $LatestCkpt) -and (Get-Item -LiteralPath $LatestCkpt).Length -eq 0 -and (Test-Path -LiteralPath $BestCkpt)) {
        $corruptPath = "$LatestCkpt.corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Move-Item -LiteralPath $LatestCkpt -Destination $corruptPath
        Copy-Item -LiteralPath $BestCkpt -Destination $LatestCkpt
        Write-Host "[repair] zero-byte latest checkpoint moved to $corruptPath"
        Write-Host "[repair] copied model_best.pth to model_latest.pth for resume"
    } elseif ((-not (Test-Path -LiteralPath $LatestCkpt)) -and (Test-Path -LiteralPath $BestCkpt)) {
        Copy-Item -LiteralPath $BestCkpt -Destination $LatestCkpt
        Write-Host "[repair] model_latest.pth missing; copied model_best.pth for resume"
    }

    if (Test-Path -LiteralPath $LatestCkpt) {
        $arguments += "--resume"
    }

    Write-Host ""
    Write-Host "[run] $Name"
    Write-Host "[out] $RunDir"
    Write-Host "[log] $LogPath"
    Write-Host "[cmd] $(Join-CommandLine -Arguments (@($Python) + $arguments))"

    if ($DryRun) {
        Add-StatusRow -Name $Name -RunDir $RunDir -Status "dry_run" -ExitCode 0 -LogPath $LogPath
        return
    }

    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    $env:PYTHONWARNINGS = "ignore"

    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @arguments 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference

    if ($exitCode -eq 0 -and (Test-Path -LiteralPath $MetricsPath)) {
        Add-StatusRow -Name $Name -RunDir $RunDir -Status "completed" -ExitCode $exitCode -LogPath $LogPath
        Write-Host "[ok] $Name"
    } elseif ($exitCode -eq 0) {
        Add-StatusRow -Name $Name -RunDir $RunDir -Status "finished_without_metrics" -ExitCode $exitCode -LogPath $LogPath
        Write-Host "[warn] $Name finished without metrics.npy"
    } else {
        Add-StatusRow -Name $Name -RunDir $RunDir -Status "failed" -ExitCode $exitCode -LogPath $LogPath
        Write-Host "[fail] $Name exit=$exitCode"
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}

New-Item -ItemType Directory -Force -Path $ConsoleDir | Out-Null

$FundusData = Join-Path (Join-Path $ProcessedData "fundus") "official"
$ProstateData = Join-Path (Join-Path $ProcessedData "prostate") "official"

if (-not (Test-Path -LiteralPath $FundusData)) {
    throw "Fundus official data not found: $FundusData"
}
if (-not (Test-Path -LiteralPath $ProstateData)) {
    throw "Prostate official data not found: $ProstateData"
}

Write-Host "[remaining] repo: $Repo"
Write-Host "[remaining] output: $PaperRoot"
Write-Host "[remaining] status: $StatusCsv"
Write-Host "[remaining] dry-run: $($DryRun.IsPresent)"
Write-Host "[remaining] CUDA_VISIBLE_DEVICES: $($env:CUDA_VISIBLE_DEVICES)"

Push-Location $Repo
try {
    # Remaining Prostate component ablations.
    Invoke-FedclamRun `
        -Name "remaining_prostate_ablation_clam_paper_seed0" `
        -RuntimeDataset "prostate" `
        -DataPath $ProstateData `
        -ExpName "ablation_clam" `
        -RunNotes "paper_seed0" `
        -Config "config.prostate_mri.fedclam" `
        -Loss "diceFIMce" `
        -Lfim "0" `
        -FimWarmup "10" `
        -Lr "1e-4" `
        -BatchSize "16"

    # Prostate FIM+CLAM with the paper-text parameters.
    Invoke-FedclamRun `
        -Name "remaining_prostate_ablation_fim_clam_paper_seed0" `
        -RuntimeDataset "prostate" `
        -DataPath $ProstateData `
        -ExpName "ablation_fim_clam_paper" `
        -RunNotes "paper_seed0" `
        -Config "config.prostate_mri.fedclam" `
        -Loss "diceFIMce" `
        -Lfim "0.01" `
        -FimWarmup "10" `
        -Lr "1e-4" `
        -BatchSize "16"

    # Paper k sensitivity on Fundus.
    foreach ($k in @("1", "2", "5", "10", "20")) {
        Invoke-FedclamRun `
            -Name "remaining_fundus_sensitivity_k_$k" `
            -RuntimeDataset "fundus" `
            -DataPath $FundusData `
            -ExpName "sensitivity_k_$k" `
            -RunNotes "paper_seed0" `
            -Config "config.fundus.fedclam" `
            -Loss "diceFIMfundus" `
            -Lfim "0.01" `
            -FimWarmup "10" `
            -Lr "1e-3" `
            -BatchSize "8" `
            -Alpha "1" `
            -Beta $k
    }

    # Paper lambda_FIM sensitivity on Fundus.
    foreach ($lambda in @("1e-4", "1e-3", "1e-2", "1e-1", "1")) {
        Invoke-FedclamRun `
            -Name "remaining_fundus_sensitivity_lfim_$lambda" `
            -RuntimeDataset "fundus" `
            -DataPath $FundusData `
            -ExpName "sensitivity_lfim_$lambda" `
            -RunNotes "paper_seed0" `
            -Config "config.fundus.fedclam" `
            -Loss "diceFIMfundus" `
            -Lfim $lambda `
            -FimWarmup "10" `
            -Lr "1e-3" `
            -BatchSize "8" `
            -Alpha "1" `
            -Beta "1"
    }
} finally {
    Pop-Location
}

Write-Host "[done] Remaining ablation and sensitivity queue finished."
