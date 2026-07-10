param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$DataDir = "D:\MLPractice\datasets\processed\fundus_rimone_crop_diagnostic\official",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [string]$StartAtName = "",
    [int]$ExpectedRounds = 200,
    [switch]$Force,
    [switch]$NoResume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $OutputRoot "fundus\console\rimone_crop_followup_queue_status.csv"
$masterLog = Join-Path $OutputRoot "fundus\console\rimone_crop_followup_queue_master.log"
$consoleRoot = Join-Path $OutputRoot "fundus\console"

function New-Run {
    param(
        [string]$Group,
        [string]$Name,
        [string]$Config,
        [string]$ExpName,
        [string]$RunNotes,
        [string]$Loss,
        [double]$LFim = 0.0,
        [double]$Alpha = 1.0,
        [double]$Beta = 1.0,
        [int]$FimWarmup = 10,
        [string]$ConsoleLog
    )
    $runDir = Join-Path $OutputRoot ("fundus\official\{0}_{1}" -f $ExpName, $RunNotes)
    [PSCustomObject]@{
        Group = $Group
        Name = $Name
        Config = $Config
        ExpName = $ExpName
        RunNotes = $RunNotes
        Loss = $Loss
        LFim = $LFim
        Alpha = $Alpha
        Beta = $Beta
        FimWarmup = $FimWarmup
        ConsoleLog = $ConsoleLog
        RunDir = $runDir
        Metrics = (Join-Path $runDir "log\metrics_long.csv")
        MetricsNpy = (Join-Path $runDir "log\metrics.npy")
        LatestCkpt = (Join-Path $runDir "ckpt\model_latest.pth")
        BestCkpt = (Join-Path $runDir "ckpt\model_best.pth")
    }
}

function Test-RunComplete {
    param([object]$Run)

    if (Test-Path -LiteralPath $Run.MetricsNpy) {
        return [PSCustomObject]@{ Complete = $true; Reason = "metrics_npy" }
    }

    if ((Test-Path -LiteralPath $Run.ConsoleLog) -and
        (Select-String -LiteralPath $Run.ConsoleLog -Pattern "Finished training" -Quiet)) {
        return [PSCustomObject]@{ Complete = $true; Reason = "finished_training_log" }
    }

    if (Test-Path -LiteralPath $Run.Metrics) {
        try {
            $serverRounds = Import-Csv -LiteralPath $Run.Metrics |
                Where-Object { $_.split -eq "server_test" } |
                Select-Object -ExpandProperty round -Unique
            $roundCount = @($serverRounds).Count
            if ($roundCount -ge $ExpectedRounds) {
                return [PSCustomObject]@{ Complete = $true; Reason = "server_test_rounds=$roundCount" }
            }
            return [PSCustomObject]@{ Complete = $false; Reason = "server_test_rounds=$roundCount" }
        } catch {
            return [PSCustomObject]@{ Complete = $false; Reason = "metrics_parse_failed: $($_.Exception.Message)" }
        }
    }

    return [PSCustomObject]@{ Complete = $false; Reason = "no_metrics" }
}

function Test-CheckpointReadable {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $null = & $Python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" $Path 2>&1
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Repair-ResumeCheckpoint {
    param([object]$Run)

    if ((Test-Path -LiteralPath $Run.LatestCkpt) -and
        ((Get-Item -LiteralPath $Run.LatestCkpt).Length -eq 0) -and
        (Test-Path -LiteralPath $Run.BestCkpt)) {
        $corruptPath = "$($Run.LatestCkpt).corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Move-Item -LiteralPath $Run.LatestCkpt -Destination $corruptPath
        Copy-Item -LiteralPath $Run.BestCkpt -Destination $Run.LatestCkpt
        return "repaired_zero_byte_latest"
    }

    if ((Test-Path -LiteralPath $Run.LatestCkpt) -and
        ((Get-Item -LiteralPath $Run.LatestCkpt).Length -gt 0) -and
        (-not (Test-CheckpointReadable -Path $Run.LatestCkpt)) -and
        (Test-Path -LiteralPath $Run.BestCkpt)) {
        $corruptPath = "$($Run.LatestCkpt).corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Move-Item -LiteralPath $Run.LatestCkpt -Destination $corruptPath
        Copy-Item -LiteralPath $Run.BestCkpt -Destination $Run.LatestCkpt
        return "repaired_unreadable_latest"
    }

    if ((-not (Test-Path -LiteralPath $Run.LatestCkpt)) -and
        (Test-Path -LiteralPath $Run.BestCkpt)) {
        Copy-Item -LiteralPath $Run.BestCkpt -Destination $Run.LatestCkpt
        return "copied_best_to_latest"
    }

    return ""
}

function Write-Status {
    param(
        [System.Collections.ArrayList]$Rows,
        [object]$Run,
        [string]$Status,
        [string]$Message = ""
    )
    $Rows.Add([PSCustomObject]@{
        Time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Group = $Run.Group
        Name = $Run.Name
        ExpName = $Run.ExpName
        RunNotes = $Run.RunNotes
        Status = $Status
        Message = $Message
        Metrics = $Run.Metrics
        ConsoleLog = $Run.ConsoleLog
    }) | Out-Null
    $Rows | Export-Csv -Path $statusPath -NoTypeInformation -Encoding UTF8
}

function Invoke-QueuedRun {
    param([object]$Run)

    $complete = Test-RunComplete -Run $Run
    if ($complete.Complete -and -not $Force) {
        return "skipped_complete:$($complete.Reason)"
    }

    $args = @(
        "main.py",
        "--config", $Run.Config,
        "--exp_name", $Run.ExpName,
        "--run_notes", $Run.RunNotes,
        "--seed", "0",
        "--norm", "in",
        "--protocol", "official",
        "--data-dir", $DataDir,
        "--output-dir", $OutputRoot,
        "--loss", $Run.Loss,
        "--lr", "1e-3",
        "--batch_size", "8",
        "--local_epochs", "1",
        "--fl_rounds", "200",
        "--weight_decay", "1e-4",
        "--fim_method", "w2"
    )

    if ($Run.Loss -like "*FIM*") {
        $args += @("--l_fim", "$($Run.LFim)")
    }
    if ($Run.Config -eq "config.fundus.fedclam") {
        $args += @("--alpha", "$($Run.Alpha)", "--beta", "$($Run.Beta)", "--fim_warmup", "$($Run.FimWarmup)")
    }

    if (-not $DryRun) {
        $repairMessage = Repair-ResumeCheckpoint -Run $Run
        if ($repairMessage) {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $repairMessage $($Run.Group) $($Run.Name)" |
                Tee-Object -FilePath $masterLog -Append
        }
    }
    if ((Test-Path -LiteralPath $Run.LatestCkpt) -and -not $NoResume) {
        $args += "--resume"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Run.ConsoleLog) | Out-Null
    if ($DryRun) {
        "DRYRUN $Python $($args -join ' ')" | Tee-Object -FilePath $Run.ConsoleLog
        return "dry_run"
    }

    Push-Location $repo
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python @args 2>&1 | Tee-Object -FilePath $Run.ConsoleLog
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "Experiment failed with exit code $exitCode"
    }
    if (-not (Test-Path -LiteralPath $Run.Metrics)) {
        throw "Experiment finished but metrics file was not created: $($Run.Metrics)"
    }
    $postComplete = Test-RunComplete -Run $Run
    if (-not $postComplete.Complete) {
        throw "Experiment exited but is still incomplete: $($postComplete.Reason)"
    }
    return "completed"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statusPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $masterLog) | Out-Null

$runs = @()
$runs += New-Run -Group "ablation" -Name "none_crop" `
    -Config "config.fundus.fedavg" `
    -ExpName "ablation/rimone_crop_diagnostic/ablation_none" `
    -RunNotes "rimone_crop_diagnostic_seed0" `
    -Loss "dicefundus" `
    -ConsoleLog (Join-Path $consoleRoot "ablation\ablation_fundus_none_rimone_crop_diagnostic_seed0.log")

$runs += New-Run -Group "ablation" -Name "fim_crop" `
    -Config "config.fundus.fedavg" `
    -ExpName "ablation/rimone_crop_diagnostic/ablation_fim" `
    -RunNotes "rimone_crop_diagnostic_seed0" `
    -Loss "diceFIMfundus" -LFim 0.01 `
    -ConsoleLog (Join-Path $consoleRoot "ablation\ablation_fundus_fim_rimone_crop_diagnostic_seed0.log")

$runs += New-Run -Group "ablation" -Name "clam_crop_existing_check" `
    -Config "config.fundus.fedclam" `
    -ExpName "ablation/rimone_crop_diagnostic/ablation_clam" `
    -RunNotes "rimone_crop_diagnostic_seed0" `
    -Loss "diceFIMfundus" -LFim 0 -Alpha 1 -Beta 1 -FimWarmup 10 `
    -ConsoleLog (Join-Path $consoleRoot "ablation\ablation_fundus_clam_rimone_crop_diagnostic_seed0.log")

$runs += New-Run -Group "ablation" -Name "fim_clam_crop_existing_check" `
    -Config "config.fundus.fedclam" `
    -ExpName "fedclam" `
    -RunNotes "rimone_crop_diagnostic_seed0" `
    -Loss "diceFIMfundus" -LFim 0.01 -Alpha 1 -Beta 1 -FimWarmup 10 `
    -ConsoleLog (Join-Path $consoleRoot "fedclam_rimone_crop_diagnostic_seed0.log")

foreach ($k in @(1, 2, 5, 10, 20)) {
    $runs += New-Run -Group "sensitivity_k" -Name "k_$k" `
        -Config "config.fundus.fedclam" `
        -ExpName "sensitivity/rimone_crop_diagnostic/k/sensitivity_k_$k" `
        -RunNotes "rimone_crop_diagnostic_seed0" `
        -Loss "diceFIMfundus" -LFim 0.01 -Alpha 1 -Beta $k -FimWarmup 10 `
        -ConsoleLog (Join-Path $consoleRoot "sensitivity\rimone_crop_diagnostic\k\sensitivity_k_${k}_rimone_crop_diagnostic_seed0.log")
}

$fimValues = @(
    @{Name="1e-4"; Value=0.0001},
    @{Name="1e-3"; Value=0.001},
    @{Name="1e-2"; Value=0.01},
    @{Name="1e-1"; Value=0.1},
    @{Name="1";    Value=1.0}
)
foreach ($fim in $fimValues) {
    $runs += New-Run -Group "sensitivity_fim" -Name "lfim_$($fim.Name)" `
        -Config "config.fundus.fedclam" `
        -ExpName "sensitivity/rimone_crop_diagnostic/fim/sensitivity_lfim_$($fim.Name)" `
        -RunNotes "rimone_crop_diagnostic_seed0" `
        -Loss "diceFIMfundus" -LFim $fim.Value -Alpha 1 -Beta 1 -FimWarmup 10 `
        -ConsoleLog (Join-Path $consoleRoot "sensitivity\rimone_crop_diagnostic\fim\sensitivity_lfim_$($fim.Name)_rimone_crop_diagnostic_seed0.log")
}

if ($StartAtName) {
    $startIndex = -1
    for ($i = 0; $i -lt $runs.Count; $i++) {
        if ($runs[$i].Name -eq $StartAtName) {
            $startIndex = $i
            break
        }
    }
    if ($startIndex -lt 0) {
        $validNames = ($runs | ForEach-Object { $_.Name }) -join ", "
        throw "StartAtName '$StartAtName' not found. Valid names: $validNames"
    }
    $runs = @($runs[$startIndex..($runs.Count - 1)])
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "3"
$env:PYTHONWARNINGS = "ignore"
$env:PYTHONIOENCODING = "utf-8"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {
    # Non-interactive shells may not expose a writable console.
}

"[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] queue start; runs=$($runs.Count); startAt=$StartAtName; expectedRounds=$ExpectedRounds; force=$Force; noResume=$NoResume; dryRun=$DryRun" | Tee-Object -FilePath $masterLog -Append
$statusRows = [System.Collections.ArrayList]::new()

foreach ($run in $runs) {
    Write-Status -Rows $statusRows -Run $run -Status "started"
    try {
        $result = Invoke-QueuedRun -Run $run
        Write-Status -Rows $statusRows -Run $run -Status $result
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $result $($run.Group) $($run.Name)" | Tee-Object -FilePath $masterLog -Append
    } catch {
        Write-Status -Rows $statusRows -Run $run -Status "failed" -Message $_.Exception.Message
        "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] failed $($run.Group) $($run.Name): $($_.Exception.Message)" | Tee-Object -FilePath $masterLog -Append
        throw
    }
}

"[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] queue complete" | Tee-Object -FilePath $masterLog -Append
