param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$DataDir = "D:\MLPractice\datasets\processed\prostate\official",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [string]$StartAtName = "",
    [int]$MaxParallel = 1,
    [int]$ExpectedRounds = 500,
    [switch]$Force,
    [switch]$NoResume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$ConsoleRoot = Join-Path $OutputRoot "prostate\console\ablation_sensitivity"
$StatusPath = Join-Path $ConsoleRoot "prostate_ablation_sensitivity_status.csv"
$MasterLog = Join-Path $ConsoleRoot "prostate_ablation_sensitivity_master.log"

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

function New-Run {
    param(
        [string]$Group,
        [string]$Name,
        [string]$Config,
        [string]$ExpName,
        [string]$RunNotes,
        [string]$Loss,
        [AllowNull()][string]$LFim = $null,
        [string]$Alpha = "1",
        [string]$Beta = "1",
        [int]$FimWarmup = 10
    )

    $RunDir = Join-Path $OutputRoot ("prostate\official\{0}_{1}" -f $ExpName, $RunNotes)
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
        RunDir = $RunDir
        Metrics = (Join-Path $RunDir "log\metrics_long.csv")
        MetricsNpy = (Join-Path $RunDir "log\metrics.npy")
        LatestCkpt = (Join-Path $RunDir "ckpt\model_latest.pth")
        BestCkpt = (Join-Path $RunDir "ckpt\model_best.pth")
        ConsoleLog = (Join-Path $ConsoleRoot ("{0}_{1}.log" -f $Group, ($Name -replace '[\\/:*?""<>|]', '_')))
    }
}

function Test-RunComplete {
    param([object]$Run)

    if (Test-Path -LiteralPath $Run.MetricsNpy) {
        return [PSCustomObject]@{ Complete = $true; Reason = "metrics_npy" }
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

    $OldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $null = & $Python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" $Path 2>&1
        return $LASTEXITCODE -eq 0
    } finally {
        $ErrorActionPreference = $OldPreference
    }
}

function Repair-ResumeCheckpoint {
    param([object]$Run)

    if ((Test-Path -LiteralPath $Run.LatestCkpt) -and
        ((Get-Item -LiteralPath $Run.LatestCkpt).Length -eq 0) -and
        (Test-Path -LiteralPath $Run.BestCkpt)) {
        $CorruptPath = "$($Run.LatestCkpt).corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Move-Item -LiteralPath $Run.LatestCkpt -Destination $CorruptPath
        Copy-Item -LiteralPath $Run.BestCkpt -Destination $Run.LatestCkpt
        return "repaired_zero_byte_latest"
    }

    if ((Test-Path -LiteralPath $Run.LatestCkpt) -and
        ((Get-Item -LiteralPath $Run.LatestCkpt).Length -gt 0) -and
        (-not (Test-CheckpointReadable -Path $Run.LatestCkpt)) -and
        (Test-Path -LiteralPath $Run.BestCkpt)) {
        $CorruptPath = "$($Run.LatestCkpt).corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Move-Item -LiteralPath $Run.LatestCkpt -Destination $CorruptPath
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

function Write-StatusRow {
    param(
        [object]$Run,
        [string]$Status,
        [string]$Message = "",
        [int]$ExitCode = 0
    )

    $Row = [PSCustomObject]@{
        Time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        Group = $Run.Group
        Name = $Run.Name
        Status = $Status
        Message = $Message
        ExitCode = $ExitCode
        RunDir = $Run.RunDir
        Metrics = $Run.MetricsNpy
        ConsoleLog = $Run.ConsoleLog
    }

    if (-not (Test-Path -LiteralPath $StatusPath)) {
        $Row | Export-Csv -LiteralPath $StatusPath -NoTypeInformation -Encoding UTF8
    } else {
        $Row | Export-Csv -LiteralPath $StatusPath -NoTypeInformation -Encoding UTF8 -Append
    }
}

function Get-RunArguments {
    param([object]$Run)

    $Args = @(
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
        "--lr", "1e-4",
        "--batch_size", "16",
        "--local_epochs", "1",
        "--fl_rounds", "500",
        "--weight_decay", "1e-4",
        "--fim_method", "w2"
    )

    if ($null -ne $Run.LFim -and $Run.LFim -ne "") {
        $Args += @("--l_fim", "$($Run.LFim)")
    }

    if ($Run.Config -eq "config.prostate_mri.fedclam") {
        $Args += @("--alpha", "$($Run.Alpha)", "--beta", "$($Run.Beta)", "--fim_warmup", "$($Run.FimWarmup)")
    }

    if ((Test-Path -LiteralPath $Run.LatestCkpt) -and -not $NoResume) {
        $Args += "--resume"
    }

    return $Args
}

function Invoke-Run {
    param([object]$Run)

    $Complete = Test-RunComplete -Run $Run
    if ($Complete.Complete -and -not $Force) {
        Write-Host "[skip] $($Run.Name): $($Complete.Reason)"
        Write-StatusRow -Run $Run -Status "skipped_complete" -Message $Complete.Reason
        return
    }

    if (-not $DryRun) {
        $RepairMessage = Repair-ResumeCheckpoint -Run $Run
        if ($RepairMessage) {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $RepairMessage $($Run.Name)" |
                Tee-Object -FilePath $MasterLog -Append
        }
    }

    $Args = Get-RunArguments -Run $Run
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Run.ConsoleLog) | Out-Null

    Write-Host ""
    Write-Host "[run] $($Run.Group) / $($Run.Name)"
    Write-Host "[out] $($Run.RunDir)"
    Write-Host "[log] $($Run.ConsoleLog)"
    Write-Host "[cmd] $(Join-CommandLine -Arguments (@($Python) + $Args))"

    if ($DryRun) {
        Write-StatusRow -Run $Run -Status "dry_run" -Message (Join-CommandLine -Arguments (@($Python) + $Args))
        return
    }

    Push-Location $Repo
    $OldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $env:TF_CPP_MIN_LOG_LEVEL = "3"
        $env:PYTHONWARNINGS = "ignore"
        & $Python @Args 2>&1 | Tee-Object -FilePath $Run.ConsoleLog
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $OldPreference
        Pop-Location
    }

    if ($ExitCode -ne 0) {
        Write-StatusRow -Run $Run -Status "failed" -Message "python_exit=$ExitCode" -ExitCode $ExitCode
        throw "Experiment failed: $($Run.Name), exit code $ExitCode"
    }

    $Post = Test-RunComplete -Run $Run
    if (-not $Post.Complete) {
        Write-StatusRow -Run $Run -Status "finished_incomplete" -Message $Post.Reason
        throw "Experiment finished but is incomplete: $($Run.Name), $($Post.Reason)"
    }

    Write-StatusRow -Run $Run -Status "completed" -Message $Post.Reason
}

function Start-RunJob {
    param([object]$Run)

    $Complete = Test-RunComplete -Run $Run
    if ($Complete.Complete -and -not $Force) {
        Write-Host "[skip] $($Run.Name): $($Complete.Reason)"
        Write-StatusRow -Run $Run -Status "skipped_complete" -Message $Complete.Reason
        return $null
    }

    if (-not $DryRun) {
        $RepairMessage = Repair-ResumeCheckpoint -Run $Run
        if ($RepairMessage) {
            "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] $RepairMessage $($Run.Name)" |
                Tee-Object -FilePath $MasterLog -Append
        }
    }

    $Args = Get-RunArguments -Run $Run
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Run.ConsoleLog) | Out-Null

    Write-Host ""
    Write-Host "[queue] $($Run.Group) / $($Run.Name)"
    Write-Host "[out] $($Run.RunDir)"
    Write-Host "[log] $($Run.ConsoleLog)"
    Write-Host "[cmd] $(Join-CommandLine -Arguments (@($Python) + $Args))"

    if ($DryRun) {
        Write-StatusRow -Run $Run -Status "dry_run" -Message (Join-CommandLine -Arguments (@($Python) + $Args))
        return $null
    }

    Write-StatusRow -Run $Run -Status "started" -Message "parallel_job"
    $Job = Start-Job -Name $Run.Name -ScriptBlock {
        param($PythonPath, $Arguments, $WorkingDir, $LogPath)
        Set-Location $WorkingDir
        $env:TF_CPP_MIN_LOG_LEVEL = "3"
        $env:PYTHONWARNINGS = "ignore"
        & $PythonPath @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        exit $LASTEXITCODE
    } -ArgumentList $Python, $Args, $Repo, $Run.ConsoleLog

    return [PSCustomObject]@{ Job = $Job; Run = $Run }
}

function Invoke-ParallelQueue {
    param([object[]]$Runs)

    $Pending = [System.Collections.Queue]::new()
    foreach ($Run in $Runs) {
        $Pending.Enqueue($Run)
    }

    $Active = @()
    while ($Pending.Count -gt 0 -or $Active.Count -gt 0) {
        while ($Pending.Count -gt 0 -and $Active.Count -lt $MaxParallel) {
            $NextRun = $Pending.Dequeue()
            $Started = Start-RunJob -Run $NextRun
            if ($null -ne $Started) {
                $Active += $Started
            }
        }

        if ($Active.Count -eq 0) {
            continue
        }

        Start-Sleep -Seconds 15
        $StillActive = @()
        foreach ($Item in $Active) {
            if ($Item.Job.State -eq "Running" -or $Item.Job.State -eq "NotStarted") {
                $StillActive += $Item
                continue
            }

            Receive-Job -Job $Item.Job | Out-Null
            $ExitCode = 0
            if ($Item.Job.ChildJobs.Count -gt 0 -and $null -ne $Item.Job.ChildJobs[0].JobStateInfo.Reason) {
                $ExitCode = 1
            }
            Remove-Job -Job $Item.Job -Force

            $Post = Test-RunComplete -Run $Item.Run
            if ($Post.Complete -and $ExitCode -eq 0) {
                Write-Host "[ok] $($Item.Run.Name): $($Post.Reason)"
                Write-StatusRow -Run $Item.Run -Status "completed" -Message $Post.Reason
            } else {
                Write-StatusRow -Run $Item.Run -Status "failed_or_incomplete" -Message $Post.Reason -ExitCode $ExitCode
                throw "Parallel experiment failed or incomplete: $($Item.Run.Name), $($Post.Reason)"
            }
        }
        $Active = $StillActive
    }
}

if ($MaxParallel -lt 1) {
    throw "MaxParallel must be >= 1"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $DataDir)) {
    throw "Prostate official data not found: $DataDir"
}

New-Item -ItemType Directory -Force -Path $ConsoleRoot | Out-Null

$Runs = @()

# Table 3 component ablation on Prostate.
$Runs += New-Run -Group "ablation" -Name "none_fedavg_joint" `
    -Config "config.prostate_mri.fedavg" -ExpName "ablation/ablation_none" `
    -RunNotes "paper_seed0" -Loss "joint"
$Runs += New-Run -Group "ablation" -Name "fim_fedavg_lfim_1e-2" `
    -Config "config.prostate_mri.fedavg" -ExpName "ablation/ablation_fim" `
    -RunNotes "paper_seed0" -Loss "diceFIMce" -LFim "0.01"
$Runs += New-Run -Group "ablation" -Name "clam_lfim_0" `
    -Config "config.prostate_mri.fedclam" -ExpName "ablation/ablation_clam" `
    -RunNotes "paper_seed0" -Loss "diceFIMce" -LFim "0" -Beta "1"
$Runs += New-Run -Group "ablation" -Name "fim_clam_base_lfim_1e-2_k_1" `
    -Config "config.prostate_mri.fedclam" -ExpName "fedclam_paperparams_clean2" `
    -RunNotes "seed0" -Loss "diceFIMce" -LFim "0.01" -Beta "1"

# Prostate-only extrapolation of the paper's Fundus sensitivity grid.
foreach ($K in @("2", "5", "10", "20")) {
    $Runs += New-Run -Group "sensitivity_k" -Name "k_$K" `
        -Config "config.prostate_mri.fedclam" -ExpName "sensitivity/prostate_paper_base/k/sensitivity_k_$K" `
        -RunNotes "paper_seed0" -Loss "diceFIMce" -LFim "0.01" -Beta $K
}

foreach ($Lambda in @("1e-4", "1e-3", "1e-1", "1")) {
    $Runs += New-Run -Group "sensitivity_lfim" -Name "lfim_$Lambda" `
        -Config "config.prostate_mri.fedclam" -ExpName "sensitivity/prostate_paper_base/fim/sensitivity_lfim_$Lambda" `
        -RunNotes "paper_seed0" -Loss "diceFIMce" -LFim $Lambda -Beta "1"
}

if ($StartAtName) {
    $Index = -1
    for ($i = 0; $i -lt $Runs.Count; $i++) {
        if ($Runs[$i].Name -eq $StartAtName) {
            $Index = $i
            break
        }
    }
    if ($Index -lt 0) {
        throw "StartAtName not found: $StartAtName"
    }
    $Runs = $Runs[$Index..($Runs.Count - 1)]
}

Write-Host "[prostate-extra] repo: $Repo"
Write-Host "[prostate-extra] data: $DataDir"
Write-Host "[prostate-extra] output: $OutputRoot"
Write-Host "[prostate-extra] status: $StatusPath"
Write-Host "[prostate-extra] runs queued: $($Runs.Count)"
Write-Host "[prostate-extra] max parallel: $MaxParallel"
Write-Host "[prostate-extra] dry-run: $($DryRun.IsPresent)"
Write-Host "[prostate-extra] CUDA_VISIBLE_DEVICES: $($env:CUDA_VISIBLE_DEVICES)"

if ($MaxParallel -eq 1) {
    foreach ($Run in $Runs) {
        Invoke-Run -Run $Run
    }
} else {
    Invoke-ParallelQueue -Runs $Runs
}

Write-Host ""
Write-Host "[done] Prostate ablation and sensitivity queue finished."
