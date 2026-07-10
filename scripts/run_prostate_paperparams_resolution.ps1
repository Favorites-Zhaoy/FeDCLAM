param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$Repo = "D:\MLPractice\FedCLAM",
    [string]$RawProstate = "D:\MLPractice\datasets\Prostate MRI",
    [string]$ProcessedProstate = "D:\MLPractice\datasets\processed\prostate\official",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [switch]$Resume,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$runNotes = "paperparams_clean2_seed0"
$runDir = Join-Path $OutputRoot "prostate\official\fedclam_$runNotes"
$logDir = Join-Path $runDir "log"
$ckptPath = Join-Path $runDir "ckpt\model_best.pth"
$metricsPath = Join-Path $logDir "metrics.npy"
$analysisDir = Join-Path $OutputRoot "prostate\analysis"
$caseAuditDir = Join-Path $analysisDir "paperparams_case_audit"
$consoleDir = Join-Path $OutputRoot "prostate\console"
$statusPath = Join-Path $consoleDir "prostate_paperparams_resolution_status.csv"
$consoleLog = Join-Path $consoleDir "prostate_paperparams_resolution_master.log"

New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null
New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null

function Write-Status {
    param([string]$Step, [string]$Status, [string]$Detail = "")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if (-not (Test-Path -LiteralPath $statusPath)) {
        "timestamp,step,status,detail" | Set-Content -LiteralPath $statusPath -Encoding UTF8
    }
    $safeDetail = $Detail.Replace('"', '""')
    "`"$timestamp`",`"$Step`",`"$Status`",`"$safeDetail`"" | Add-Content -LiteralPath $statusPath -Encoding UTF8
}

function Invoke-Logged {
    param([string]$Step, [string[]]$Arguments)
    Write-Status -Step $Step -Status "started" -Detail ($Arguments -join " ")
    Write-Host "[step] $Step"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $consoleLog -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($exitCode -ne 0) {
        Write-Status -Step $Step -Status "failed" -Detail "exit=$exitCode"
        throw "$Step failed with exit code $exitCode"
    }
    Write-Status -Step $Step -Status "completed"
}

Push-Location $Repo
try {
    Invoke-Logged -Step "official_split_manifest" -Arguments @(
        "tools\audit_prostate_official_split.py",
        "--source-dir", $RawProstate,
        "--processed-dir", $ProcessedProstate,
        "--output-dir", $analysisDir
    )

    if ($SkipTrain) {
        Write-Status -Step "paper_clean_run" -Status "skipped" -Detail "SkipTrain requested"
    } elseif ((Test-Path -LiteralPath $metricsPath) -and -not $Resume) {
        Write-Status -Step "paper_clean_run" -Status "skipped" -Detail "metrics already exist"
        Write-Host "[skip] paper clean run already has metrics: $metricsPath"
    } else {
        $trainArgs = @(
            "main.py",
            "--config", "config.prostate_mri.fedclam",
            "--exp_name", "fedclam",
            "--run_notes", $runNotes,
            "--seed", "0",
            "--norm", "in",
            "--protocol", "official",
            "--data-dir", $ProcessedProstate,
            "--output-dir", $OutputRoot,
            "--loss", "diceFIMce",
            "--l_fim", "0.01",
            "--fim_warmup", "10",
            "--alpha", "1",
            "--beta", "1"
        )
        if ($Resume) { $trainArgs += "--resume" }
        Invoke-Logged -Step "paper_clean_run" -Arguments $trainArgs
    }

    if (Test-Path -LiteralPath $ckptPath) {
        Invoke-Logged -Step "case_audit" -Arguments @(
            "tools\evaluate_prostate_checkpoint_cases.py",
            "--data-dir", $ProcessedProstate,
            "--manifest", (Join-Path $analysisDir "prostate_official_split_manifest.csv"),
            "--checkpoint", $ckptPath,
            "--output-dir", $caseAuditDir,
            "--batch-size", "16"
        )
    } else {
        Write-Status -Step "case_audit" -Status "pending" -Detail "checkpoint not found yet: $ckptPath"
        Write-Host "[pending] case audit waits for checkpoint: $ckptPath"
    }

    Write-Status -Step "resolution_queue" -Status "completed"
} catch {
    Write-Status -Step "resolution_queue" -Status "failed" -Detail $_.Exception.Message
    throw
} finally {
    Pop-Location
}
