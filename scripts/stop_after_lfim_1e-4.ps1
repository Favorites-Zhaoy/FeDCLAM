param(
    [int]$QueuePid = 28188,
    [int]$CurrentPythonPid = 29872
)

$ErrorActionPreference = "Continue"

$metricsPath = "D:\MLPractice\FedCLAM\outputs\fundus\official\sensitivity\rimone_crop_diagnostic\fim\sensitivity_lfim_1e-4_rimone_crop_diagnostic_seed0\log\metrics_long.csv"
$consoleLog = "D:\MLPractice\FedCLAM\outputs\fundus\console\sensitivity\rimone_crop_diagnostic\fim\sensitivity_lfim_1e-4_rimone_crop_diagnostic_seed0.log"
$watchLog = "D:\MLPractice\FedCLAM\outputs\fundus\console\stop_after_lfim_1e-4_watcher.log"

function Write-WatchLog {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $watchLog -Value "[$ts] $Message"
}

function Get-ServerTestRoundCount {
    if (-not (Test-Path $metricsPath)) {
        return 0
    }

    try {
        $rows = Import-Csv $metricsPath
        return (($rows | Where-Object { $_.split -eq "server_test" } | Select-Object -ExpandProperty round -Unique).Count)
    } catch {
        Write-WatchLog "metrics read failed: $($_.Exception.Message)"
        return 0
    }
}

function Test-LfimComplete {
    $rounds = Get-ServerTestRoundCount
    if ($rounds -ge 200) {
        Write-WatchLog "lfim_1e-4 complete by metrics: rounds=$rounds"
        return $true
    }

    if ((Test-Path $consoleLog) -and (Select-String -Path $consoleLog -Pattern "Finished training" -Quiet)) {
        Write-WatchLog "lfim_1e-4 complete by console log"
        return $true
    }

    return $false
}

function Stop-FollowupQueue {
    Write-WatchLog "stopping queue pid=$QueuePid"
    $queue = Get-Process -Id $QueuePid -ErrorAction SilentlyContinue
    if ($queue) {
        Stop-Process -Id $QueuePid -Force
    }

    Start-Sleep -Seconds 2

    $nextRuns = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match "main.py" -and
            $_.CommandLine -match "rimone_crop_diagnostic/fim/sensitivity_lfim_" -and
            $_.CommandLine -notmatch "sensitivity_lfim_1e-4"
        }

    foreach ($proc in $nextRuns) {
        Write-WatchLog "stopping next queued python pid=$($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

Write-WatchLog "watcher started; queue=$QueuePid current_python=$CurrentPythonPid"

$currentGoneSince = $null
while ($true) {
    if (Test-LfimComplete) {
        Stop-FollowupQueue
        Write-WatchLog "watcher done after completed lfim_1e-4"
        break
    }

    $current = Get-Process -Id $CurrentPythonPid -ErrorAction SilentlyContinue
    if (-not $current) {
        if (-not $currentGoneSince) {
            $currentGoneSince = Get-Date
            Write-WatchLog "current python pid=$CurrentPythonPid exited; waiting briefly for final logs"
        } elseif (((Get-Date) - $currentGoneSince).TotalSeconds -ge 20) {
            $rounds = Get-ServerTestRoundCount
            Write-WatchLog "current python stayed exited; rounds=$rounds; stopping queue to prevent next experiment"
            Stop-FollowupQueue
            Write-WatchLog "watcher done after current python exit"
            break
        }
    } else {
        $currentGoneSince = $null
    }

    Start-Sleep -Seconds 15
}
