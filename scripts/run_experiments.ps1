param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$ProcessedData = "D:\MLPractice\datasets\processed",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs",
    [ValidateSet("all", "fundus", "prostate_mri")]
    [string]$Dataset = "all",
    [string[]]$Methods = @("fedclam"),
    [switch]$Smoke,
    [switch]$Resume,
    [switch]$IncludeReleased
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$allMethods = @("fedavg", "fedavgm", "fedevi", "fedfa", "fedprox", "fedsam", "feddyn", "fedharmo", "fedclam")
$methods = if ($Methods -contains "all") { $allMethods } else { $Methods }
$datasets = if ($Dataset -eq "all") { @("fundus", "prostate_mri") } else { @($Dataset) }
$rounds = if ($Smoke) { 2 } else { $null }
$runNotes = if ($Smoke) { "smoke_seed0" } else { "paper_seed0" }

function Invoke-FedCLAMRun {
    param([string]$DataName, [string]$Method, [string]$Protocol, [string]$Variant = "paper")
    $configDataset = $DataName
    $runtimeDataset = if ($DataName -eq "prostate_mri") { "prostate" } else { "fundus" }
    $dataPath = Join-Path (Join-Path $ProcessedData $runtimeDataset) $Protocol
    $experiment = if ($Variant -eq "released") { "${Method}_released" } else { $Method }
    $marker = Join-Path $OutputRoot "$runtimeDataset\$Protocol\${experiment}_${runNotes}\log\metrics.npy"
    if ((Test-Path -LiteralPath $marker) -and -not $Resume) {
        Write-Host "[skip] $DataName $Method $Protocol $Variant"
        return
    }
    $arguments = @(
        "main.py", "--config", "config.$configDataset.$Method",
        "--exp_name", $experiment, "--run_notes", $runNotes,
        "--seed", "0", "--norm", "in", "--protocol", $Protocol,
        "--data-dir", $dataPath, "--output-dir", $OutputRoot
    )
    if ($rounds) { $arguments += @("--fl_rounds", "$rounds") }
    if ($Resume) { $arguments += "--resume" }
    if ($Method -eq "fedclam") {
        $loss = if ($DataName -eq "fundus") { "diceFIMfundus" } else { "diceFIMce" }
        $arguments += @("--loss", $loss, "--alpha", "1", "--beta", "1")
        if ($Variant -eq "released") {
            $arguments += @("--l_fim", "0.1", "--fim_warmup", "100")
        } else {
            $arguments += @("--l_fim", "0.01", "--fim_warmup", "10")
        }
    }
    $consoleDir = Join-Path $OutputRoot "console"
    New-Item -ItemType Directory -Force -Path $consoleDir | Out-Null
    $consoleLog = Join-Path $consoleDir "${DataName}_${Protocol}_${experiment}_${runNotes}.log"
    Write-Host "[run] $DataName $Method $Protocol $Variant"
    $env:TF_CPP_MIN_LOG_LEVEL = "3"
    $env:PYTHONWARNINGS = "ignore"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @arguments 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldPreference
    if ($exitCode -ne 0) { throw "Experiment failed with exit code $exitCode" }
}

Push-Location $repo
try {
    foreach ($dataName in $datasets) {
        foreach ($method in $methods) {
            Invoke-FedCLAMRun -DataName $dataName -Method $method -Protocol "official"
        }
    }
    if (-not $Smoke) {
        foreach ($dataName in $datasets) {
            foreach ($method in @("fedclam")) {
                Invoke-FedCLAMRun -DataName $dataName -Method $method -Protocol "strict"
            }
        }
        if ($IncludeReleased -and $Dataset -in @("all", "prostate_mri")) {
            Invoke-FedCLAMRun -DataName "prostate_mri" -Method "fedclam" -Protocol "official" -Variant "released"
        }
    }
} finally {
    Pop-Location
}



