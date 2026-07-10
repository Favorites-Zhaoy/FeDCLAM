param(
    [string]$Python = "python",
    [ValidateSet("prostate_mri", "fundus")]
    [string]$Dataset = "prostate_mri",
    [string]$DataDir = "",
    [string]$OutputDir = "outputs",
    [int]$Rounds = 200,
    [int]$LocalEpochs = 1,
    [double]$Lr = 0,
    [int]$BatchSize = 0,
    [string]$Loss = "",
    [double]$LFim = 0,
    [int]$FimWarmup = 0,
    [string]$Protocol = "official",
    [string]$RunPrefix = "backbone"
)

$ErrorActionPreference = "Stop"

$runtimeDataset = $Dataset
$defaultLoss = ""
$defaultLFim = 0.0
$defaultFimWarmup = 10

if ($Dataset -eq "prostate_mri") {
    $runtimeDataset = "prostate"
    $defaultLoss = "diceFIMce"
    $defaultLFim = 0.1
    $defaultFimWarmup = 100
} elseif ($Dataset -eq "fundus") {
    $defaultLoss = "diceFIMfundus"
    $defaultLFim = 0.01
    $defaultFimWarmup = 10
}

if ($Loss -eq "") { $Loss = $defaultLoss }
if ($LFim -le 0) { $LFim = $defaultLFim }
if ($FimWarmup -le 0) { $FimWarmup = $defaultFimWarmup }

$configs = @(
    @{Id = "E0"; Name = "baseline_unet"; Config = "config.$Dataset.fedclam"},
    @{Id = "E1"; Name = "resenc"; Config = "config.$Dataset.fedclam_resenc"},
    @{Id = "E2"; Name = "resenc_ds"; Config = "config.$Dataset.fedclam_resenc_ds"},
    @{Id = "E3"; Name = "resenc_scse"; Config = "config.$Dataset.fedclam_resenc_scse"},
    @{Id = "E4"; Name = "resenc_ag"; Config = "config.$Dataset.fedclam_resenc_ag"},
    @{Id = "E5"; Name = "resattunet"; Config = "config.$Dataset.fedclam_resattunet"}
)

foreach ($item in $configs) {
    $args = @(
        "main.py",
        "--dataset", $runtimeDataset,
        "--config", $item.Config,
        "--exp_name", "$RunPrefix`_$($item.Id)`_$($item.Name)",
        "--run_notes", $item.Name,
        "--protocol", $Protocol,
        "--output-dir", $OutputDir,
        "--fl_rounds", $Rounds,
        "--local_epochs", $LocalEpochs,
        "--loss", $Loss,
        "--l_fim", $LFim,
        "--fim_warmup", $FimWarmup
    )
    if ($DataDir -ne "") { $args += @("--data-dir", $DataDir) }
    if ($Lr -gt 0) { $args += @("--lr", $Lr) }
    if ($BatchSize -gt 0) { $args += @("--batch_size", $BatchSize) }

    Write-Host "Running $($item.Id): $($item.Config) dataset=$runtimeDataset loss=$Loss l_fim=$LFim fim_warmup=$FimWarmup"
    & $Python @args
}
