param(
    [string]$DataRoot = "D:\MLPractice\datasets\processed_extended",
    [string]$OutputRoot = "D:\MLPractice\FedCLAM\outputs\extended",
    [switch]$Smoke,
    [switch]$FundusLODO,
    [switch]$FundusAll,
    [switch]$NucleiLODO,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$seeds = if ($Smoke) { @(0) } else { @(0, 1, 2) }
$rounds = if ($Smoke) { 2 } else { 200 }
$fundusMethods = if ($Smoke) { @("fedavg", "fedclam") } else { @("fedavg", "fedprox", "feddyn", "fedsam", "fedclam") }
$nucleiMethods = $fundusMethods

function Invoke-FedCLAM {
    param(
        [string]$Dataset,
        [string]$Method,
        [int]$Seed,
        [string[]]$InnerSites,
        [string[]]$OuterSites,
        [string]$Mode,
        [string]$Notes,
        [string]$Loss = ""
    )

    $config = "config.$Dataset.$Method"
    $dataDir = Join-Path $DataRoot "$Dataset\strict"
    $args = @(
        "main.py",
        "--config", $config,
        "--dataset", $Dataset,
        "--data-dir", $dataDir,
        "--output-dir", $OutputRoot,
        "--protocol", "strict",
        "--train_mode", $Mode,
        "--fl_rounds", "$rounds",
        "--seed", "$Seed",
        "--exp_name", $Method,
        "--run_notes", $Notes,
        "--inner_sites"
    ) + $InnerSites
    if ($OuterSites.Count -gt 0) {
        $args += @("--outer_sites") + $OuterSites
    }
    if ($Loss -ne "") {
        $args += @("--loss", $Loss)
    }
    if ($Resume) {
        $args += "--resume"
    }
    Write-Host "python $($args -join ' ')"
    python @args
}

if ($Smoke -or $FundusLODO) {
    $fundusSites = @("Drishti-GS", "RIM-ONE", "REFUGE_t", "REFUGE_v", "REFUGE2", "G1020", "ORIGA")
    foreach ($outer in @("REFUGE2", "G1020", "ORIGA")) {
        $inner = $fundusSites | Where-Object { $_ -ne $outer }
        foreach ($method in $fundusMethods) {
            foreach ($seed in $seeds) {
                $loss = if ($method -eq "fedclam") { "diceFIMfundus" } else { "dicefundus" }
                Invoke-FedCLAM -Dataset "fundus" -Method $method -Seed $seed `
                    -InnerSites $inner -OuterSites @($outer) -Mode "innerouter" `
                    -Notes "extended_fundus_lodo_${outer}_seed${seed}" -Loss $loss
            }
        }
    }
}

if ($FundusAll) {
    $inner = @("Drishti-GS", "RIM-ONE", "REFUGE_t", "REFUGE_v", "REFUGE2", "G1020", "ORIGA")
    foreach ($method in @("fedavg", "fedclam")) {
        foreach ($seed in $seeds) {
            $loss = if ($method -eq "fedclam") { "diceFIMfundus" } else { "dicefundus" }
            Invoke-FedCLAM -Dataset "fundus" -Method $method -Seed $seed `
                -InnerSites $inner -OuterSites @() -Mode "federated" `
                -Notes "extended_fundus_all_seed${seed}" -Loss $loss
        }
    }
}

if ($Smoke -or $NucleiLODO) {
    $nucleiSites = @("CoNSeP", "CPM17", "CryoNuSeg", "TNBC")
    foreach ($outer in $nucleiSites) {
        $inner = $nucleiSites | Where-Object { $_ -ne $outer }
        foreach ($method in $nucleiMethods) {
            foreach ($seed in $seeds) {
                $loss = if ($method -eq "fedclam") { "diceFIM" } else { "joint" }
                Invoke-FedCLAM -Dataset "nuclei" -Method $method -Seed $seed `
                    -InnerSites $inner -OuterSites @($outer) -Mode "innerouter" `
                    -Notes "extended_nuclei_lodo_${outer}_seed${seed}" -Loss $loss
            }
        }
    }
}
