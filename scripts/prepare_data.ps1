param(
    [string]$Python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python39\python.exe",
    [string]$FundusArchive = "D:\MLPractice\datasets\Fundus-doFE.zip",
    [string]$ProstateSource = "D:\MLPractice\datasets\Prostate MRI",
    [string]$OutputRoot = "D:\MLPractice\datasets\processed"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$sourceRoot = "D:\MLPractice\datasets\sources\Fundus-doFE"
if (-not (Test-Path -LiteralPath $FundusArchive)) { throw "Missing $FundusArchive" }
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null
    tar -xf $FundusArchive -C $sourceRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not extract Fundus archive" }
}
$fundusSource = Get-ChildItem -LiteralPath $sourceRoot -Directory -Recurse |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "Domain1") } |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $fundusSource) { throw "Could not locate the Fundus client directories after extraction" }

& $Python (Join-Path $repo "tools\prepare_datasets.py") `
    --dataset all --fundus-source $fundusSource --prostate-source $ProstateSource `
    --output-root $OutputRoot --seed 0
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed" }
