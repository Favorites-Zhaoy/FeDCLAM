# FedCLAM reproduction on Windows

This workspace keeps the released algorithms intact while adding deterministic data preparation, resumable runs, long-form metrics and two evaluation protocols.

## Runtime

The measured runtime uses:

- Python 3.9.11
- PyTorch 2.7.1 + CUDA 12.8
- NVIDIA RTX 4090 (24 GB), driver 610.62

The exact runtime fingerprint is written to `outputs/environment.json`; `requirements-lock.txt` records the core packages. The optional `environment.yml` is provided for recreating the environment on another machine.

## Data preparation

The released FedCLAM RIM-ONE link contains classification images but no optic disc/cup masks. `scripts/prepare_data.ps1` therefore consumes the Fundus client 1-4 archive published by the official FedEvi repository, whose client identities and totals match the FedCLAM paper.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_data.ps1
python tools\validate_processed_data.py --root D:\MLPractice\datasets\processed
```

The preparation step writes SHA256-backed manifests. The `official` Fundus protocol merges the deterministic validation partition into training and mirrors the released validation/test behavior. The `strict` protocol keeps all three partitions separate. Strict Prostate splits are made by volume instead of foreground slice.

## Tests and experiments

```powershell
python -m unittest tests.test_reproduction -v

# Two-round smoke tests for all 18 dataset/method combinations
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Smoke

# Main tables, strict FedAvg/FedCLAM audit, and released Prostate variant
powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Resume

python tools\analyze_results.py --output-root outputs
```

Runs are sequential on one GPU and may be safely restarted with `-Resume`. A completed run is skipped. Per-round metrics are available as TensorBoard events, `metrics.npy`, and `metrics_long.csv`.

## Measured runtime

One complete synthetic federated round includes local training, aggregation, four client-wise evaluation passes and checkpoint I/O. On the RTX 4090:

- Fundus methods: 10.7-12.6 seconds/round; FedSAM and HarmoFL: about 18 seconds/round.
- Prostate methods: 15-16 seconds/round; FedSAM and HarmoFL: about 24 seconds/round.
- The 18 paper-table runs require about 28.5 GPU-hours.
- The complete requested matrix, including strict reruns and the released-script variant, should be budgeted at 40-45 wall-clock hours after real-data overhead.

## Evaluation caveat

The released Fundus loader uses the same files for validation and test, and the manager evaluates test Dice every round. Prostate keeps validation and test arrays separate but still reports the best observed test round. This establishes a test-leakage risk in the released evaluation path; it does not establish intent or prove subjective cherry-picking. The strict protocol exists to quantify the effect without making that inference.
