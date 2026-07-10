# Extended Dataset Generalization

This workflow prepares and runs the revised FedCLAM generalization plan.

## 1. Prepare Data

G1020 and ORIGA require real OD/OC ground-truth masks. Do not use predicted masks.
Pass either a combined mask root or separate disc/cup mask roots.

```powershell
python tools\prepare_extended_datasets.py `
  --dataset all `
  --extended-root D:\MLPractice\ExtendedDatasets `
  --existing-fundus-root D:\MLPractice\datasets\processed\fundus `
  --output-root D:\MLPractice\datasets\processed_extended `
  --g1020-mask-root D:\path\to\G1020\ground_truth_masks `
  --origa-mask-root D:\path\to\ORIGA\ground_truth_masks `
  --require-complete-fundus
```

If masks are stored separately:

```powershell
python tools\prepare_extended_datasets.py `
  --dataset all `
  --extended-root D:\MLPractice\ExtendedDatasets `
  --existing-fundus-root D:\MLPractice\datasets\processed\fundus `
  --output-root D:\MLPractice\datasets\processed_extended `
  --g1020-disc-mask-root D:\path\to\G1020\disc_masks `
  --g1020-cup-mask-root D:\path\to\G1020\cup_masks `
  --origa-disc-mask-root D:\path\to\ORIGA\disc_masks `
  --origa-cup-mask-root D:\path\to\ORIGA\cup_masks `
  --require-complete-fundus
```

Without G1020/ORIGA GT paths, the script prepares REFUGE2 and nuclei datasets, records
G1020/ORIGA as skipped, and fails only when `--require-complete-fundus` is used.

## 2. Validate Data

```powershell
python tools\validate_extended_data.py `
  --root D:\MLPractice\datasets\processed_extended `
  --protocol strict `
  --require-complete-fundus
```

Expected strict counts:

- REFUGE2: 1200
- G1020: 1020
- ORIGA: 650
- CoNSeP: 41
- CPM17: 64
- CryoNuSeg: 30
- TNBC: 50

## 3. Run Experiments

Linux/A100 full run with automatic resume and completed-run skipping:

```bash
nohup bash scripts/run_extended_generalization.sh \
  --data-root /data/FedCLAM/processed_extended \
  --output-root /data/FedCLAM/outputs/extended \
  --gpus 0 \
  > extended_master.log 2>&1 &
```

Linux/A100 prepare + smoke test:

```bash
bash scripts/run_extended_generalization.sh \
  --prepare \
  --mode smoke \
  --extended-root /data/FedCLAM/ExtendedDatasets \
  --existing-fundus-root /data/FedCLAM/datasets/processed/fundus \
  --data-root /data/FedCLAM/processed_extended \
  --output-root /data/FedCLAM/outputs/extended \
  --g1020-mask-root /data/G1020/gt_masks \
  --origa-mask-root /data/ORIGA/gt_masks \
  --gpus 0
```

Multiple A100s:

```bash
nohup bash scripts/run_extended_generalization.sh \
  --data-root /data/FedCLAM/processed_extended \
  --output-root /data/FedCLAM/outputs/extended \
  --gpus 0,1,2,3 \
  --parallel-jobs 4 \
  > extended_master.log 2>&1 &
```

Smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_extended_generalization.ps1 -Smoke
```

Main eye fundus leave-one-dataset-out:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_extended_generalization.ps1 -FundusLODO -Resume
```

Eye fundus all-client training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_extended_generalization.ps1 -FundusAll -Resume
```

Nuclei leave-one-dataset-out:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_extended_generalization.ps1 -NucleiLODO -Resume
```

## 4. Summarize

```powershell
python tools\summarize_extended_results.py `
  --output-root D:\MLPractice\FedCLAM\outputs\extended `
  --summary-dir D:\MLPractice\FedCLAM\outputs\analysis
```
