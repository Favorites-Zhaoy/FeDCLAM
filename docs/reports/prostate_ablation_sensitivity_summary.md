# Prostate Ablation And Sensitivity Summary

This report uses `fedclam_paperparams_clean2_seed0` as the shared paper-parameter base for `FIM + CLAM`, `k=1`, and `lambda_FIM=1e-2`.
The K and lambda_FIM grids follow the paper's Fundus sensitivity grid, but are run on Prostate as an additional reproduction diagnostic.

## Component Ablation

| Setting | Status | Paper | Best-val test | Best-test | Last test | Delta vs paper | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| None | complete | 92.77 | 92.62 | 92.93 | 92.79 | -0.15 |  |
| FIM | complete | 93.07 | 93.09 | 93.16 | 92.85 | 0.02 |  |
| CLAM | complete | 93.50 | 92.70 | 93.08 | 92.73 | -0.80 |  |
| FIM + CLAM | complete | 93.74 | 93.04 | 93.16 | 93.03 | -0.70 | reuses paperparams clean2 base |

## K Sensitivity

| Setting | Status | Paper | Best-val test | Best-test | Last test | Delta vs paper | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| k=1 | complete |  | 93.04 | 93.16 | 93.03 |  | reuses paperparams clean2 base |
| k=2 | complete |  | 92.62 | 93.15 | 93.11 |  | Prostate-only extension of paper Fundus grid |
| k=5 | complete |  | 93.06 | 93.12 | 92.90 |  | Prostate-only extension of paper Fundus grid |
| k=10 | complete |  | 92.75 | 92.90 | 92.65 |  | Prostate-only extension of paper Fundus grid |
| k=20 | complete |  | 92.58 | 92.67 | 92.58 |  | Prostate-only extension of paper Fundus grid |

## Lambda_FIM Sensitivity

| Setting | Status | Paper | Best-val test | Best-test | Last test | Delta vs paper | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| lambda_FIM=1e-4 | complete |  | 92.98 | 93.08 | 92.85 |  | Prostate-only extension of paper Fundus grid |
| lambda_FIM=1e-3 | complete |  | 92.92 | 93.14 | 92.84 |  | Prostate-only extension of paper Fundus grid |
| lambda_FIM=1e-2 | complete |  | 93.04 | 93.16 | 93.03 |  | reuses paperparams clean2 base |
| lambda_FIM=1e-1 | complete |  | 92.20 | 92.20 | 92.08 |  | Prostate-only extension of paper Fundus grid |
| lambda_FIM=1 | complete |  | 90.55 | 90.59 | 90.49 |  | Prostate-only extension of paper Fundus grid |
