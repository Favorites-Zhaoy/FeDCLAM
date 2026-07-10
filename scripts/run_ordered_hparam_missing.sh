#!/usr/bin/env bash
set -euo pipefail
cd /root/private_data/FedClAM_pro
export DTK_ROOT=/opt/dtk-25.04.2
export LD_LIBRARY_PATH=/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}
OUT=/root/private_data/FedClAM_pro/outputs/ordered_hparam
mkdir -p "$OUT/console" "$OUT/locks"
GPUS=(0 1 2 3 0 1 2 3)
MAX_JOBS=8
idx=0
running_jobs(){ jobs -rp | wc -l; }
run_job(){
  local dataset="$1" name="$2" gpu="$3" config="$4" data_dir="$5" loss="$6" l_fim="$7" fim_warmup="$8"; shift 8
  local exp="${dataset}_${name}"
  local metrics="$OUT/${dataset}/official/${exp}_hparam_seed0/log/metrics.npy"
  local lock="$OUT/locks/${exp}.lock"
  local log="$OUT/console/${exp}.log"
  if [[ -f "$metrics" ]]; then echo "[skip] $exp metrics exists"; return 0; fi
  if [[ -d "$lock" ]] && [[ -f "$lock/pid" ]] && kill -0 "$(cat "$lock/pid")" 2>/dev/null; then echo "[skip] $exp already running"; return 0; fi
  rm -rf "$lock"; mkdir -p "$lock"
  echo "[run] $exp gpu=$gpu config=$config loss=$loss l_fim=$l_fim fim_warmup=$fim_warmup $*"
  (
    echo $$ > "$lock/pid"
    CUDA_VISIBLE_DEVICES="$gpu" python3 main.py \
      --dataset "$dataset" \
      --config "$config" \
      --data-dir "$data_dir" \
      --output-dir "$OUT" \
      --protocol official \
      --resume \
      --seed 0 \
      --norm in \
      --fl_rounds 200 \
      --exp_name "$exp" \
      --run_notes hparam_seed0 \
      --loss "$loss" \
      --l_fim "$l_fim" \
      --fim_warmup "$fim_warmup" \
      "$@"
    rc=$?
    rm -rf "$lock"
    exit $rc
  ) > "$log" 2>&1 &
  echo "[bg] $exp pid=$! log=$log"
}
queue(){
  local dataset="$1" name="$2" config="$3" data_dir="$4" loss="$5" l_fim="$6" fim_warmup="$7"; shift 7
  while [[ "$(running_jobs)" -ge "$MAX_JOBS" ]]; do sleep 20; done
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"; idx=$((idx+1))
  run_job "$dataset" "$name" "$gpu" "$config" "$data_dir" "$loss" "$l_fim" "$fim_warmup" "$@"
}
P_DATA=/root/private_data/FedClAM_pro/data/prostate_compat/official
P_CFG=config.prostate_mri.fedclam
for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do queue prostate "hp_lr_${lr}" "$P_CFG" "$P_DATA" diceFIMce 0.1 100 --lr "$lr" --batch_size 8 --local_epochs 1 --weight_decay 1e-4 --alpha 1 --beta 1; done
for bs in 4 8 16 32; do queue prostate "hp_batch_size_${bs}" "$P_CFG" "$P_DATA" diceFIMce 0.1 100 --lr 1e-4 --batch_size "$bs" --local_epochs 1 --weight_decay 1e-4 --alpha 1 --beta 1; done
for le in 1 2 3 5; do queue prostate "hp_local_epochs_${le}" "$P_CFG" "$P_DATA" diceFIMce 0.1 100 --lr 1e-4 --batch_size 8 --local_epochs "$le" --weight_decay 1e-4 --alpha 1 --beta 1; done
for a in 0 0.5 1 2 4; do queue prostate "hp_alpha_${a}" "$P_CFG" "$P_DATA" diceFIMce 0.1 100 --lr 1e-4 --batch_size 8 --local_epochs 1 --weight_decay 1e-4 --alpha "$a" --beta 1; done
for wd in 0 1e-6 1e-5 1e-4 1e-3; do queue prostate "hp_weight_decay_${wd}" "$P_CFG" "$P_DATA" diceFIMce 0.1 100 --lr 1e-4 --batch_size 8 --local_epochs 1 --weight_decay "$wd" --alpha 1 --beta 1; done
F_DATA=/root/private_data/FedClAM_pro/data/fundus_rimone_crop_diagnostic/official
F_CFG=config.fundus.fedclam
for wd in 0 1e-6 1e-5 1e-4 1e-3; do queue fundus "crop_hp_weight_decay_${wd}" "$F_CFG" "$F_DATA" diceFIMfundus 0.01 10 --lr 1e-3 --batch_size 8 --local_epochs 1 --weight_decay "$wd" --alpha 1 --beta 1 --fim_method w2; done
wait
echo "[done] ordered hparam missing complete at $(date)"
