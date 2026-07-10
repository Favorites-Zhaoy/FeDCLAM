#!/usr/bin/env bash
set -euo pipefail
cd /root/private_data/FedClAM_pro
export DTK_ROOT=/opt/dtk-25.04.2
export LD_LIBRARY_PATH=/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}
OUT=/root/private_data/FedClAM_pro/outputs/ordered_extended_fedclam
mkdir -p "$OUT/console" "$OUT/locks"
GPUS=(1 3 0 2)
MAX_JOBS=4
idx=0
running_jobs(){ jobs -rp | wc -l; }
run_one(){
  local dataset="$1" notes="$2" gpu="$3" data_dir="$4" loss="$5" l_fim="$6" fim_warmup="$7"; shift 7
  local exp="fedclam_${notes}"
  local metrics="$OUT/${dataset}/strict/${exp}/log/metrics.npy"
  local lock="$OUT/locks/${dataset}_${exp}.lock"
  local log="$OUT/console/${dataset}_${exp}.log"
  if [[ -f "$metrics" ]]; then echo "[skip] $dataset/$exp metrics exists"; return 0; fi
  if [[ -d "$lock" ]] && [[ -f "$lock/pid" ]] && kill -0 "$(cat "$lock/pid")" 2>/dev/null; then echo "[skip] $dataset/$exp running"; return 0; fi
  rm -rf "$lock"; mkdir -p "$lock"
  echo "[run] $dataset/$exp gpu=$gpu loss=$loss l_fim=$l_fim fim_warmup=$fim_warmup"
  (
    echo $$ > "$lock/pid"
    CUDA_VISIBLE_DEVICES="$gpu" python3 main.py \
      --dataset "$dataset" \
      --config "config.${dataset}.fedclam" \
      --data-dir "$data_dir" \
      --output-dir "$OUT" \
      --protocol strict \
      --train_mode federated \
      --fl_rounds 200 \
      --seed "${notes##*_seed}" \
      --exp_name fedclam \
      --run_notes "$notes" \
      --loss "$loss" \
      --l_fim "$l_fim" \
      --fim_warmup "$fim_warmup" \
      "$@" \
      --resume
    rc=$?
    rm -rf "$lock"
    exit $rc
  ) > "$log" 2>&1 &
  echo "[bg] $dataset/$exp pid=$! log=$log"
}
queue(){
  local dataset="$1" notes="$2" data_dir="$3" loss="$4" l_fim="$5" fim_warmup="$6"; shift 6
  while [[ "$(running_jobs)" -ge "$MAX_JOBS" ]]; do sleep 20; done
  local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"; idx=$((idx+1))
  run_one "$dataset" "$notes" "$gpu" "$data_dir" "$loss" "$l_fim" "$fim_warmup" "$@"
}
F_DATA=/root/private_data/mlp/processed_extended_fundus_gt/fundus/strict
F_SITES=(Drishti-GS RIM-ONE REFUGE_t REFUGE_v REFUGE2 G1020 ORIGA)
for seed in 0 1 2; do queue fundus "extended_fundus_all_seed${seed}" "$F_DATA" diceFIMfundus 0.01 10 --inner_sites "${F_SITES[@]}"; done
N_DATA=/root/private_data/mlp/processed_extended/nuclei/strict
N_SITES=(TNBC CPM17 CoNSeP)
for seed in 0 1 2; do queue nuclei "extended_nuclei_all_seed${seed}" "$N_DATA" diceFIM 0.01 10 --inner_sites "${N_SITES[@]}"; done
wait
echo "[done] ordered extended FedCLAM complete at $(date)"
