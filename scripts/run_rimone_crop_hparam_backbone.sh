#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
DATA_ROOT="/root/private_data/FedClAM_pro/data/fundus_rimone_crop_diagnostic/official"
OUTPUT_ROOT="/root/private_data/FedClAM_pro/outputs/rimone_crop_diagnostic"
ROUNDS="200"
SEED="0"
GPUS_CSV="0,3,0,3,1,2"
PARALLEL_JOBS="6"
RESUME=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Run RIM-ONE crop diagnostic fundus hparam sweep and backbone ablations.

Options:
  --data-root PATH
  --output-root PATH
  --rounds N
  --seed N
  --gpus LIST
  --parallel-jobs N
  --python PATH
  --no-resume
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpus) GPUS_CSV="$2"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if (( ${#GPUS[@]} == 0 )); then
  echo "--gpus must not be empty" >&2
  exit 2
fi
if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer" >&2
  exit 2
fi

export DTK_ROOT=/opt/dtk-25.04.2
export LD_LIBRARY_PATH=/opt/dtk-25.04.2/lib:/opt/dtk-25.04.2/hip/lib:/opt/dtk/lib:/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUTPUT_ROOT/console" "$OUTPUT_ROOT/locks"

TASKS=(
  "backbone|E0_baseline_unet|config.fundus.fedclam|backbone_crop_E0_baseline_unet|baseline_unet_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|lr_1e-4|config.fundus.fedclam|hparam_crop_lr_1e-4|hparam_seed${SEED}|diceFIMfundus|0.01|1e-4|8|1|1"
  "backbone|E1_resenc|config.fundus.fedclam_resenc|backbone_crop_E1_resenc|resenc_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|lr_3e-4|config.fundus.fedclam|hparam_crop_lr_3e-4|hparam_seed${SEED}|diceFIMfundus|0.01|3e-4|8|1|1"
  "backbone|E2_resenc_ds|config.fundus.fedclam_resenc_ds|backbone_crop_E2_resenc_ds|resenc_ds_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|lr_1e-3|config.fundus.fedclam|hparam_crop_lr_1e-3|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "backbone|E3_resenc_scse|config.fundus.fedclam_resenc_scse|backbone_crop_E3_resenc_scse|resenc_scse_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|lr_3e-3|config.fundus.fedclam|hparam_crop_lr_3e-3|hparam_seed${SEED}|diceFIMfundus|0.01|3e-3|8|1|1"
  "backbone|E4_resenc_ag|config.fundus.fedclam_resenc_ag|backbone_crop_E4_resenc_ag|resenc_ag_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|lr_1e-2|config.fundus.fedclam|hparam_crop_lr_1e-2|hparam_seed${SEED}|diceFIMfundus|0.01|1e-2|8|1|1"
  "backbone|E5_resattunet|config.fundus.fedclam_resattunet|backbone_crop_E5_resattunet|resattunet_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|batch_size_4|config.fundus.fedclam|hparam_crop_batch_size_4|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|4|1|1"
  "hparam|batch_size_8|config.fundus.fedclam|hparam_crop_batch_size_8|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|batch_size_16|config.fundus.fedclam|hparam_crop_batch_size_16|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|16|1|1"
  "hparam|batch_size_32|config.fundus.fedclam|hparam_crop_batch_size_32|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|32|1|1"
  "hparam|local_epochs_1|config.fundus.fedclam|hparam_crop_local_epochs_1|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|local_epochs_2|config.fundus.fedclam|hparam_crop_local_epochs_2|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|2|1"
  "hparam|local_epochs_3|config.fundus.fedclam|hparam_crop_local_epochs_3|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|3|1"
  "hparam|local_epochs_5|config.fundus.fedclam|hparam_crop_local_epochs_5|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|5|1"
  "hparam|alpha_0|config.fundus.fedclam|hparam_crop_alpha_0|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|0"
  "hparam|alpha_0.5|config.fundus.fedclam|hparam_crop_alpha_0.5|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|0.5"
  "hparam|alpha_1|config.fundus.fedclam|hparam_crop_alpha_1|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|1"
  "hparam|alpha_2|config.fundus.fedclam|hparam_crop_alpha_2|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|2"
  "hparam|alpha_4|config.fundus.fedclam|hparam_crop_alpha_4|hparam_seed${SEED}|diceFIMfundus|0.01|1e-3|8|1|4"
)

pids=()
names=()
job_index=0

compact_finished() {
  local new_pids=()
  local new_names=()
  for i in "${!pids[@]}"; do
    if kill -0 "${pids[$i]}" 2>/dev/null; then
      new_pids+=("${pids[$i]}")
      new_names+=("${names[$i]}")
    else
      wait "${pids[$i]}" || {
        echo "[error] failed: ${names[$i]}" >&2
        exit 1
      }
    fi
  done
  pids=("${new_pids[@]}")
  names=("${new_names[@]}")
}

run_one() {
  local task="$1"
  IFS='|' read -r group name config_module exp_name run_notes loss l_fim lr batch_size local_epochs alpha <<< "$task"
  local run_dir="${OUTPUT_ROOT}/fundus/official/${exp_name}_${run_notes}"
  local done_marker="${run_dir}/log/metrics.npy"
  local console_log="${OUTPUT_ROOT}/console/${group}_${name}_seed${SEED}.log"
  local lock="${OUTPUT_ROOT}/locks/${group}_${name}_seed${SEED}.lock"
  local gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  job_index=$((job_index + 1))

  if [[ -f "$done_marker" ]]; then
    echo "[skip] completed: $run_dir"
    return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then
    if [[ -f "$lock/pid" ]] && kill -0 "$(cat "$lock/pid")" 2>/dev/null; then
      echo "[skip] already running: ${group}_${name} pid=$(cat "$lock/pid")"
      return 0
    fi
    rm -rf "$lock"
    mkdir "$lock"
  fi

  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}"
    "$PYTHON_BIN" main.py
    --dataset fundus
    --config "$config_module"
    --exp_name "$exp_name"
    --run_notes "$run_notes"
    --seed "$SEED"
    --norm in
    --protocol official
    --data-dir "$DATA_ROOT"
    --output-dir "$OUTPUT_ROOT"
    --fl_rounds "$ROUNDS"
    --local_epochs "$local_epochs"
    --lr "$lr"
    --batch_size "$batch_size"
    --weight_decay 1e-4
    --loss "$loss"
    --l_fim "$l_fim"
    --fim_method w2
    --fim_warmup 10
    --alpha "$alpha"
    --beta 1
  )
  [[ "$RESUME" -eq 1 ]] && cmd+=(--resume)

  echo "[run] ${group}/${name} gpu=${gpu} config=${config_module} lr=${lr} bs=${batch_size} local=${local_epochs} alpha=${alpha}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
    rm -rf "$lock"
    return 0
  fi

  (
    echo "$BASHPID" > "$lock/pid"
    trap 'rm -rf "$lock"' EXIT
    "${cmd[@]}"
  ) > "$console_log" 2>&1 &
  pids+=("$!")
  names+=("${group}_${name}")
  echo "[bg] pid=${pids[-1]} name=${names[-1]} log=${console_log}"
}

for task in "${TASKS[@]}"; do
  while (( ${#pids[@]} >= PARALLEL_JOBS )); do
    compact_finished
    sleep 15
  done
  run_one "$task"
done

while (( ${#pids[@]} > 0 )); do
  compact_finished
  sleep 15
done

echo "[done] rimone crop hparam/backbone complete"
