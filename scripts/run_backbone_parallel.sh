#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
FUNDUS_DATA="/root/private_data/mlp/datasets/processed/fundus/official"
PROSTATE_DATA="/root/private_data/mlp/datasets/processed/prostate/official"
OUTPUT_ROOT="/root/private_data/FedClAM_pro/outputs/backbone"
ROUNDS="200"
LOCAL_EPOCHS="1"
SEED="0"
GPUS_CSV="1,2,3,1,2,3,0,0"
PARALLEL_JOBS="8"
RESUME=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Run fundus and prostate FedCLAM backbone ablations in parallel.

Options:
  --fundus-data PATH
  --prostate-data PATH
  --output-root PATH
  --rounds N
  --local-epochs N
  --seed N
  --gpus LIST             Comma-separated physical GPU ids. Default: 1,2,3,1,2,3,0,0
  --parallel-jobs N       Concurrent Python processes. Default: 8
  --python PATH
  --no-resume
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fundus-data) FUNDUS_DATA="$2"; shift 2 ;;
    --prostate-data) PROSTATE_DATA="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --local-epochs) LOCAL_EPOCHS="$2"; shift 2 ;;
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
  "fundus|E0|baseline_unet|config.fundus.fedclam|fundus|$FUNDUS_DATA|diceFIMfundus|0.01|10"
  "fundus|E1|resenc|config.fundus.fedclam_resenc|fundus|$FUNDUS_DATA|diceFIMfundus|0.01|10"
  "fundus|E2|resenc_ds|config.fundus.fedclam_resenc_ds|fundus|$FUNDUS_DATA|diceFIMfundus|0.01|10"
  "fundus|E3|resenc_scse|config.fundus.fedclam_resenc_scse|fundus|$FUNDUS_DATA|diceFIMfundus|0.01|10"
  "fundus|E4|resenc_ag|config.fundus.fedclam_resenc_ag|fundus|$FUNDUS_DATA|diceFIMfundus|0.01|10"
  "fundus|E5|resattunet|config.fundus.fedclam_resattunet|fundus|$FUNDUS_DATA|diceFIMfundus|0.01|10"
  "prostate|E0|baseline_unet|config.prostate_mri.fedclam|prostate|$PROSTATE_DATA|diceFIMce|0.1|100"
  "prostate|E1|resenc|config.prostate_mri.fedclam_resenc|prostate|$PROSTATE_DATA|diceFIMce|0.1|100"
  "prostate|E2|resenc_ds|config.prostate_mri.fedclam_resenc_ds|prostate|$PROSTATE_DATA|diceFIMce|0.1|100"
  "prostate|E3|resenc_scse|config.prostate_mri.fedclam_resenc_scse|prostate|$PROSTATE_DATA|diceFIMce|0.1|100"
  "prostate|E4|resenc_ag|config.prostate_mri.fedclam_resenc_ag|prostate|$PROSTATE_DATA|diceFIMce|0.1|100"
  "prostate|E5|resattunet|config.prostate_mri.fedclam_resattunet|prostate|$PROSTATE_DATA|diceFIMce|0.1|100"
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
  IFS='|' read -r dataset id name config_module runtime_dataset data_dir loss l_fim fim_warmup <<< "$task"
  local exp_name="backbone_${id}_${name}"
  local run_notes="${name}_seed${SEED}"
  local run_dir="${OUTPUT_ROOT}/${runtime_dataset}/official/${exp_name}_${run_notes}"
  local done_marker="${run_dir}/log/metrics.npy"
  local console_log="${OUTPUT_ROOT}/console/${runtime_dataset}_${id}_${name}_seed${SEED}.log"
  local lock="${OUTPUT_ROOT}/locks/${runtime_dataset}_${id}_${name}_seed${SEED}.lock"
  local gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
  job_index=$((job_index + 1))

  if [[ -f "$done_marker" ]]; then
    echo "[skip] completed: $run_dir"
    return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then
    if [[ -f "$lock/pid" ]] && kill -0 "$(cat "$lock/pid")" 2>/dev/null; then
      echo "[skip] already running: ${runtime_dataset}_${id}_${name} pid=$(cat "$lock/pid")"
      return 0
    fi
    rm -rf "$lock"
    mkdir "$lock"
  fi

  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}"
    "$PYTHON_BIN" main.py
    --dataset "$runtime_dataset"
    --config "$config_module"
    --exp_name "$exp_name"
    --run_notes "$run_notes"
    --seed "$SEED"
    --norm in
    --protocol official
    --data-dir "$data_dir"
    --output-dir "$OUTPUT_ROOT"
    --fl_rounds "$ROUNDS"
    --local_epochs "$LOCAL_EPOCHS"
    --loss "$loss"
    --l_fim "$l_fim"
    --fim_warmup "$fim_warmup"
    --alpha 1
    --beta 1
  )
  [[ "$RESUME" -eq 1 ]] && cmd+=(--resume)

  echo "[run] ${runtime_dataset} ${id}/${name} gpu=${gpu} loss=${loss} out=${run_dir}"
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
  names+=("${runtime_dataset}_${id}_${name}")
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

echo "[done] backbone ablations complete"
