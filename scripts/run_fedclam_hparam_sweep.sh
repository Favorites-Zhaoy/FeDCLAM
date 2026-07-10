#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
FUNDUS_DATA=""
PROSTATE_DATA=""
OUTPUT_ROOT="outputs"
DATASET="all"
GROUP="all"
RESUME=0
DRY_RUN=0
RUN_NOTES="hparam_seed0"
GPUS=""
PARALLEL_JOBS=1
GPU_LIST=()
JOB_INDEX=0
PIDS=()
PID_NAMES=()

usage() {
  cat <<'EOF'
Run FedCLAM one-factor-at-a-time hyperparameter sweeps.

Required data arguments depend on --dataset:
  --fundus-data PATH       Preprocessed official Fundus dataset path
  --prostate-data PATH     Preprocessed official Prostate MRI dataset path

Options:
  --output-root PATH       Root output directory. Default: outputs
  --python PATH            Python executable. Default: python
  --dataset NAME           fundus | prostate | all. Default: all
  --group NAME             lr | batch_size | local_epochs | alpha | weight_decay | all. Default: all
  --gpus LIST              Comma-separated physical GPU ids, e.g. 5,6. Default: do not set CUDA_VISIBLE_DEVICES
  --parallel-jobs N        Number of experiments to run concurrently. Default: 1
  --resume                 Pass --resume to main.py for incomplete runs
  --dry-run                Print commands without running them
  -h, --help               Show this help

Example:
  bash scripts/run_fedclam_hparam_sweep.sh \
    --fundus-data /path/to/processed/fundus/official \
    --prostate-data /path/to/processed/prostate/official \
    --output-root /path/to/FedCLAM/outputs \
    --dataset all \
    --group all

Two A100 example:
  nohup bash scripts/run_fedclam_hparam_sweep.sh \
    --fundus-data /path/to/processed/fundus/official \
    --prostate-data /path/to/processed/prostate/official \
    --output-root /path/to/FedCLAM/outputs \
    --dataset all \
    --group all \
    --gpus 5,6 \
    --parallel-jobs 2 \
    > hparam_sweep_master.log 2>&1 &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fundus-data)
      FUNDUS_DATA="$2"
      shift 2
      ;;
    --prostate-data)
      PROSTATE_DATA="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --parallel-jobs)
      PARALLEL_JOBS="$2"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$DATASET" in
  fundus|prostate|all) ;;
  *)
    echo "--dataset must be one of: fundus, prostate, all" >&2
    exit 2
    ;;
esac

case "$GROUP" in
  lr|batch_size|local_epochs|alpha|weight_decay|all) ;;
  *)
    echo "--group must be one of: lr, batch_size, local_epochs, alpha, weight_decay, all" >&2
    exit 2
    ;;
esac

if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer" >&2
  exit 2
fi

if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
  if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo "--gpus must contain at least one GPU id when provided" >&2
    exit 2
  fi
fi

if [[ "$DATASET" == "fundus" || "$DATASET" == "all" ]]; then
  if [[ -z "$FUNDUS_DATA" ]]; then
    echo "--fundus-data is required when --dataset is fundus or all" >&2
    exit 2
  fi
  if [[ ! -d "$FUNDUS_DATA" ]]; then
    echo "Fundus data directory not found: $FUNDUS_DATA" >&2
    exit 2
  fi
fi

if [[ "$DATASET" == "prostate" || "$DATASET" == "all" ]]; then
  if [[ -z "$PROSTATE_DATA" ]]; then
    echo "--prostate-data is required when --dataset is prostate or all" >&2
    exit 2
  fi
  if [[ ! -d "$PROSTATE_DATA" ]]; then
    echo "Prostate data directory not found: $PROSTATE_DATA" >&2
    exit 2
  fi
fi

SWEEP_ROOT="${OUTPUT_ROOT%/}/hparam_sweeps"
CONSOLE_DIR="${SWEEP_ROOT}/console"
mkdir -p "$CONSOLE_DIR"

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

launch_cmd() {
  local name="$1"
  local console_log="$2"
  shift 2
  local cmd=("$@")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    quote_cmd "${cmd[@]}"
    return 0
  fi

  if [[ "$PARALLEL_JOBS" -le 1 ]]; then
    "${cmd[@]}" 2>&1 | tee "$console_log"
    return $?
  fi

  while [[ "${#PIDS[@]}" -ge "$PARALLEL_JOBS" ]]; do
    wait_for_one_job
  done

  (
    set -euo pipefail
    "${cmd[@]}" 2>&1 | tee "$console_log"
  ) &
  local started_pid="$!"
  PIDS+=("$started_pid")
  PID_NAMES+=("$name")
  echo "[bg] started pid=${started_pid} name=${name}"
}

wait_for_one_job() {
  local pid="${PIDS[0]}"
  local name="${PID_NAMES[0]}"
  local status=0

  set +e
  wait "$pid"
  status=$?
  set -e

  PIDS=("${PIDS[@]:1}")
  PID_NAMES=("${PID_NAMES[@]:1}")

  if [[ "$status" -ne 0 ]]; then
    echo "[error] failed: ${name} (pid=${pid}, exit=${status})" >&2
    exit "$status"
  fi

  echo "[bg] completed pid=${pid} name=${name}"
}

wait_for_all_jobs() {
  while [[ "${#PIDS[@]}" -gt 0 ]]; do
    wait_for_one_job
  done
}

run_one() {
  local dataset="$1"
  local group="$2"
  local value="$3"

  local config_module runtime_dataset data_dir loss l_fim fim_warmup
  local lr batch_size local_epochs alpha weight_decay

  if [[ "$dataset" == "fundus" ]]; then
    config_module="config.fundus.fedclam"
    runtime_dataset="fundus"
    data_dir="$FUNDUS_DATA"
    loss="diceFIMfundus"
    l_fim="0.01"
    fim_warmup="10"
    lr="1e-3"
    batch_size="8"
    local_epochs="1"
    alpha="1"
    weight_decay="1e-4"
  elif [[ "$dataset" == "prostate" ]]; then
    config_module="config.prostate_mri.fedclam"
    runtime_dataset="prostate"
    data_dir="$PROSTATE_DATA"
    loss="diceFIMce"
    l_fim="0.1"
    fim_warmup="100"
    lr="1e-4"
    batch_size="16"
    local_epochs="1"
    alpha="1"
    weight_decay="1e-4"
  else
    echo "Internal error: unsupported dataset $dataset" >&2
    exit 2
  fi

  case "$group" in
    lr) lr="$value" ;;
    batch_size) batch_size="$value" ;;
    local_epochs) local_epochs="$value" ;;
    alpha) alpha="$value" ;;
    weight_decay) weight_decay="$value" ;;
    *)
      echo "Internal error: unsupported group $group" >&2
      exit 2
      ;;
  esac

  local exp_name="fedclam_hp_${group}_${value}"
  local run_dir="${SWEEP_ROOT}/${runtime_dataset}/official/${exp_name}_${RUN_NOTES}"
  local done_marker="${run_dir}/log/metrics.npy"
  local console_log="${CONSOLE_DIR}/${runtime_dataset}_${group}_${value}.log"
  local job_name="${runtime_dataset}_${group}_${value}"
  local gpu=""

  if [[ -f "$done_marker" ]]; then
    echo "[skip] completed: ${run_dir}"
    return 0
  fi

  if [[ "${#GPU_LIST[@]}" -gt 0 ]]; then
    gpu="${GPU_LIST[$((JOB_INDEX % ${#GPU_LIST[@]}))]}"
    JOB_INDEX=$((JOB_INDEX + 1))
  fi

  local cmd=()
  if [[ -n "$gpu" ]]; then
    cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}")
  fi

  cmd+=(
    "$PYTHON_BIN" main.py
    --config "$config_module"
    --exp_name "$exp_name"
    --run_notes "$RUN_NOTES"
    --seed 0
    --norm in
    --protocol official
    --data-dir "$data_dir"
    --output-dir "$SWEEP_ROOT"
    --loss "$loss"
    --l_fim "$l_fim"
    --fim_warmup "$fim_warmup"
    --alpha "$alpha"
    --beta 1
    --lr "$lr"
    --batch_size "$batch_size"
    --local_epochs "$local_epochs"
    --weight_decay "$weight_decay"
  )

  if [[ "$RESUME" -eq 1 ]]; then
    cmd+=(--resume)
  fi

  echo
  echo "[run] dataset=${runtime_dataset} group=${group} value=${value}"
  if [[ -n "$gpu" ]]; then
    echo "[gpu] CUDA_VISIBLE_DEVICES=${gpu}"
  fi
  echo "[out] ${run_dir}"
  echo "[log] ${console_log}"

  launch_cmd "$job_name" "$console_log" "${cmd[@]}"
}

run_group() {
  local dataset="$1"
  local group="$2"

  local values=()
  case "$group" in
    lr)
      if [[ "$dataset" == "fundus" ]]; then
        values=(1e-4 3e-4 1e-3 3e-3 1e-2)
      else
        values=(1e-5 3e-5 1e-4 3e-4 1e-3)
      fi
      ;;
    batch_size)
      values=(4 8 16 32)
      ;;
    local_epochs)
      values=(1 2 3 5)
      ;;
    alpha)
      values=(0 0.5 1 2 4)
      ;;
    weight_decay)
      values=(0 1e-6 1e-5 1e-4 1e-3)
      ;;
  esac

  local value
  for value in "${values[@]}"; do
    run_one "$dataset" "$group" "$value"
  done
}

run_dataset() {
  local dataset="$1"
  local groups=()

  if [[ "$GROUP" == "all" ]]; then
    groups=(lr batch_size local_epochs alpha weight_decay)
  else
    groups=("$GROUP")
  fi

  local group
  for group in "${groups[@]}"; do
    run_group "$dataset" "$group"
  done
}

echo "[sweep] output root: ${SWEEP_ROOT}"
echo "[sweep] dataset: ${DATASET}; group: ${GROUP}; resume: ${RESUME}; dry-run: ${DRY_RUN}"
echo "[sweep] gpus: ${GPUS:-none}; parallel-jobs: ${PARALLEL_JOBS}"

if [[ "$DATASET" == "fundus" || "$DATASET" == "all" ]]; then
  run_dataset fundus
fi

if [[ "$DATASET" == "prostate" || "$DATASET" == "all" ]]; then
  run_dataset prostate
fi

wait_for_all_jobs

echo
echo "[done] FedCLAM hyperparameter sweep command list completed."
