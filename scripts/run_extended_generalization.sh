#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/../datasets/processed_extended"
EXTENDED_ROOT="${REPO_ROOT}/../ExtendedDatasets"
EXISTING_FUNDUS_ROOT="${REPO_ROOT}/../datasets/processed/fundus"
OUTPUT_ROOT="${REPO_ROOT}/outputs/extended"
MODE="full"
PREPARE=0
REQUIRE_COMPLETE_FUNDUS=1
ROUNDS=""
SEEDS="0,1,2"
GPUS=""
PARALLEL_JOBS=1
DRY_RUN=0
RESUME=1
G1020_MASK_ROOT=""
G1020_DISC_MASK_ROOT=""
G1020_CUP_MASK_ROOT=""
ORIGA_MASK_ROOT=""
ORIGA_DISC_MASK_ROOT=""
ORIGA_CUP_MASK_ROOT=""
PIDS=()
PID_NAMES=()
GPU_LIST=()
JOB_INDEX=0

usage() {
  cat <<'EOF'
Run FedCLAM extended generalization experiments on Linux/A100 with auto-resume.

Default behavior:
  - Runs the full revised matrix.
  - Passes --resume to every unfinished run.
  - Skips a run when its metrics.npy already exists.
  - Writes console logs under OUTPUT_ROOT/console.

Options:
  --data-root PATH                 Prepared extended data root. Default: ../datasets/processed_extended
  --extended-root PATH             Raw ExtendedDatasets root for --prepare
  --existing-fundus-root PATH      Existing processed FedCLAM fundus root for --prepare
  --output-root PATH               Experiment output root. Default: outputs/extended
  --python PATH                    Python executable. Default: python
  --mode NAME                      full | smoke | fundus-lodo | fundus-all | nuclei-lodo. Default: full
  --prepare                        Prepare extended data before running
  --no-require-complete-fundus     Allow G1020/ORIGA to be skipped when their real GT masks are absent
  --g1020-mask-root PATH           Combined real G1020 OD/OC GT masks
  --g1020-disc-mask-root PATH      Real G1020 disc GT masks
  --g1020-cup-mask-root PATH       Real G1020 cup GT masks
  --origa-mask-root PATH           Combined real ORIGA OD/OC GT masks
  --origa-disc-mask-root PATH      Real ORIGA disc GT masks
  --origa-cup-mask-root PATH       Real ORIGA cup GT masks
  --rounds N                       Override FL rounds. Default: 2 for smoke, 200 otherwise
  --seeds LIST                     Comma-separated seeds. Default: 0,1,2; smoke forces seed 0 unless set
  --gpus LIST                      Comma-separated physical GPU ids, e.g. 0 or 0,1
  --parallel-jobs N                Number of concurrent runs. Default: 1
  --no-resume                      Do not pass --resume
  --dry-run                        Print commands without running
  -h, --help                       Show this help

Examples:
  # One A100, full matrix, continue after interruption:
  nohup bash scripts/run_extended_generalization.sh \
    --data-root /data/FedCLAM/processed_extended \
    --output-root /data/FedCLAM/outputs/extended \
    --gpus 0 > extended_master.log 2>&1 &

  # Prepare + smoke test:
  bash scripts/run_extended_generalization.sh \
    --prepare --mode smoke --gpus 0 \
    --g1020-mask-root /data/G1020/gt_masks \
    --origa-mask-root /data/ORIGA/gt_masks
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --extended-root) EXTENDED_ROOT="$2"; shift 2 ;;
    --existing-fundus-root) EXISTING_FUNDUS_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --prepare) PREPARE=1; shift ;;
    --no-require-complete-fundus) REQUIRE_COMPLETE_FUNDUS=0; shift ;;
    --g1020-mask-root) G1020_MASK_ROOT="$2"; shift 2 ;;
    --g1020-disc-mask-root) G1020_DISC_MASK_ROOT="$2"; shift 2 ;;
    --g1020-cup-mask-root) G1020_CUP_MASK_ROOT="$2"; shift 2 ;;
    --origa-mask-root) ORIGA_MASK_ROOT="$2"; shift 2 ;;
    --origa-disc-mask-root) ORIGA_DISC_MASK_ROOT="$2"; shift 2 ;;
    --origa-cup-mask-root) ORIGA_CUP_MASK_ROOT="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --parallel-jobs) PARALLEL_JOBS="$2"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODE" in
  full|smoke|fundus-lodo|fundus-all|nuclei-lodo) ;;
  *) echo "--mode must be one of: full, smoke, fundus-lodo, fundus-all, nuclei-lodo" >&2; exit 2 ;;
esac

if ! [[ "$PARALLEL_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--parallel-jobs must be a positive integer" >&2
  exit 2
fi

if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
  if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo "--gpus must contain at least one GPU id" >&2
    exit 2
  fi
fi

if [[ -z "$ROUNDS" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    ROUNDS=2
    if [[ "$SEEDS" == "0,1,2" ]]; then SEEDS="0"; fi
  else
    ROUNDS=200
  fi
fi

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
CONSOLE_DIR="${OUTPUT_ROOT%/}/console"
mkdir -p "$CONSOLE_DIR"

quote_cmd() {
  local arg
  for arg in "$@"; do printf '%q ' "$arg"; done
  printf '\n'
}

run_done() {
  local dataset="$1"
  local method="$2"
  local notes="$3"
  local exp_name="${method}_${notes}"
  [[ -f "${OUTPUT_ROOT%/}/${dataset}/strict/${exp_name}/log/metrics.npy" ]]
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
    echo "[fail] ${name} exited with status ${status}" >&2
    exit "$status"
  fi
  echo "[done] ${name}"
}

wait_for_all_jobs() {
  while [[ "${#PIDS[@]}" -gt 0 ]]; do
    wait_for_one_job
  done
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
    return "${PIPESTATUS[0]}"
  fi

  while [[ "${#PIDS[@]}" -ge "$PARALLEL_JOBS" ]]; do
    wait_for_one_job
  done

  (
    set -euo pipefail
    "${cmd[@]}" 2>&1 | tee "$console_log"
  ) &
  local pid="$!"
  PIDS+=("$pid")
  PID_NAMES+=("$name")
  echo "[bg] started pid=${pid} name=${name}"
}

run_python() {
  local name="$1"
  shift
  local cmd=("$PYTHON_BIN" "$@")
  if [[ -n "$GPUS" ]]; then
    local gpu="${GPU_LIST[$((JOB_INDEX % ${#GPU_LIST[@]}))]}"
    JOB_INDEX=$((JOB_INDEX + 1))
    cmd=("env" "CUDA_VISIBLE_DEVICES=${gpu}" "${cmd[@]}")
  fi
  launch_cmd "$name" "${CONSOLE_DIR}/${name}.log" "${cmd[@]}"
}

without_site() {
  local csv="$1"
  local excluded="$2"
  local result=()
  local sites=()
  local site
  IFS=',' read -r -a sites <<< "$csv"
  for site in "${sites[@]}"; do
    if [[ "$site" != "$excluded" ]]; then
      result+=("$site")
    fi
  done
  local IFS=','
  echo "${result[*]}"
}

maybe_prepare() {
  if [[ "$PREPARE" -ne 1 ]]; then
    return 0
  fi

  local args=(
    "${REPO_ROOT}/tools/prepare_extended_datasets.py"
    --dataset all
    --extended-root "$EXTENDED_ROOT"
    --existing-fundus-root "$EXISTING_FUNDUS_ROOT"
    --output-root "$DATA_ROOT"
  )
  [[ -n "$G1020_MASK_ROOT" ]] && args+=(--g1020-mask-root "$G1020_MASK_ROOT")
  [[ -n "$G1020_DISC_MASK_ROOT" ]] && args+=(--g1020-disc-mask-root "$G1020_DISC_MASK_ROOT")
  [[ -n "$G1020_CUP_MASK_ROOT" ]] && args+=(--g1020-cup-mask-root "$G1020_CUP_MASK_ROOT")
  [[ -n "$ORIGA_MASK_ROOT" ]] && args+=(--origa-mask-root "$ORIGA_MASK_ROOT")
  [[ -n "$ORIGA_DISC_MASK_ROOT" ]] && args+=(--origa-disc-mask-root "$ORIGA_DISC_MASK_ROOT")
  [[ -n "$ORIGA_CUP_MASK_ROOT" ]] && args+=(--origa-cup-mask-root "$ORIGA_CUP_MASK_ROOT")
  [[ "$REQUIRE_COMPLETE_FUNDUS" -eq 1 ]] && args+=(--require-complete-fundus)

  echo "[prepare] ${PYTHON_BIN} ${args[*]}"
  "$PYTHON_BIN" "${args[@]}"

  local validate_args=("${REPO_ROOT}/tools/validate_extended_data.py" --root "$DATA_ROOT" --protocol strict)
  [[ "$REQUIRE_COMPLETE_FUNDUS" -eq 1 ]] && validate_args+=(--require-complete-fundus)
  echo "[validate] ${PYTHON_BIN} ${validate_args[*]}"
  "$PYTHON_BIN" "${validate_args[@]}"
}

ensure_data() {
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Prepared data root not found: $DATA_ROOT" >&2
    echo "Run with --prepare and real G1020/ORIGA GT mask paths, or set --data-root." >&2
    exit 2
  fi
}

invoke_fedclam() {
  local dataset="$1"; shift
  local method="$1"; shift
  local seed="$1"; shift
  local mode="$1"; shift
  local notes="$1"; shift
  local loss="$1"; shift
  local inner_sites_csv="$1"; shift
  local outer_sites_csv="$1"; shift

  if run_done "$dataset" "$method" "$notes"; then
    echo "[skip] ${dataset}/${method}/${notes} already has metrics.npy"
    return 0
  fi

  IFS=',' read -r -a inner_sites <<< "$inner_sites_csv"
  IFS=',' read -r -a outer_sites <<< "$outer_sites_csv"

  local data_dir="${DATA_ROOT%/}/${dataset}/strict"
  local config="config.${dataset}.${method}"
  local args=(
    "${REPO_ROOT}/main.py"
    --config "$config"
    --dataset "$dataset"
    --data-dir "$data_dir"
    --output-dir "$OUTPUT_ROOT"
    --protocol strict
    --train_mode "$mode"
    --fl_rounds "$ROUNDS"
    --seed "$seed"
    --exp_name "$method"
    --run_notes "$notes"
    --loss "$loss"
    --inner_sites
  )
  args+=("${inner_sites[@]}")
  if [[ -n "$outer_sites_csv" ]]; then
    args+=(--outer_sites)
    args+=("${outer_sites[@]}")
  fi
  [[ "$RESUME" -eq 1 ]] && args+=(--resume)

  run_python "${dataset}_${method}_${notes}" "${args[@]}"
}

run_fundus_lodo() {
  local all_sites="Drishti-GS,RIM-ONE,REFUGE_t,REFUGE_v,REFUGE2,G1020,ORIGA"
  local outer method seed inner loss notes
  for outer in REFUGE2 G1020 ORIGA; do
    inner="$(without_site "$all_sites" "$outer")"
    for method in "${FUNDUS_METHODS[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        loss="dicefundus"
        [[ "$method" == "fedclam" ]] && loss="diceFIMfundus"
        notes="extended_fundus_lodo_${outer}_seed${seed}"
        invoke_fedclam fundus "$method" "$seed" innerouter "$notes" "$loss" "$inner" "$outer"
      done
    done
  done
}

run_fundus_all() {
  local inner="Drishti-GS,RIM-ONE,REFUGE_t,REFUGE_v,REFUGE2,G1020,ORIGA"
  local method seed loss notes
  for method in fedavg fedclam; do
    for seed in "${SEED_LIST[@]}"; do
      loss="dicefundus"
      [[ "$method" == "fedclam" ]] && loss="diceFIMfundus"
      notes="extended_fundus_all_seed${seed}"
      invoke_fedclam fundus "$method" "$seed" federated "$notes" "$loss" "$inner" ""
    done
  done
}

run_nuclei_lodo() {
  local all_sites="CoNSeP,CPM17,CryoNuSeg,TNBC"
  local outer method seed inner loss notes
  for outer in CoNSeP CPM17 CryoNuSeg TNBC; do
    inner="$(without_site "$all_sites" "$outer")"
    for method in "${NUCLEI_METHODS[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        loss="joint"
        [[ "$method" == "fedclam" ]] && loss="diceFIM"
        notes="extended_nuclei_lodo_${outer}_seed${seed}"
        invoke_fedclam nuclei "$method" "$seed" innerouter "$notes" "$loss" "$inner" "$outer"
      done
    done
  done
}

main() {
  cd "$REPO_ROOT"
  maybe_prepare
  ensure_data

  if [[ "$MODE" == "smoke" ]]; then
    FUNDUS_METHODS=(fedavg fedclam)
    NUCLEI_METHODS=(fedavg fedclam)
  else
    FUNDUS_METHODS=(fedavg fedprox feddyn fedsam fedclam)
    NUCLEI_METHODS=(fedavg fedprox feddyn fedsam fedclam)
  fi

  case "$MODE" in
    smoke)
      run_fundus_lodo
      run_nuclei_lodo
      ;;
    full)
      run_fundus_lodo
      run_fundus_all
      run_nuclei_lodo
      ;;
    fundus-lodo) run_fundus_lodo ;;
    fundus-all) run_fundus_all ;;
    nuclei-lodo) run_nuclei_lodo ;;
  esac

  wait_for_all_jobs
  "$PYTHON_BIN" "${REPO_ROOT}/tools/summarize_extended_results.py" \
    --output-root "$OUTPUT_ROOT" \
    --summary-dir "${REPO_ROOT}/outputs/analysis"
}

main "$@"
