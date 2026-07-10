#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python3"
DATASET="prostate"
DATA_DIR=""
OUTPUT_DIR="outputs"
ROUNDS="200"
LOCAL_EPOCHS="1"
LR=""
BATCH_SIZE=""
LOSS=""
L_FIM=""
FIM_WARMUP=""
PROTOCOL="official"
RUN_PREFIX="backbone"
DRY_RUN=0

usage() {
  cat <<'EOF'
Run FedCLAM backbone ablations for fundus or prostate.

Options:
  --dataset NAME        fundus | prostate | prostate_mri. Default: prostate
  --data-dir PATH       Preprocessed dataset root
  --output-dir PATH     Output root. Default: outputs
  --rounds N            FL rounds. Default: 200
  --local-epochs N      Local epochs per round. Default: 1
  --lr VALUE            Override learning rate
  --batch-size N        Override batch size
  --loss NAME           Override loss. Defaults: fundus=diceFIMfundus, prostate=diceFIMce
  --l-fim VALUE         Override lambda FIM. Defaults: fundus=0.01, prostate=0.1
  --fim-warmup N        Override FIM warmup. Defaults: fundus=10, prostate=100
  --protocol NAME       official | strict. Default: official
  --run-prefix NAME     Experiment name prefix. Default: backbone
  --python PATH         Python executable. Default: python3
  --dry-run             Print commands without running
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --local-epochs) LOCAL_EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --loss) LOSS="$2"; shift 2 ;;
    --l-fim) L_FIM="$2"; shift 2 ;;
    --fim-warmup) FIM_WARMUP="$2"; shift 2 ;;
    --protocol) PROTOCOL="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$DATASET" in
  fundus)
    CONFIG_DATASET="fundus"
    RUNTIME_DATASET="fundus"
    DEFAULT_LOSS="diceFIMfundus"
    DEFAULT_L_FIM="0.01"
    DEFAULT_FIM_WARMUP="10"
    ;;
  prostate|prostate_mri)
    CONFIG_DATASET="prostate_mri"
    RUNTIME_DATASET="prostate"
    DEFAULT_LOSS="diceFIMce"
    DEFAULT_L_FIM="0.1"
    DEFAULT_FIM_WARMUP="100"
    ;;
  *)
    echo "--dataset must be one of: fundus, prostate, prostate_mri" >&2
    exit 2
    ;;
esac

LOSS="${LOSS:-$DEFAULT_LOSS}"
L_FIM="${L_FIM:-$DEFAULT_L_FIM}"
FIM_WARMUP="${FIM_WARMUP:-$DEFAULT_FIM_WARMUP}"

CONFIGS=(
  "E0|baseline_unet|config.${CONFIG_DATASET}.fedclam"
  "E1|resenc|config.${CONFIG_DATASET}.fedclam_resenc"
  "E2|resenc_ds|config.${CONFIG_DATASET}.fedclam_resenc_ds"
  "E3|resenc_scse|config.${CONFIG_DATASET}.fedclam_resenc_scse"
  "E4|resenc_ag|config.${CONFIG_DATASET}.fedclam_resenc_ag"
  "E5|resattunet|config.${CONFIG_DATASET}.fedclam_resattunet"
)

for item in "${CONFIGS[@]}"; do
  IFS='|' read -r id name config_module <<< "$item"
  cmd=(
    "$PYTHON_BIN" main.py
    --dataset "$RUNTIME_DATASET"
    --config "$config_module"
    --exp_name "${RUN_PREFIX}_${id}_${name}"
    --run_notes "$name"
    --protocol "$PROTOCOL"
    --output-dir "$OUTPUT_DIR"
    --fl_rounds "$ROUNDS"
    --local_epochs "$LOCAL_EPOCHS"
    --loss "$LOSS"
    --l_fim "$L_FIM"
    --fim_warmup "$FIM_WARMUP"
  )
  [[ -n "$DATA_DIR" ]] && cmd+=(--data-dir "$DATA_DIR")
  [[ -n "$LR" ]] && cmd+=(--lr "$LR")
  [[ -n "$BATCH_SIZE" ]] && cmd+=(--batch_size "$BATCH_SIZE")

  echo "Running ${id}: ${config_module} dataset=${RUNTIME_DATASET} loss=${LOSS} l_fim=${L_FIM} fim_warmup=${FIM_WARMUP}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  %q' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
done
