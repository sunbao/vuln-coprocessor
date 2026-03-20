#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv311"
PYTHON_BIN="${ENV_DIR}/bin/python"
BASE_MODEL_PATH="${ROOT_DIR}/cache/hf/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
TRAIN_FILE="${ROOT_DIR}/data/processed/v2-frozen-2026-03-19-train.jsonl"
EVAL_FILE="${ROOT_DIR}/data/processed/v2-frozen-2026-03-19-validation.jsonl"
OUTPUT_DIR="${ROOT_DIR}/checkpoints/qwen25-7b-explain-v2-py311-tmux"
TRAIN_LOG="${ROOT_DIR}/artifacts/train_qwen25_v2_py311_tmux.log"
TB_LOG="${ROOT_DIR}/artifacts/tensorboard_v2_tmux.log"
TRAIN_SESSION="vuln-v2-train"
TB_SESSION="vuln-v2-tensorboard"
TB_PORT="${TENSORBOARD_PORT:-6006}"
TB_HOST="${TENSORBOARD_HOST:-0.0.0.0}"

usage() {
  cat <<EOF
Usage: scripts/run_v2_tmux.sh <command>

Commands:
  start        Start V2 training and TensorBoard in tmux sessions
  status       Show tmux session state, GPU state, and output files
  attach       Attach to the training tmux session
  attach-tb    Attach to the TensorBoard tmux session
  stop         Stop both tmux sessions if they exist

Session names:
  train:       ${TRAIN_SESSION}
  tensorboard: ${TB_SESSION}
EOF
}

require_tmux() {
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but not installed" >&2
    exit 1
  fi
}

require_runtime() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing python runtime: ${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
    echo "missing base model path: ${BASE_MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${TRAIN_FILE}" || ! -f "${EVAL_FILE}" ]]; then
    echo "missing train or eval dataset" >&2
    exit 1
  fi
}

tmux_has_session() {
  local name="$1"
  tmux has-session -t "${name}" 2>/dev/null
}

start_train() {
  if tmux_has_session "${TRAIN_SESSION}"; then
    echo "training session already exists: ${TRAIN_SESSION}"
    return
  fi
  mkdir -p "${ROOT_DIR}/artifacts" "${OUTPUT_DIR}"
  local cmd
  cmd="cd ${ROOT_DIR} && exec env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} ${ROOT_DIR}/scripts/train_lora.py \
--model_name_or_path ${BASE_MODEL_PATH} \
--train_file ${TRAIN_FILE} \
--eval_file ${EVAL_FILE} \
--output_dir ${OUTPUT_DIR} \
--run_name qwen25-7b-explain-v2-py311-tmux \
--load_in_4bit \
--fp16 \
--gradient_checkpointing \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 2e-4 \
--num_train_epochs 2 \
--logging_steps 10 \
--save_steps 100 \
--eval_steps 100 \
--report_to tensorboard 2>&1 | tee -a ${TRAIN_LOG}"
  tmux new-session -d -s "${TRAIN_SESSION}" "${cmd}"
  echo "started training session: ${TRAIN_SESSION}"
}

start_tensorboard() {
  if tmux_has_session "${TB_SESSION}"; then
    echo "tensorboard session already exists: ${TB_SESSION}"
    return
  fi
  mkdir -p "${ROOT_DIR}/artifacts"
  local cmd
  cmd="cd ${ROOT_DIR} && exec env TENSORBOARD_HOST=${TB_HOST} TENSORBOARD_PORT=${TB_PORT} ${ROOT_DIR}/scripts/run_tensorboard.sh ${OUTPUT_DIR} 2>&1 | tee -a ${TB_LOG}"
  tmux new-session -d -s "${TB_SESSION}" "${cmd}"
  echo "started tensorboard session: ${TB_SESSION}"
}

show_status() {
  echo "tmux sessions:"
  tmux ls 2>/dev/null | rg "${TRAIN_SESSION}|${TB_SESSION}" || true
  echo
  echo "gpu:"
  nvidia-smi || true
  echo
  echo "output files:"
  find "${OUTPUT_DIR}" -maxdepth 3 -type f 2>/dev/null | sort || true
}

stop_sessions() {
  tmux kill-session -t "${TRAIN_SESSION}" 2>/dev/null || true
  tmux kill-session -t "${TB_SESSION}" 2>/dev/null || true
  echo "stopped tmux sessions if they existed"
}

main() {
  require_tmux
  require_runtime

  case "${1:-}" in
    start)
      start_train
      start_tensorboard
      show_status
      ;;
    status)
      show_status
      ;;
    attach)
      exec tmux attach -t "${TRAIN_SESSION}"
      ;;
    attach-tb)
      exec tmux attach -t "${TB_SESSION}"
      ;;
    stop)
      stop_sessions
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "unknown command: ${1}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
