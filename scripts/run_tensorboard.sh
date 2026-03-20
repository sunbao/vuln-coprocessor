#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ROOT_DIR}/.venv311"
DEFAULT_LOGDIR="${ROOT_DIR}/checkpoints"
HOST="${TENSORBOARD_HOST:-0.0.0.0}"
PORT="${TENSORBOARD_PORT:-6006}"

if [[ ! -x "${ENV_DIR}/bin/tensorboard" ]]; then
  echo "tensorboard is not installed in ${ENV_DIR}" >&2
  echo "install with: ${ENV_DIR}/bin/pip install tensorboard" >&2
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: scripts/run_tensorboard.sh [logdir]

Environment variables:
  TENSORBOARD_HOST   default: 0.0.0.0
  TENSORBOARD_PORT   default: 6006
EOF
  exit 0
fi

LOGDIR="${1:-${DEFAULT_LOGDIR}}"

exec "${ENV_DIR}/bin/tensorboard" \
  --logdir "${LOGDIR}" \
  --host "${HOST}" \
  --port "${PORT}"
