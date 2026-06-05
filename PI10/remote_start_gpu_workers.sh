#!/usr/bin/env bash
set -euo pipefail

# Starts one detached screen worker per GPU on the remote GPU server.
# Run from /data/woutdecrop/projects/planktonclass/PI10, or set PI10_REMOTE_PI10_DIR.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PI10_REMOTE_PI10_DIR:-$SCRIPT_DIR}"
ENV_DIR="${PI10_REMOTE_ENV_DIR:-/data/woutdecrop/envs/planktonclass-gpu}"
LOG_DIR="${PI10_REMOTE_LOG_DIR:-$PROJECT_DIR/run_logs}"
GPUS="${PI10_GPUS:-0 1}"
EMAIL_GPU="${PI10_EMAIL_GPU:-0}"
SLEEP_SECONDS="${PI10_SLEEP_SECONDS:-3600}"
STALE_LOCK_HOURS="${PI10_STALE_LOCK_HOURS:-72}"
QARCHIVE_ROOT="${PI10_QARCHIVE_ROOT:-/mnt/qarchive_data_sensors}"
QARCHIVE_CHECK_DIR="${PI10_QARCHIVE_CHECK_DIR:-$QARCHIVE_ROOT/plankton-imager-10}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is not installed or not on PATH."
  exit 1
fi

if [[ -d "$QARCHIVE_ROOT" ]]; then
  if mountpoint -q "$QARCHIVE_ROOT"; then
    echo "qarchive is mounted."
  else
    echo "qarchive is not mounted; trying non-interactive mount."
    if sudo -n mount "$QARCHIVE_ROOT" 2>/dev/null; then
      echo "qarchive mounted."
    else
      echo "qarchive still not mounted. If needed, run once on the server:"
      echo "  cifscreds add -u wout.decrop -d vliz.be"
      echo "  sudo mount $QARCHIVE_ROOT/"
    fi
  fi
fi

if [[ ! -r "$QARCHIVE_CHECK_DIR" || ! -x "$QARCHIVE_CHECK_DIR" ]]; then
  echo "Cannot read qarchive path: $QARCHIVE_CHECK_DIR"
  echo "Run these on the server, then start the workers again:"
  echo "  cifscreds add -u wout.decrop -d vliz.be"
  echo "  sudo mount $QARCHIVE_ROOT/"
  exit 1
fi

for gpu in $GPUS; do
  session="predict_gpu_gpu${gpu}"
  log_file="$LOG_DIR/${session}.log"
  email_arg=""

  if [[ "$gpu" != "$EMAIL_GPU" ]]; then
    email_arg="--disable-email"
  fi

  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "Screen $session is already running; leaving it alone."
    continue
  fi

  echo "Starting $session on GPU $gpu; log: $log_file"
  screen -dmS "$session" bash -lc "
    exec > >(tee -a '$log_file') 2>&1
    set -e
    cd '$PROJECT_DIR'
    source '$ENV_DIR/bin/activate'
    export CUDA_VISIBLE_DEVICES='$gpu'
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    exec python -u predict_gpu_remote.py \
      --worker-id 'gpu$gpu' \
      --gpu '$gpu' \
      --sleep-seconds '$SLEEP_SECONDS' \
      --stale-lock-hours '$STALE_LOCK_HOURS' \
      $email_arg
  "
done

echo
screen -ls || true
