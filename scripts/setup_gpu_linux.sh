#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="${1:-$HOME/planktonclass-gpu}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Creating Linux GPU environment at: ${ENV_DIR}"
"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[gpu]"

echo
echo "Verifying TensorFlow GPU visibility..."
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"

echo
echo "Environment ready."
echo "Activate with: source ${ENV_DIR}/bin/activate"
