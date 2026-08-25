#!/usr/bin/env bash
# Project Pulli / ValiMeli — Launch ValiMeli-A3-MT Training on Grounded Dataset
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="$SCRIPT_DIR/valimeli-env/bin/python3"
LOG_FILE="$SCRIPT_DIR/scratch/valimeli/train_valimeli_a3.log"
mkdir -p "$SCRIPT_DIR/scratch/valimeli/runs"

echo "================================================================================"
echo " 🚀 Launching ValiMeli-A3-MT Grounded Multi-Task Training Run"
echo " 📝 Logging output to: $LOG_FILE"
echo "================================================================================"

$PYTHON_EXEC -u src/train_valimeli_a3_multitask.py 2>&1 | tee "$LOG_FILE"
