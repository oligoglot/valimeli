#!/usr/bin/env bash
# Project ValiMeli — Large-Scale Multilingual Multi-Task Pre-training Run (A1-MT)
# 8 Indic Languages (3.2M pairs) + Auxiliary Dravidian Stop-Voicing Head

set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

mkdir -p scratch/valimeli
LOG_FILE="scratch/valimeli/multilingual_multitask_scaling.log"

echo "================================================================================"
echo " Starting Multilingual Multi-Task Pre-training Run (Multilingual-A1-MT)"
echo " Log file: $LOG_FILE"
echo "================================================================================"

./valimeli-env/bin/python3 src/multilingual_multitask_scaling_benchmark.py > "$LOG_FILE" 2>&1
