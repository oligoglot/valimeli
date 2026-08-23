#!/usr/bin/env bash
# Project ValiMeli — Option C: Large-Scale Multilingual Pre-training Run (Overnight)
# Trains 11.5M Multi-Script Transformer across 8 Indic Languages (~3.5M pairs)

set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

mkdir -p scratch/valimeli
LOG_FILE="scratch/valimeli/multilingual_scaling.log"

echo "================================================================================"
echo " Starting Large-Scale Multilingual Pre-training Run (Option C) Overnight"
echo " Log file: $LOG_FILE"
echo "================================================================================"

./valimeli-env/bin/python3 src/multilingual_scaling_benchmark.py > "$LOG_FILE" 2>&1
