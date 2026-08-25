#!/usr/bin/env bash
# Project ValiMeli — Train ValiMeli-A2-MT with Vowel-Collapse Data Augmentation
# Preserves ValiMeli-A1-MT checkpoint completely intact.

set -e
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

mkdir -p scratch/valimeli
LOG_FILE="scratch/valimeli/train_valimeli_a2.log"

echo "================================================================================"
echo " Starting ValiMeli-A2-MT Vowel-Collapse Augmented Training"
echo " Preserving ValiMeli-A1-MT Checkpoint • Logging to: $LOG_FILE"
echo "================================================================================"

exec ./valimeli-env/bin/python3 src/train_valimeli_a2_multitask.py "$@"
