#!/bin/bash
# ==============================================================================
# Project ValiMeli & Project Pulli — Overnight Auto-Training Pipeline
# Monitors ValiMeli-A3 training completion and immediately launches ValiMeli-A4.
# ==============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_EXEC="${SCRIPT_DIR}/valimeli-env/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

echo "================================================================================"
echo " 🌙 OVERNIGHT AUTO-TRAINING PIPELINE INITIALIZED"
echo " 💻 System: Apple Silicon MPS with caffeinate process lock"
echo " 📊 Disk Space Check:"
df -h /
echo "================================================================================"

# 1. Wait for active ValiMeli-A3-MT process to complete
echo " -> Monitoring active ValiMeli-A3-MT training process..."
while pgrep -f "train_valimeli_a3_multitask.py" > /dev/null; do
    echo "    [$(date '+%Y-%m-%d %H:%M:%S')] ValiMeli-A3 is running... sleeping 60s."
    sleep 60
done

echo " ⭐ ValiMeli-A3-MT process completed successfully!"

# 2. Verify disk space before launching A4
FREE_KB=$(df -k / | tail -1 | awk '{print $4}')
if [ "$FREE_KB" -lt 20000000 ]; then # Require at least 20 GB
    echo " ❌ ERROR: Less than 20GB disk space remaining ($FREE_KB KB). Aborting A4 launch."
    exit 1
fi

echo " ✅ Disk space verified ($((FREE_KB / 1024 / 1024)) GB free)."

# 3. Launch Flagship ValiMeli-A4-PanIndic Training Run (27.0M pairs, 3 epochs)
echo "================================================================================"
echo " 🚀 LAUNCHING VALIMELI-A4-PAN-INDIC TRAINING (27.0M PAIRS)"
echo " 🧠 Dataset: pulli/data/valimeli_a4_pan_indic_train.jsonl.gz"
echo " ⚙️ Hyperparameters: 3 Epochs, LR 6e-4 (2k Warmup + Cosine Decay)"
echo "================================================================================"

caffeinate -dis "$PYTHON_EXEC" src/train_valimeli_a4_multitask.py 2>&1 | tee train_valimeli_a4.log

echo "================================================================================"
echo " 🏆 OVERNIGHT PIPELINE COMPLETE: VALIMELI-A4 TRAINED & CHECKPOINT SYNCED!"
echo "================================================================================"
