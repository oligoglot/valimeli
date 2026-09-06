#!/bin/bash
# Runner script for ValiMeli-A4-PanIndic-MT Training Run on Apple Silicon MPS with caffeinate
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_EXEC="${SCRIPT_DIR}/valimeli-env/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

echo "================================================================================"
echo " 🚀 Launching ValiMeli-A4-PanIndic-MT Multi-Task Multilingual Training"
echo " 📚 Dataset: pulli/data/valimeli_a4_pan_indic_train.jsonl.gz (27.0M pairs)"
echo " 🧠 Architecture: 3.2M Edge Transformer + Multi-Task Phonotactic Head"
echo " 🌐 Target Scripts: 8 Indic Languages (ta, ml, te, kn, hi, bn, gu, mr)"
echo " ☕ Power: Apple Silicon MPS with caffeinate process lock"
echo "================================================================================"

caffeinate -dis "$PYTHON_EXEC" src/train_valimeli_a4_multitask.py 2>&1 | tee train_valimeli_a4.log
