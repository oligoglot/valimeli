#!/bin/sh
# =====================================================================
# Project ValiMeli — Multi-Task Phonology Architecture Matrix
# =====================================================================
# Evaluates:
# 1. Low-Resource Regime (25k pairs) to test inductive bias under data sparsity
# 2. Full 500k Scale Matrix with Multi-Task Auxiliary Loss

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="./valimeli-env/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

LOG_FILE="scratch/valimeli/multitask_benchmark.log"
mkdir -p scratch/valimeli

echo "====================================================================" | tee -a "$LOG_FILE"
echo " INITIATING MULTI-TASK PHONOLOGY ARCHITECTURE BENCHMARK" | tee -a "$LOG_FILE"
echo " Python: ${PYTHON_EXEC} | Started: $(date)" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

# --- PHASE 1: LOW-RESOURCE REGIME (25,000 Samples) ---
echo "" | tee -a "$LOG_FILE"
echo ">>> [PHASE 1] LOW-RESOURCE REGIME (25k Samples per cell)..." | tee -a "$LOG_FILE"

printf "%s\n" \
  "tam A0 en-indic 25000" \
  "tam A1-MT en-indic 25000" \
  "mal A0 en-indic 25000" \
  "mal A1-MT en-indic 25000" | while read -r lang arm dir samples; do
    
    [ -z "$lang" ] && continue
    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo " >>> Running Low-Resource Cell: ${lang} ${arm} (${dir}, ${samples} samples)..." | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    
    $PYTHON_EXEC src/valimeli-benchmark.py \
        --lang "$lang" \
        --arm "$arm" \
        --direction "$dir" \
        --epochs 8 \
        --batch_size 256 \
        --aux_lambda 0.3 \
        --max_train_samples "$samples" 2>&1 | tee -a "$LOG_FILE"
done

# --- PHASE 2: FULL-SCALE 500k REGIME ---
echo "" | tee -a "$LOG_FILE"
echo ">>> [PHASE 2] FULL-SCALE REGIME (500k Samples per cell)..." | tee -a "$LOG_FILE"

printf "%s\n" \
  "tam A1-MT en-indic 500000" \
  "mal A1-MT en-indic 500000" | while read -r lang arm dir samples; do
    
    [ -z "$lang" ] && continue
    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo " >>> Running Full-Scale Cell: ${lang} ${arm} (${dir}, ${samples} samples)..." | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    
    $PYTHON_EXEC src/valimeli-benchmark.py \
        --lang "$lang" \
        --arm "$arm" \
        --direction "$dir" \
        --epochs 8 \
        --batch_size 256 \
        --aux_lambda 0.3 \
        --max_train_samples "$samples" 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
echo " MULTI-TASK BENCHMARK COMPLETED SUCCESSFULLY! [$(date)]" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
