#!/bin/sh
# =====================================================================
# Project ValiMeli — 11M Transformer Benchmark Runner (IndicXlit Parity)
# =====================================================================
# Evaluates the 11.0M parameter Transformer architecture (6 Enc + 6 Dec)
# across Baseline (A0) and ValiMeli (A1) on Tamil and Malayalam (en-indic)
# with full holdout test set evaluations.

set -e

EPOCHS=8
BATCH_SIZE=256
MAX_TRAIN_SAMPLES=250000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="./valimeli-env/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

LOG_FILE="scratch/valimeli/transformer_run.log"
mkdir -p scratch/valimeli

echo "====================================================================" | tee -a "$LOG_FILE"
echo " INITIATING 11M PARAMETER TRANSFORMER BENCHMARK (IndicXlit Architecture)" | tee -a "$LOG_FILE"
echo " Config: Epochs=${EPOCHS} | Batch Size=${BATCH_SIZE} | Train Samples=${MAX_TRAIN_SAMPLES}" | tee -a "$LOG_FILE"
echo " Python: ${PYTHON_EXEC} | Started: $(date)" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

printf "%s\n" \
  "tam A0 en-indic" \
  "tam A1 en-indic" \
  "mal A0 en-indic" \
  "mal A1 en-indic" | while read -r lang arm dir; do
    
    [ -z "$lang" ] && continue
    lang_upper=$(echo "$lang" | tr '[:lower:]' '[:upper:]')
    dir_upper=$(echo "$dir" | tr '[:lower:]' '[:upper:]')
    
    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo " >>> Executing 11M Transformer: ${lang_upper} ${arm} (${dir_upper})..." | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    
    $PYTHON_EXEC src/valimeli-benchmark.py \
        --lang "$lang" \
        --arm "$arm" \
        --direction "$dir" \
        --model_type "transformer" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --max_train_samples "$MAX_TRAIN_SAMPLES" 2>&1 | tee -a "$LOG_FILE"
        
done

echo "" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
echo " 11M TRANSFORMER BENCHMARK COMPLETED SUCCESSFULLY! [$(date)]" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
