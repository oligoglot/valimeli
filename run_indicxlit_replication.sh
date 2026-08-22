#!/bin/sh
# =====================================================================
# Project ValiMeli — IndicXlit Exact Replication Runner (500k Matrix)
# =====================================================================
# Evaluates the 11.0M Transformer with Unigram LM Rescoring and
# partitioned reporting on Native Words (Dakshina + AK-Freq) vs Named Entities (AK-NEI/NEF).

set -e

EPOCHS=8
BATCH_SIZE=256
MAX_TRAIN_SAMPLES=500000
LM_WEIGHT=0.5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="./valimeli-env/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

LOG_FILE="scratch/valimeli/indicxlit_replication.log"
mkdir -p scratch/valimeli

echo "====================================================================" | tee -a "$LOG_FILE"
echo " INITIATING EXACT INDICXLIT REPLICATION MATRIX (500,000 Samples/Cell)" | tee -a "$LOG_FILE"
echo " Config: Epochs=${EPOCHS} | Batch Size=${BATCH_SIZE} | LM Weight=${LM_WEIGHT}" | tee -a "$LOG_FILE"
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
    echo " >>> Running Replication Cell: ${lang_upper} ${arm} (${dir_upper})..." | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    
    $PYTHON_EXEC src/valimeli-benchmark.py \
        --lang "$lang" \
        --arm "$arm" \
        --direction "$dir" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lm_weight "$LM_WEIGHT" \
        --max_train_samples "$MAX_TRAIN_SAMPLES" 2>&1 | tee -a "$LOG_FILE"
        
done

echo "" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
echo " INDICXLIT REPLICATION BENCHMARK COMPLETED SUCCESSFULLY! [$(date)]" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
