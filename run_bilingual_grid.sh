#!/bin/sh
# =====================================================================
# Project ValiMeli — Joint Bilingual Dravidian (Tamil + Malayalam) Matrix
# =====================================================================
# Evaluates 1.0 Million combined pairs (500k Tamil + 500k Malayalam)
# testing cross-lingual cognate and Named Entity transfer.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXEC="./valimeli-env/bin/python3"
if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

LOG_FILE="scratch/valimeli/bilingual_benchmark.log"
mkdir -p scratch/valimeli

echo "====================================================================" | tee -a "$LOG_FILE"
echo " INITIATING JOINT BILINGUAL DRAVIDIAN MATRIX (1.0M Pairs Total)" | tee -a "$LOG_FILE"
echo " Python: ${PYTHON_EXEC} | Started: $(date)" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

printf "%s\n" \
  "Bilingual-A0" \
  "Bilingual-A1-MT" | while read -r arm; do
    
    [ -z "$arm" ] && continue
    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo " >>> Running Joint Bilingual Cell: ${arm} (500k Tamil + 500k Malayalam)..." | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    
    $PYTHON_EXEC src/valimeli-benchmark.py \
        --mode bilingual \
        --arm "$arm" \
        --epochs 8 \
        --batch_size 256 \
        --max_samples_per_lang 500000 2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
echo " BILINGUAL BENCHMARK COMPLETED SUCCESSFULLY! [$(date)]" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
