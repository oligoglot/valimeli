#!/bin/sh
# =====================================================================
# Project ValiMeli (வலி–മെലി) — 3-Arm Comparative Benchmarking Grid
# =====================================================================
# Runs A0 (Baseline) vs A1 (ValiMeli Phonology) vs A2 (Morphology)
# across Tamil and Malayalam bidirectionally with full test evaluations.

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

LOG_FILE="scratch/valimeli/grid_run.log"
mkdir -p scratch/valimeli

echo "====================================================================" | tee -a "$LOG_FILE"
echo " INITIATING PROJECT VALIMELI 3-ARM COMPARATIVE BENCHMARK GRID" | tee -a "$LOG_FILE"
echo " Config: Epochs=${EPOCHS} | Batch Size=${BATCH_SIZE} | Train Samples=${MAX_TRAIN_SAMPLES}" | tee -a "$LOG_FILE"
echo " Python: ${PYTHON_EXEC} | Started: $(date)" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

RUN_COUNT=0
TOTAL_RUNS=12

# Format: lang arm direction
printf "%s\n" \
  "tam A0 indic-en" \
  "tam A1 indic-en" \
  "tam A2 indic-en" \
  "tam A0 en-indic" \
  "tam A1 en-indic" \
  "tam A2 en-indic" \
  "mal A0 indic-en" \
  "mal A1 indic-en" \
  "mal A2 indic-en" \
  "mal A0 en-indic" \
  "mal A1 en-indic" \
  "mal A2 en-indic" | while read -r lang arm dir; do
    
    [ -z "$lang" ] && continue
    RUN_COUNT=$((RUN_COUNT + 1))
    
    lang_upper=$(echo "$lang" | tr '[:lower:]' '[:upper:]')
    dir_upper=$(echo "$dir" | tr '[:lower:]' '[:upper:]')
    
    echo "" | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    echo " >>> [${RUN_COUNT}/${TOTAL_RUNS}] Executing ${lang_upper} ${arm} (${dir_upper})..." | tee -a "$LOG_FILE"
    echo "====================================================================" | tee -a "$LOG_FILE"
    
    $PYTHON_EXEC src/valimeli-benchmark.py \
        --lang "$lang" \
        --arm "$arm" \
        --direction "$dir" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --max_train_samples "$MAX_TRAIN_SAMPLES" 2>&1 | tee -a "$LOG_FILE"
        
    $PYTHON_EXEC src/plot_results.py || true
    $PYTHON_EXEC src/plot_attention.py || true
    cp scratch/valimeli/*.png artifacts/ 2>/dev/null || true
done

echo "" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"
echo " ALL 12 EXPERIMENTAL CELLS COMPLETED SUCCESSFULLY! [$(date)]" | tee -a "$LOG_FILE"
echo "====================================================================" | tee -a "$LOG_FILE"

$PYTHON_EXEC src/monitor_progress.py | tee -a "$LOG_FILE"
