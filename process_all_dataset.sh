#!/bin/bash

DATASETS=(
    "2WikiMultiHopQA"
    "HotpotQA"
    "Musique"
    "NQ"
    "PopQA"
    "TriviaQA"
)

TOP_N_VALUES=(3 5)

API_SCRIPT="script_api.py"
PROCESSOR_SCRIPT="get_knowledge.py" 
API_URL="http://localhost:9001"      

API_BOOT_TIME=90                    
COOLDOWN_TIME=30                     

if [ ! -f "$PROCESSOR_SCRIPT" ]; then
    echo "[ERROR] Data processing script '$PROCESSOR_SCRIPT' not found. Please ensure it's in the current directory."
    exit 1
fi

for DATASET in "${DATASETS[@]}"; do
    echo "======================================================================"
    echo "======> [START] Processing Dataset: $DATASET"
    echo "======================================================================"

    echo "[INFO] Starting API script for '$DATASET'..."
    nohup python -u "$API_SCRIPT" --data_source "$DATASET" > "log_api_${DATASET}.log" 2>&1 &
    API_PID=$!
    echo "[INFO] API script for '$DATASET' started with PID: $API_PID"

    echo "[INFO] Waiting $API_BOOT_TIME seconds for API to initialize..."
    sleep $API_BOOT_TIME

    for N in "${TOP_N_VALUES[@]}"; do
        echo "--------------------------------------------------------------"
        echo "-----> [RUN] Processing for top_n = $N"
        echo "--------------------------------------------------------------"

        LOG_FILE="log_processor_${DATASET}_n${N}.log"
        echo "[INFO] Running data processor '$PROCESSOR_SCRIPT' for '$DATASET' with top_n=$N. Log: $LOG_FILE"
        python "$PROCESSOR_SCRIPT" \
            --data_source "$DATASET" \
            --api_url "$API_URL" \
            --top_n "$N" > "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] Data processing for '$DATASET' with top_n=$N completed successfully."
        else
            echo "[ERROR] Data processing for '$DATASET' with top_n=$N failed. Check '$LOG_FILE' for details."
        fi
        echo ""
    done

    echo "[INFO] Shutting down API script with PID: $API_PID..."
    if kill -0 $API_PID 2>/dev/null; then
        kill -9 $API_PID
        echo "[INFO] API script (PID: $API_PID) has been terminated."
    else
        echo "[WARN] API script (PID: $API_PID) was not found. It might have crashed or finished early."
    fi
    
    echo "======> [END] Finished all 'top_n' processing for Dataset: $DATASET"

    echo "[INFO] Cooling down for $COOLDOWN_TIME seconds..."
    echo "======================================================================"
    echo ""
    sleep $COOLDOWN_TIME
done

echo "######################################################################"
echo "############ ALL DATASETS HAVE BEEN PROCESSED. SCRIPT END. #############"
echo "######################################################################"