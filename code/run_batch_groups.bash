#!/bin/bash

# Check number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_name> <port> <api_key>"
    echo "Example:"
    echo "  $0 Qwen/Qwen2.5-VL-3B-Instruct 8081 testkey1"
    exit 1
fi

MODEL_NAME=$1
PORT=$2
API_KEY=$3

BASE_URL="http://localhost:${PORT}/v1"

# List of group IDs to run
IDS=(42 22 2)

for ID in "${IDS[@]}"; do
    echo "Running for ID ${ID}..."
    bash scripts/batch_fetch_groups_methods_single.bash \
        "$MODEL_NAME" \
        "$BASE_URL" \
        "$API_KEY" \
        "$ID"
done

echo "All jobs finished."
