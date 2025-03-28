#!/bin/bash

# Usage: ./scripts/read_lag.sh path/to/file.lag

FILE_PATH="$1"

if [ -z "$FILE_PATH" ]; then
    echo "Please provide a .lag file path."
    echo "Usage: ./scripts/read_lag.sh path/to/file.lag"
    exit 1
fi

echo "Processing: $FILE_PATH"
poetry run python scripts/run_data_reader.py "$FILE_PATH"