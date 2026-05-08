#!/bin/bash

# Uso:
#   ./scripts/run_data_reader.sh path/to/file.lag
#   ./scripts/run_data_reader.sh path/to/dir/

PATH_ARG="$1"

if [ -z "$PATH_ARG" ]; then
    echo "Por favor pasa un archivo .lag o un directorio."
    echo "Uso: ./scripts/run_data_reader.sh <archivo.lag | directorio>"
    exit 1
fi

echo "Target: $PATH_ARG"
python scripts/run_data_reader.py "$PATH_ARG"
