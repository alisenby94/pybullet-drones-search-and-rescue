#!/bin/bash
# Run the LM server

# Check for conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    echo "Warning: No conda environment active. Consider running 'conda activate combine-sar'"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the server
echo "Starting LM Server on http://0.0.0.0:8000"
echo "Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
