#!/bin/bash
set -e

export PYTHONPATH=$(pwd)

echo "Starting preprocessing pipeline..."

python -m pgha.pipeline.run_preprocessing

echo "Preprocessing completed."