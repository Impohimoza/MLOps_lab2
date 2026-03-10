#!/bin/bash

set -e

LOG_FILE="pipeline_$(date +'%Y%m%d_%H%M%S').log"
START_TIME=$(date +%s)

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

log "ML Pipeline Started!"

log "Setting up virtual environment"
if [ ! -d "venv" ]; then
    python -m venv venv
    log "Virtual environment created"
else
    log "Virtual environment already exists"
fi

source venv/Scripts/activate

log "Installing dependencies from requirements.txt"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    log "Dependencies installed from requirements.txt"
else
    log "requirements.txt not found!"
    exit 1
fi

log "Creating datasets"
python data_creation.py

log "Preprocessing data"
python model_preprocessing.py

log "Training model"
python model_preparation.py | grep -E "MSE" | tee -a "$LOG_FILE"

if [ ! -f "models/model.pkl" ]; then
    log "Model file missing!"
    exit 1
fi
log "Model saved"

log "Testing model"
python model_testing.py | grep -E "MSE" | tee -a "$LOG_FILE"

DURATION=$(( $(date +%s) - START_TIME ))
log "Pipeline completed in ${DURATION}s!"

deactivate