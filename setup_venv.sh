#!/bin/bash

# Script to create a new virtual environment with torch1 requirements
# Usage: ./setup_venv.sh [environment_name]

set -e

# Default environment name
ENV_NAME=${1:-"multi-face-classifier"}

# Create venv directory if it doesn't exist
VENV_DIR="venv"
mkdir -p "$VENV_DIR"

# Full path to the virtual environment
FULL_ENV_PATH="$VENV_DIR/$ENV_NAME"

echo "Creating virtual environment: $FULL_ENV_PATH"

# Create virtual environment
python3 -m venv "$FULL_ENV_PATH"

# Activate virtual environment
source "$FULL_ENV_PATH/bin/activate"

echo "Virtual environment activated"

# Upgrade pip
pip install --upgrade pip

echo "Installing requirements..."

# Install requirements
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the environment, run: source $FULL_ENV_PATH/bin/activate"
echo "To deactivate, run: deactivate"