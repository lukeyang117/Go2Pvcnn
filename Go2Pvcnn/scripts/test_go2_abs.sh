#!/bin/bash

# Launch script for ABS model testing
# This script sets up the environment variables required for Isaac Sim and launches test scripts

set -e  # Exit on error

# Print usage
print_usage() {
    echo "Usage: $0 PYTHON_SCRIPT [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  PYTHON_SCRIPT        Path to Python script to run (absolute or relative path)"
    echo ""
    echo "Options:"
    echo "  --checkpoint PATH     Path to ABS model checkpoint"
    echo "  --num_envs N         Number of parallel environments"
    echo "  --num_steps N        Number of test steps"
    echo "  --goal_distance D    Goal distance in meters"
    echo "  --goal_threshold T   Success threshold in meters"
    echo "  --headless          Run in headless mode (no GUI)"
    echo "  --save_results      Save results to YAML file"
    echo "  --device cuda:N     Device to use (default: cuda:0)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Test ABS model"
    echo "  $0 scripts/test_go2_abs_collision.py --checkpoint other_model/ABS/model_4000.pt --num_envs 4 --headless"
    echo ""
}

# Initialize conda for bash
echo "Initializing conda..."
# Try to find conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh, trying to activate anyway..."
fi

# Activate conda environment
echo "Activating conda environment: env_isaacsim..."
conda activate env_isaacsim || {
    echo "Error: Failed to activate conda environment 'env_isaacsim'"
    echo "Please make sure the environment exists: conda env list"
    exit 1
}

echo "✓ Conda environment activated: $CONDA_DEFAULT_ENV"

# Set up Isaac Sim environment variables
echo ""
echo "Setting up Isaac Sim environment variables..."

# Unset and reset GLX vendor library (required for Isaac Sim rendering)
unset __GLX_VENDOR_LIBRARY_NAME
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Add NVIDIA library path
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH

echo "✓ Environment variables set:"
echo "  __GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Get project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo ""
echo "✓ Project root: $PROJECT_ROOT"

# Add Go2Pvcnn to Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
echo "✓ PYTHONPATH updated"

# Check if help is requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

# First argument should be the Python script path
if [[ $# -eq 0 ]]; then
    echo "Error: No Python script specified!"
    echo ""
    print_usage
    exit 1
fi

PYTHON_SCRIPT="$1"
shift  # Remove the script path from arguments

# Get full path to Python script
if [[ "$PYTHON_SCRIPT" = /* ]]; then
    # Absolute path
    FULL_SCRIPT_PATH="$PYTHON_SCRIPT"
else
    # Relative path - check if it's relative to current dir or project root
    if [[ -f "$PYTHON_SCRIPT" ]]; then
        # File exists relative to current directory
        FULL_SCRIPT_PATH="$(realpath "$PYTHON_SCRIPT")"
    elif [[ -f "${PROJECT_ROOT}/${PYTHON_SCRIPT}" ]]; then
        # File exists relative to project root
        FULL_SCRIPT_PATH="${PROJECT_ROOT}/${PYTHON_SCRIPT}"
    else
        # Try removing Go2Pvcnn prefix if it exists
        SCRIPT_WITHOUT_PREFIX="${PYTHON_SCRIPT#Go2Pvcnn/}"
        if [[ -f "${PROJECT_ROOT}/${SCRIPT_WITHOUT_PREFIX}" ]]; then
            FULL_SCRIPT_PATH="${PROJECT_ROOT}/${SCRIPT_WITHOUT_PREFIX}"
        else
            FULL_SCRIPT_PATH="${PROJECT_ROOT}/${PYTHON_SCRIPT}"
        fi
    fi
fi

# Check if script exists
if [[ ! -f "$FULL_SCRIPT_PATH" ]]; then
    echo "Error: Python script not found: $FULL_SCRIPT_PATH"
    echo "Tried paths:"
    echo "  - $PYTHON_SCRIPT"
    echo "  - ${PROJECT_ROOT}/${PYTHON_SCRIPT}"
    exit 1
fi

# Collect remaining arguments
PYTHON_ARGS="$@"

echo ""
echo "========================================"
echo "ABS Model Testing"
echo "========================================"
echo "Script:    $FULL_SCRIPT_PATH"
echo "Arguments: $PYTHON_ARGS"
echo "========================================"
echo ""

# Run the script
python "$FULL_SCRIPT_PATH" $PYTHON_ARGS

echo ""
echo "========================================"
echo "Script execution completed"
echo "========================================"
