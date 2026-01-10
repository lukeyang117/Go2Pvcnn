#!/bin/bash

# Launch script for Go2 PVCNN training/testing
# This script sets up the environment variables required for Isaac Sim and launches scripts
# Supports both single-GPU and multi-GPU training

set -e  # Exit on error

# Print usage
print_usage() {
    echo "Usage: $0 PYTHON_SCRIPT [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  PYTHON_SCRIPT        Path to Python script to run (absolute or relative path)"
    echo ""
    echo "Options:"
    echo "  --num_envs N          Number of parallel environments (total across all GPUs)"
    echo "  --max_iterations N    Maximum training iterations"
    echo "  --headless           Run in headless mode (no GUI)"
    echo "  --video              Record training videos"
    echo "  --resume             Resume training from checkpoint"
    echo "  --load_run NAME      Name of run to resume from"
    echo "  --multi_gpu          Enable multi-GPU training (auto-detect available GPUs)"
    echo "  --gpus N             Number of GPUs to use (default: all available)"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Single GPU training (256 envs on GPU 1)"
    echo "  $0 train_go2_pvcnn.py --num_envs 256 --headless"
    echo ""
    echo "  # Multi-GPU training (512 envs split across 2 GPUs: 256 each)"
    echo "  $0 train_go2_pvcnn.py --num_envs 512 --headless --multi_gpu --gpus 2"
    echo ""
    echo "  # Multi-GPU with all available GPUs"
    echo "  $0 train_go2_pvcnn.py --num_envs 1024 --headless --multi_gpu"
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

# Remaining arguments are for the Python script
PYTHON_ARGS="$@"

# Resolve script path
if [[ -f "$PYTHON_SCRIPT" ]]; then
    # Absolute or relative path that exists
    FULL_SCRIPT_PATH="$(realpath "$PYTHON_SCRIPT")"
elif [[ -f "${PROJECT_ROOT}/scripts/${PYTHON_SCRIPT}" ]]; then
    # Script name in scripts directory
    FULL_SCRIPT_PATH="${PROJECT_ROOT}/scripts/${PYTHON_SCRIPT}"
else
    # Try as-is
    FULL_SCRIPT_PATH="$PYTHON_SCRIPT"
fi

# Check if script exists
if [[ ! -f "$FULL_SCRIPT_PATH" ]]; then
    echo "Error: Script not found at $FULL_SCRIPT_PATH"
    echo ""
    echo "Available scripts:"
    ls -1 "${PROJECT_ROOT}/scripts/"*.py 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check if multi-GPU mode is requested
MULTI_GPU=false
NUM_GPUS=""
for arg in $PYTHON_ARGS; do
    if [[ "$arg" == "--multi_gpu" ]]; then
        MULTI_GPU=true
    elif [[ "$arg" == "--gpus" ]]; then
        # Next argument will be the number
        NEXT_IS_GPUS=true
    elif [[ "$NEXT_IS_GPUS" == true ]]; then
        NUM_GPUS="$arg"
        NEXT_IS_GPUS=false
    fi
done

# Display launch info
echo ""
echo "========================================"
echo "Launching Go2 PVCNN Script"
echo "========================================"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Script:    $FULL_SCRIPT_PATH"

if [[ "$MULTI_GPU" == true ]]; then
    echo "Mode:      Multi-GPU Training"
    
    # Detect available GPUs
    if command -v nvidia-smi &> /dev/null; then
        TOTAL_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Available GPUs: $TOTAL_GPUS"
    else
        echo "Warning: nvidia-smi not found, assuming 2 GPUs"
        TOTAL_GPUS=2
    fi
    
    # Determine number of GPUs to use
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS=$TOTAL_GPUS
        echo "Using all $NUM_GPUS GPUs"
    else
        echo "Using $NUM_GPUS GPUs (requested)"
    fi
    
    # Remove --multi_gpu and --gpus from PYTHON_ARGS and add --distributed
    PYTHON_ARGS=$(echo "$PYTHON_ARGS" | sed 's/--multi_gpu//g' | sed 's/--gpus [0-9]*//g')
    PYTHON_ARGS="$PYTHON_ARGS --distributed"
    
    echo "Arguments: $PYTHON_ARGS"
    echo "========================================"
    echo ""
    
    # Use torch.distributed.run for multi-GPU (official Isaac Lab method)
    echo "Starting multi-GPU training with torch.distributed.run..."
    echo "Command: python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS $FULL_SCRIPT_PATH $PYTHON_ARGS"
    echo ""
    
    # Set GPU_OFFSET environment variable to skip GPU 0
    # Python code will read this and use GPUs starting from GPU_OFFSET
    export GPU_OFFSET=1
    echo "✓ GPU_OFFSET=$GPU_OFFSET (skipping GPU 0, using GPU 1, 2, 3...)"
    
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS "$FULL_SCRIPT_PATH" $PYTHON_ARGS
else
    echo "Mode:      Single-GPU Training"
    echo "Arguments: $PYTHON_ARGS"
    echo "========================================"
    echo ""
    
    # Run the script normally
    python "$FULL_SCRIPT_PATH" $PYTHON_ARGS
fi

echo ""
echo "========================================"
echo "Script execution completed"
echo "========================================"