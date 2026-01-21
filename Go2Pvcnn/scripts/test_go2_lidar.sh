#!/bin/bash

# Launch script for Go2 LiDAR sensor testing
# This script sets up the environment variables required for Isaac Sim and launches LiDAR test scripts

set -e  # Exit on error

# Print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --num_envs N         Number of parallel environments (default: 4)"
    echo "  --num_steps N        Number of simulation steps (default: 200)"
    echo "  --seed N             Random seed (default: 42)"
    echo "  --headless           Run in headless mode (no GUI)"
    echo "  --video              Record test videos"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic test with 4 environments"
    echo "  $0 --num_envs 4 --headless"
    echo ""
    echo "  # Longer test with visualization"
    echo "  $0 --num_envs 8 --num_steps 500"
    echo ""
    echo "  # Test with video recording"
    echo "  $0 --num_envs 4 --headless --video"
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

# Set default values
NUM_ENVS=4
NUM_STEPS=200
SEED=42
HEADLESS=""
VIDEO=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --video)
            VIDEO="--enable_cameras"
            shift
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# LiDAR test script path
TEST_SCRIPT="${PROJECT_ROOT}/scripts/test_lidar_config.py"

# Check if script exists
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "Error: Test script not found at $TEST_SCRIPT"
    exit 1
fi

# Display launch info
echo ""
echo "========================================"
echo "Go2 LiDAR Sensor Test"
echo "========================================"
echo "Conda env:      $CONDA_DEFAULT_ENV"
echo "Test script:    test_lidar_config.py"
echo "Environments:   $NUM_ENVS"
echo "Steps:          $NUM_STEPS"
echo "Seed:           $SEED"
echo "Headless:       $([ -n "$HEADLESS" ] && echo "Yes" || echo "No")"
echo "Video:          $([ -n "$VIDEO" ] && echo "Yes" || echo "No")"
echo "========================================"
echo ""

# Build command
CMD="python $TEST_SCRIPT --num_envs $NUM_ENVS --num_steps $NUM_STEPS --seed $SEED $HEADLESS $VIDEO"

echo "Running command:"
echo "$CMD"
echo ""

# Run the test
eval $CMD

# Check exit status
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ LiDAR test completed successfully"
else
    echo "✗ LiDAR test failed with exit code $EXIT_CODE"
fi
echo "========================================"

exit $EXIT_CODE
