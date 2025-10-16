#!/bin/bash

# Script to run benchmarking across different model sizes
# Usage: ./run_benchmarks.sh [--profile] [--forward-only] [--context-length N] [--use-mixed-precision]
# --profile: Run with nsys profiling
# --forward-only: Measure forward pass only (skip backward pass)
# --context-length N: Override context length (default: 256)
# --use-mixed-precision: Enable BF16 mixed precision training

PROFILE=false
FORWARD_ONLY=false
CONTEXT_LENGTH=256
USE_MIXED_PRECISION=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE=true
            echo "Running with nsys profiling enabled"
            shift
            ;;
        --forward-only)
            FORWARD_ONLY=true
            echo "Running forward pass only (no backward)"
            shift
            ;;
        --context-length)
            CONTEXT_LENGTH="$2"
            echo "Using context length: $CONTEXT_LENGTH"
            shift 2
            ;;
        --use-mixed-precision)
            USE_MIXED_PRECISION=true
            echo "Running with mixed precision (BF16) enabled"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Define model configurations
# Format: name d_model d_ff num_layers num_heads
declare -a CONFIGS=(
    "small 768 3072 12 12"
    "medium 1024 4096 24 16"
    "large 1280 5120 36 20"
    "xl 1600 6400 48 25"
    "2.7B 2560 10240 32 32"
)

# Create results directory based on context length
RESULTS_DIR="benchmark_results"
if [ "$CONTEXT_LENGTH" != "256" ]; then
    RESULTS_DIR="benchmark_results_ctx${CONTEXT_LENGTH}"
fi
mkdir -p "$RESULTS_DIR"

# Loop through each configuration
for config in "${CONFIGS[@]}"; do
    read -r name d_model d_ff num_layers num_heads <<< "$config"

    echo "=========================================="
    echo "Benchmarking: $name (context_length=$CONTEXT_LENGTH)"
    echo "d_model=$d_model, d_ff=$d_ff, num_layers=$num_layers, num_heads=$num_heads, context_length=$CONTEXT_LENGTH"
    echo "=========================================="

    # Build common arguments
    COMMON_ARGS="--d-model $d_model --d-ff $d_ff --num-layers $num_layers --num-heads $num_heads --context-length $CONTEXT_LENGTH --warmup-steps 5 --measurement-steps 10"

    if [ "$FORWARD_ONLY" = true ]; then
        COMMON_ARGS="$COMMON_ARGS --measure-forward-only True"
    fi

    if [ "$USE_MIXED_PRECISION" = true ]; then
        COMMON_ARGS="$COMMON_ARGS --use-mixed-precision"
    fi

    if [ "$PROFILE" = true ]; then
        # Run with nsys profiling
        uv run nsys profile \
            --trace=cuda,nvtx \
            --output="${RESULTS_DIR}/${name}_profile.nsys-rep" \
            --force-overwrite=true \
            python3 benchmarking.py $COMMON_ARGS
    else
        # Run without profiling
        uv run benchmarking.py $COMMON_ARGS \
            2>&1 | tee "${RESULTS_DIR}/${name}_results.txt"
    fi

    echo ""
    echo "Completed: $name"
    echo ""
done

echo "=========================================="
echo "All benchmarks completed!"
echo "Results saved in: $RESULTS_DIR/"
echo "=========================================="
