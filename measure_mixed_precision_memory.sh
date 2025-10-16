#!/bin/bash

# Measure memory usage with and without mixed precision for 2.7B model

echo "========================================"
echo "Mixed Precision Memory Comparison"
echo "2.7B Model, Context Length = 256"
echo "========================================"
echo ""

D_MODEL=2560
NUM_LAYERS=32
NUM_HEADS=32
D_FF=10240
CTX_LEN=256

echo "=========================================="
echo "1. Forward-only with FP32"
echo "=========================================="
uv run benchmarking.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --d-ff $D_FF \
    --context-length $CTX_LEN \
    --warmup-steps 3 \
    --measurement-steps 5 \
    --measure-forward-only True

echo ""
echo "=========================================="
echo "2. Forward-only with BF16"
echo "=========================================="
uv run benchmarking.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --d-ff $D_FF \
    --context-length $CTX_LEN \
    --warmup-steps 3 \
    --measurement-steps 5 \
    --measure-forward-only True \
    --use-mixed-precision

echo ""
echo "=========================================="
echo "3. Full training with FP32"
echo "=========================================="
uv run benchmarking.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --d-ff $D_FF \
    --context-length $CTX_LEN \
    --warmup-steps 3 \
    --measurement-steps 5

echo ""
echo "=========================================="
echo "4. Full training with BF16"
echo "=========================================="
uv run benchmarking.py \
    --d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --d-ff $D_FF \
    --context-length $CTX_LEN \
    --warmup-steps 3 \
    --measurement-steps 5 \
    --use-mixed-precision

echo ""
echo "========================================"
echo "Summary: Compare the 'Peak Memory Reserved' values above"
echo "========================================"
