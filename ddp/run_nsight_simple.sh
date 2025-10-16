#!/bin/bash

# Simple Nsight Systems profiling without stats generation
# This will profile the bucket-based DDP to visualize compute-communication overlap

# Create output directory
mkdir -p ./nsight_profiles

# Run with nsys profiler (simple version)
nsys profile \
    --trace=cuda,nvtx \
    --output=./nsight_profiles/ddp_bucket_simple \
    --force-overwrite=true \
    uv run python ddp/ddp_train.py \
    --mode benchmark \
    --use_bucket \
    --bucket_size_mb 1 \
    --enable_profiling \
    --d_model 512 \
    --num_layers 8 \
    --num_heads 8 \
    --d_ff 2048 \
    --batch_size 2 \
    --context_length 128 \
    --warmup_steps 1 \
    --measurement_steps 3 \
    --world_size 2

echo ""
echo "=========================================="
echo "Profiling complete!"
echo "=========================================="
echo "Profile file: ./nsight_profiles/ddp_bucket_simple.nsys-rep"
echo ""
echo "To download and view:"
echo "  scp $(whoami)@$(hostname):$(pwd)/nsight_profiles/ddp_bucket_simple.nsys-rep ."
echo ""
echo "Then open in Nsight Systems GUI application"
