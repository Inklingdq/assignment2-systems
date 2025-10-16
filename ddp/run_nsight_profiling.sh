#!/bin/bash

# Run DDP profiling with Nsight Systems to generate .nsys-rep file
# This will profile the bucket-based DDP to visualize compute-communication overlap

# Create output directory
mkdir -p ./nsight_profiles

# Run with nsys profiler
nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=./nsight_profiles/ddp_bucket_profile \
    --force-overwrite=true \
    --cuda-memory-usage=true \
    --stats=true \
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
echo "Profiling complete! Profile saved to ./nsight_profiles/ddp_bucket_profile.nsys-rep"
echo ""
echo "To visualize in Nsight Systems:"
echo "  1. Download the .nsys-rep file to your local machine"
echo "  2. Open Nsight Systems application"
echo "  3. File -> Open -> Select the .nsys-rep file"
echo "  4. Look for CUDA kernels, NCCL operations to see compute-communication overlap"
echo ""
echo "Alternatively, view report in terminal:"
echo "  nsys stats ./nsight_profiles/ddp_bucket_profile.nsys-rep"
