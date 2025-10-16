#!/bin/bash

# Script to measure peak memory usage for different context lengths
# Runs 2.7B model with both forward-only and full training modes

echo "========================================"
echo "Memory Benchmarking: 2.7B Model"
echo "========================================"
echo ""

# 2.7B model configuration
D_MODEL=2560
NUM_LAYERS=32
NUM_HEADS=32
D_FF=10240

# Context lengths to test
# CONTEXT_LENGTHS=(128 256 512 1024 2048)
CONTEXT_LENGTHS=(128 256 512)

# Output file
OUTPUT_FILE="memory_benchmark_results.txt"
> $OUTPUT_FILE  # Clear file

echo "Context Length | Forward Pass (GB) | Full Training (GB)" >> $OUTPUT_FILE
echo "---------------|-------------------|-------------------" >> $OUTPUT_FILE

for CTX_LEN in "${CONTEXT_LENGTHS[@]}"; do
    echo "=========================================="
    echo "Testing context_length=${CTX_LEN}"
    echo "=========================================="

    # Run forward-only
    echo "Running forward-only..."
    FORWARD_OUTPUT=$(uv run benchmarking.py \
        --d-model $D_MODEL \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --d-ff $D_FF \
        --context-length $CTX_LEN \
        --warmup-steps 0 \
        --measurement-steps 1 \
        --measure-forward-only True 2>&1)

    # Extract peak memory reserved from output
    FORWARD_MEM=$(echo "$FORWARD_OUTPUT" | grep "Peak Memory Reserved:" | awk '{print $4}')

    if [ -z "$FORWARD_MEM" ]; then
        echo "Error: Could not extract forward memory for context_length=${CTX_LEN}"
        FORWARD_MEM="OOM"
    else
        echo "✓ Forward-only: ${FORWARD_MEM} GB"
    fi

    # Run full training
    echo "Running full training..."
    FULL_OUTPUT=$(uv run benchmarking.py \
        --d-model $D_MODEL \
        --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS \
        --d-ff $D_FF \
        --context-length $CTX_LEN \
        --warmup-steps 0 \
        --measurement-steps 1 2>&1)

    # Extract peak memory reserved from output
    FULL_MEM=$(echo "$FULL_OUTPUT" | grep "Peak Memory Reserved:" | awk '{print $4}')

    if [ -z "$FULL_MEM" ]; then
        echo "Error: Could not extract full training memory for context_length=${CTX_LEN}"
        FULL_MEM="OOM"
    else
        echo "✓ Full training: ${FULL_MEM} GB"
    fi

    # Write to results file
    printf "%14s | %17s | %17s\n" "$CTX_LEN" "$FORWARD_MEM" "$FULL_MEM" >> $OUTPUT_FILE

    echo ""
done

echo "========================================"
echo "Benchmarking Complete!"
echo "========================================"
echo ""

# Display results
echo "RESULTS:"
echo ""
cat $OUTPUT_FILE
echo ""

# Generate LaTeX table
echo "========================================"
echo "LaTeX Format:"
echo "========================================"
echo ""
echo "\\begin{tabular}{|c|c|c|}"
echo "\\hline"
echo "Context Length & Forward Pass (GB) & Full Training (GB) \\\\"
echo "\\hline"

while IFS='|' read -r ctx forward full; do
    # Skip header lines
    if [[ $ctx == *"Context"* ]] || [[ $ctx == *"---"* ]]; then
        continue
    fi
    # Trim whitespace
    ctx=$(echo "$ctx" | xargs)
    forward=$(echo "$forward" | xargs)
    full=$(echo "$full" | xargs)
    echo "$ctx & $forward & $full \\\\"
done < $OUTPUT_FILE

echo "\\hline"
echo "\\end{tabular}"
echo ""

# Generate Markdown table
echo "========================================"
echo "Markdown Format:"
echo "========================================"
echo ""
cat $OUTPUT_FILE
echo ""

echo "Results saved to: $OUTPUT_FILE"
