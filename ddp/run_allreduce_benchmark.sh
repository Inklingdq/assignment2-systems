#!/bin/bash
# Convenience script to run the full DDP all-reduce benchmark suite
# This will run all configurations and generate analysis plots

set -e  # Exit on error

echo "========================================================================"
echo "DDP All-Reduce Benchmark Suite"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Run benchmarks for all configurations (Gloo+CPU, NCCL+GPU)"
echo "  2. Test data sizes: 1MB, 10MB, 100MB, 1GB"
echo "  3. Test world sizes: 2, 4, 6 processes"
echo "  4. Generate plots and analysis"
echo ""
echo "Estimated time: 5-15 minutes depending on hardware"
echo "========================================================================"
echo ""

# Check if user wants to proceed
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run the benchmark
echo ""
echo "Starting benchmark..."
echo ""

python ddp/benchmark_allreduce.py

# Check if benchmark succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Benchmark failed!"
    exit 1
fi

# Run the analysis
echo ""
echo "========================================================================"
echo "Running analysis and generating plots..."
echo "========================================================================"
echo ""

python ddp/analyze_allreduce.py

# Check if analysis succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis failed!"
    exit 1
fi

echo ""
echo "========================================================================"
echo "All done! Check the benchmark_results_allreduce/ directory for:"
echo "  - allreduce_benchmark_results.csv (raw data)"
echo "  - allreduce_benchmark_results.json (detailed data with raw times)"
echo "  - time_vs_datasize.png (performance plot)"
echo "  - bandwidth_comparison.png (bandwidth plot)"
echo "  - scaling_efficiency.png (scaling plot)"
echo "========================================================================"
echo ""
