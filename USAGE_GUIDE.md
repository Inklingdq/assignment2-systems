# Quick Start Guide: Mixed Precision Benchmarking

## Overview

This guide shows you how to benchmark your language models with both full precision (FP32) and mixed precision (BF16) to measure performance improvements.

## What's New?

The `benchmarking.py` script now supports an optional `--use-mixed-precision` flag to enable BF16 mixed precision training, which:
- Reduces computation time (typically 1.5-2x speedup)
- Reduces memory usage (~50% for activations)
- Maintains numerical stability with BF16

## Quick Start

### Option 1: Test a Single Model

Compare FP32 vs BF16 for the Small model:

```bash
# Run with FP32
python benchmarking.py \
    --d-model 768 --num-layers 12 --num-heads 12 --d-ff 3072 \
    --warmup-steps 5 --measurement-steps 10

# Run with BF16
python benchmarking.py \
    --d-model 768 --num-layers 12 --num-heads 12 --d-ff 3072 \
    --warmup-steps 5 --measurement-steps 10 \
    --use-mixed-precision
```

Compare the `mean:` values in the output to see the speedup!

### Option 2: Run All Model Sizes

Automatically test all 4 model sizes (Small, Large, XL, 2.7B) with both precisions:

```bash
./run_mixed_precision_benchmarks.sh
```

This will:
1. Run 8 benchmarks total (4 models × 2 precisions)
2. Save results to `benchmark_results/`
3. Display a summary at the end

### Option 3: Run the Quick Example

Test with the Small model:

```bash
./example_mixed_precision_usage.sh
```

## Analyzing Results

After running benchmarks, analyze the results:

```bash
python analyze_mixed_precision_results.py
```

This will show:
- Detailed timing comparisons for each model
- Speedup factors
- Trends as model size increases
- Key insights about mixed precision performance

## Understanding the Output

When you run a benchmark, you'll see output like:

```
Running with mixed precision (BF16)
Model: d_model=768, d_ff=3072, num_layers=12, num_heads=12
Precision: BF16
time: [0.123456, 0.122345, 0.123567, ...]
mean: 0.123123
std: 0.000456
```

**Key metrics:**
- `mean`: Average time per training step (lower is better)
- `std`: Standard deviation (consistency of timing)
- Compare FP32 vs BF16 means to calculate speedup

## Model Configurations

The benchmarks test these models from §1.1.2:

| Model | d_model | layers | heads | d_ff   | Parameters |
|-------|---------|--------|-------|--------|------------|
| Small | 768     | 12     | 12    | 3072   | ~117M      |
| Large | 1024    | 24     | 16    | 4096   | ~355M      |
| XL    | 1536    | 24     | 16    | 6144   | ~839M      |
| 2.7B  | 2560    | 32     | 32    | 10240  | ~2768M     |

## Expected Results

You should observe:

1. **Speedup**: BF16 should be 1.5-2x faster than FP32
2. **Scaling**: Larger models may show greater speedup
3. **Memory**: Lower memory usage with BF16

Example speedup calculation:
```
FP32 mean: 0.250 seconds
BF16 mean: 0.150 seconds
Speedup: 0.250 / 0.150 = 1.67x
```

## Command-Line Arguments

All available arguments for `benchmarking.py`:

```bash
python benchmarking.py \
    --vocab-size 10000              # Vocabulary size
    --context-length 256            # Sequence length
    --d-model 768                   # Model dimension
    --num-layers 12                 # Number of transformer layers
    --num-heads 12                  # Number of attention heads
    --d-ff 3072                     # Feedforward dimension
    --warmup-steps 5                # Warmup iterations (not measured)
    --measurement-steps 10          # Measurement iterations
    --device cuda                   # Device (cuda or cpu)
    --use-mixed-precision           # Enable BF16 (optional)
```

## Troubleshooting

### No speedup observed?
- Check GPU: Requires Ampere (A100) or newer for best BF16 performance
- Verify CUDA version: Ensure recent PyTorch with BF16 support
- Check batch size: Larger batches may show more benefit

### Numerical issues?
- BF16 is numerically stable for most cases
- If loss diverges, try reducing learning rate
- Gradient clipping should help (already implemented)

### Script fails?
- Ensure all dependencies are installed: `pip install torch einops numpy`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify GPU memory: Large models need sufficient VRAM

## File Organization

After running benchmarks:

```
assignment2-systems/
├── benchmarking.py                  # Main script
├── run_mixed_precision_benchmarks.sh  # Run all benchmarks
├── analyze_mixed_precision_results.py # Analyze results
├── example_mixed_precision_usage.sh   # Quick example
└── benchmark_results/               # Results directory
    ├── small_fp32_results.txt
    ├── small_bf16_results.txt
    ├── large_fp32_results.txt
    ├── large_bf16_results.txt
    ├── xl_fp32_results.txt
    ├── xl_bf16_results.txt
    ├── 2.7B_fp32_results.txt
    └── 2.7B_bf16_results.txt
```

## Next Steps

1. **Run benchmarks**: Start with `./example_mixed_precision_usage.sh`
2. **Compare results**: Look at the mean times for FP32 vs BF16
3. **Full suite**: Run `./run_mixed_precision_benchmarks.sh` for all models
4. **Analyze**: Use `python analyze_mixed_precision_results.py`
5. **Document**: Note trends as model size increases

## Key Insights to Look For

As you analyze your results, consider:

1. **How much faster is BF16?** Calculate the speedup factor
2. **Does speedup increase with model size?** Compare Small vs 2.7B
3. **Is precision stable?** Check that loss values are reasonable
4. **Memory benefits?** Monitor GPU memory usage with `nvidia-smi`

## Further Reading

- `MIXED_PRECISION_README.md` - Detailed technical documentation
- `MIXED_PRECISION_SUMMARY.md` - Complete summary of changes
- PyTorch AMP docs: https://pytorch.org/docs/stable/amp.html

---

**Ready to start?** Run: `./example_mixed_precision_usage.sh`
