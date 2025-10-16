"""
Benchmark compiled vs uncompiled Transformer model.

Compares performance of torch.compile() on the full model.
"""

import argparse
import timeit

import numpy as np
import torch
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from einops import rearrange


def benchmark_model(model, x, y, optimizer, num_iterations=100, warmup_iterations=5):
    """
    Benchmark a model (compiled or uncompiled).
    
    Returns:
        dict with timing results
    """
    device = x.device
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
    
    # Time forward passes only
    forward_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        logits = model(x)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)
    
    # Time forward + backward + optimizer step (full training step)
    full_step_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        
        logits = model(x)
        logits_flatten = rearrange(logits, "b c v -> (b c) v")
        y_flatten = rearrange(y, "b c -> (b c)")
        loss = cross_entropy(logits_flatten, y_flatten)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        end = timeit.default_timer()
        full_step_times.append(end - start)
    
    return {
        'forward_mean': np.mean(forward_times) * 1000,  # Convert to ms
        'forward_std': np.std(forward_times) * 1000,
        'full_step_mean': np.mean(full_step_times) * 1000,
        'full_step_std': np.std(full_step_times) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark compiled vs uncompiled Transformer model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to time")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    device = args.device
    
    print(f"Benchmarking Compiled vs Uncompiled Transformer Model")
    print(f"Configuration:")
    print(f"  Model: d_model={args.d_model}, d_ff={args.d_ff}, num_layers={args.num_layers}, num_heads={args.num_heads}")
    print(f"  Data: batch_size={args.batch_size}, context_length={args.context_length}, vocab_size={args.vocab_size}")
    print(f"  Iterations: {args.iterations}, Warmup: {args.warmup}")
    print("=" * 120)
    
    # Create model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    model.to(device)
    
    # Create random data
    random_data = np.random.randint(
        low=0, high=args.vocab_size - 1, size=args.context_length * 2, dtype=np.int32
    )
    x, y = get_batch(
        dataset=random_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=device,
    )
    
    results = []
    
    # Benchmark uncompiled model
    print("\nBenchmarking uncompiled model...")
    uncompiled_optimizer = AdamW(model.parameters())
    uncompiled_result = benchmark_model(
        model,
        x,
        y,
        uncompiled_optimizer,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    uncompiled_result['version'] = 'Uncompiled'
    results.append(uncompiled_result)
    print(f"  Forward: {uncompiled_result['forward_mean']:.2f} ms")
    print(f"  Full Step: {uncompiled_result['full_step_mean']:.2f} ms")
    
    # Compile model
    print("\nCompiling model...")
    compiled_model = torch.compile(model)
    
    # Benchmark compiled model
    print("Benchmarking compiled model...")
    compiled_optimizer = AdamW(compiled_model.parameters())
    compiled_result = benchmark_model(
        compiled_model,
        x,
        y,
        compiled_optimizer,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    compiled_result['version'] = 'Compiled'
    results.append(compiled_result)
    print(f"  Forward: {compiled_result['forward_mean']:.2f} ms")
    print(f"  Full Step: {compiled_result['full_step_mean']:.2f} ms")
    
    # Print results table
    print("\n" + "=" * 120)
    print("RESULTS")
    print("=" * 120)
    print(f"{'Version':<15} {'Forward Pass (ms)':<30} {'Full Training Step (ms)':<30} {'Speedup (Fwd)':<15} {'Speedup (Full)':<15}")
    print("=" * 120)
    
    for result in results:
        if result['version'] == 'Uncompiled':
            speedup_fwd = 1.0
            speedup_full = 1.0
        else:
            speedup_fwd = uncompiled_result['forward_mean'] / result['forward_mean']
            speedup_full = uncompiled_result['full_step_mean'] / result['full_step_mean']
        
        print(f"{result['version']:<15} "
              f"{result['forward_mean']:.3f} ± {result['forward_std']:.3f}           "
              f"{result['full_step_mean']:.3f} ± {result['full_step_std']:.3f}           "
              f"{speedup_fwd:.2f}x            "
              f"{speedup_full:.2f}x")
    
    print("=" * 120)
    
    # Calculate speedups
    forward_speedup = uncompiled_result['forward_mean'] / compiled_result['forward_mean']
    full_step_speedup = uncompiled_result['full_step_mean'] / compiled_result['full_step_mean']
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Forward Pass:")
    print(f"    Uncompiled: {uncompiled_result['forward_mean']:.2f} ± {uncompiled_result['forward_std']:.2f} ms")
    print(f"    Compiled:   {compiled_result['forward_mean']:.2f} ± {compiled_result['forward_std']:.2f} ms")
    print(f"    Speedup:    {forward_speedup:.2f}x {'faster' if forward_speedup > 1 else 'slower'}")
    print()
    print(f"  Full Training Step (Forward + Backward + Optimizer):")
    print(f"    Uncompiled: {uncompiled_result['full_step_mean']:.2f} ± {uncompiled_result['full_step_std']:.2f} ms")
    print(f"    Compiled:   {compiled_result['full_step_mean']:.2f} ± {compiled_result['full_step_std']:.2f} ms")
    print(f"    Speedup:    {full_step_speedup:.2f}x {'faster' if full_step_speedup > 1 else 'slower'}")
    print()
    
    if forward_speedup > 1.0:
        print("  Benefits of torch.compile() on Transformer:")
        print("    - Kernel fusion across attention and FFN operations")
        print("    - Reduced memory traffic between operations")
        print("    - Elimination of Python interpreter overhead")
        print("    - Better GPU utilization through fused operations")
        print()
        print("  The forward pass typically sees larger speedups because:")
        print("    - Forward operations can be more aggressively fused")
        print("    - Backward pass requires saving gradients (less fusion opportunity)")
    else:
        print("  Note: Compilation overhead may dominate for small models.")
        print("  Try larger models or longer sequences for more significant speedups.")
    
    print("=" * 120)


if __name__ == "__main__":
    main()
