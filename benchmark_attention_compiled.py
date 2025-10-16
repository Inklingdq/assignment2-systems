"""
Benchmark compiled vs uncompiled attention implementation.

Compares performance of torch.compile() on attention with the vanilla version.
"""

import argparse
import timeit

import numpy as np
import torch
import torch.nn as nn
from cs336_basics.model import scaled_dot_product_attention


class AttentionModule(nn.Module):
    """Wrapper module for scaled_dot_product_attention to enable compilation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask):
        return scaled_dot_product_attention(Q, K, V, mask)


def benchmark_attention_version(attention_fn, Q, K, V, mask, num_iterations=100, warmup_iterations=5, version_name=""):
    """
    Benchmark a specific version of attention (compiled or uncompiled).
    
    Returns:
        dict with timing results
    """
    device = Q.device
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()
        loss.backward()
        Q.grad = None
        K.grad = None
        V.grad = None
        torch.cuda.synchronize()
    
    # Time forward passes
    forward_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        output = attention_fn(Q, K, V, mask)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        forward_times.append(end - start)
    
    # Time backward passes
    backward_times = []
    for _ in range(num_iterations):
        output = attention_fn(Q, K, V, mask)
        loss = output.sum()
        
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        backward_times.append(end - start)
        
        # Zero gradients for next iteration
        Q.grad = None
        K.grad = None
        V.grad = None
    
    return {
        'version': version_name,
        'forward_mean': np.mean(forward_times) * 1000,  # Convert to ms
        'forward_std': np.std(forward_times) * 1000,
        'backward_mean': np.mean(backward_times) * 1000,
        'backward_std': np.std(backward_times) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark compiled vs uncompiled attention")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--d-model", type=int, default=64, help="Head embedding dimension")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to time")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    args = parser.parse_args()
    
    device = "cuda"
    batch_size = args.batch_size
    d_model = args.d_model
    seq_len = args.seq_len
    
    print(f"Benchmarking Compiled vs Uncompiled Attention")
    print(f"Configuration: batch_size={batch_size}, d_model={d_model}, seq_len={seq_len}")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")
    print("=" * 100)
    
    # Create random inputs Q, K, V
    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    
    # Create causal mask
    seq = torch.arange(seq_len, device=device)
    causal_mask = seq.unsqueeze(0).unsqueeze(1) >= seq.unsqueeze(0).unsqueeze(2)
    causal_mask = causal_mask.expand(batch_size, seq_len, seq_len)
    
    results = []
    
    # Benchmark uncompiled version (function)
    print("\nBenchmarking uncompiled attention (function)...")
    uncompiled_result = benchmark_attention_version(
        scaled_dot_product_attention,
        Q.clone().detach().requires_grad_(True),
        K.clone().detach().requires_grad_(True),
        V.clone().detach().requires_grad_(True),
        causal_mask,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        version_name="Uncompiled"
    )
    results.append(uncompiled_result)
    
    # Benchmark compiled version (as module)
    print("Benchmarking compiled attention (module)...")
    attention_module = AttentionModule().to(device)
    compiled_attention_module = torch.compile(attention_module)
    
    compiled_result = benchmark_attention_version(
        compiled_attention_module,
        Q.clone().detach().requires_grad_(True),
        K.clone().detach().requires_grad_(True),
        V.clone().detach().requires_grad_(True),
        causal_mask,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        version_name="Compiled"
    )
    results.append(compiled_result)
    
    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"{'Version':<15} {'Forward (ms)':<25} {'Backward (ms)':<25} {'Speedup (Fwd)':<15} {'Speedup (Bwd)':<15}")
    print("=" * 100)
    
    for result in results:
        speedup_fwd = uncompiled_result['forward_mean'] / result['forward_mean'] if result['version'] != 'Uncompiled' else 1.0
        speedup_bwd = uncompiled_result['backward_mean'] / result['backward_mean'] if result['version'] != 'Uncompiled' else 1.0
        
        print(f"{result['version']:<15} "
              f"{result['forward_mean']:.3f} ± {result['forward_std']:.3f}        "
              f"{result['backward_mean']:.3f} ± {result['backward_std']:.3f}        "
              f"{speedup_fwd:.2f}x            "
              f"{speedup_bwd:.2f}x")
    
    print("=" * 100)
    
    # Summary
    compiled_speedup_fwd = uncompiled_result['forward_mean'] / compiled_result['forward_mean']
    compiled_speedup_bwd = uncompiled_result['backward_mean'] / compiled_result['backward_mean']
    
    print("\nSUMMARY:")
    print(f"  Compiled attention forward pass is {compiled_speedup_fwd:.2f}x {'faster' if compiled_speedup_fwd > 1 else 'slower'} than uncompiled")
    print(f"  Compiled attention backward pass is {compiled_speedup_bwd:.2f}x {'faster' if compiled_speedup_bwd > 1 else 'slower'} than uncompiled")
    
    if compiled_speedup_fwd > 1 or compiled_speedup_bwd > 1:
        print("\n  torch.compile() provides speedups through:")
        print("    - Kernel fusion (reducing memory traffic)")
        print("    - Eliminating Python overhead")
        print("    - Graph-level optimizations")
    else:
        print("\n  Note: For small models/sequences, compilation overhead may outweigh benefits.")
        print("  Larger models typically see more significant speedups.")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
