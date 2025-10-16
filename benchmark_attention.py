"""
Benchmark attention implementation at different scales.

This script tests attention with:
- Fixed batch size of 8
- No multihead attention (no head dimension)
- Various d_model and sequence length combinations
"""

import argparse
import timeit
from itertools import product

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


def benchmark_attention(batch_size, d_model, seq_len, num_iterations=100, warmup_iterations=5):
    """
    Benchmark attention forward and backward passes.
    
    Args:
        batch_size: Batch size
        d_model: Head embedding dimension
        seq_len: Sequence length
        num_iterations: Number of iterations to time
        warmup_iterations: Number of warmup iterations
        
    Returns:
        dict with timing and memory results, or None if OOM
    """
    device = "cuda"
    
    try:
        # Create random inputs Q, K, V
        # Shape: (batch_size, seq_len, d_model)
        Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
        
        # Create causal mask
        seq = torch.arange(seq_len, device=device)
        causal_mask = seq.unsqueeze(0).unsqueeze(1) >= seq.unsqueeze(0).unsqueeze(2)  # (1, seq_len, seq_len)
        causal_mask = causal_mask.expand(batch_size, seq_len, seq_len)
        
        # Warmup iterations
        for _ in range(warmup_iterations):
            output = scaled_dot_product_attention(Q, K, V, causal_mask)
            loss = output.sum()
            loss.backward()
            Q.grad = None
            K.grad = None
            V.grad = None
            torch.cuda.synchronize()
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Time forward passes
        forward_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = timeit.default_timer()
            output = scaled_dot_product_attention(Q, K, V, causal_mask)
            torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_times.append(end - start)
        
        # Measure memory before backward
        memory_before_backward = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        
        # Time backward passes
        backward_times = []
        for _ in range(num_iterations):
            output = scaled_dot_product_attention(Q, K, V, causal_mask)
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
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
        
        return {
            'forward_mean': np.mean(forward_times) * 1000,  # Convert to ms
            'forward_std': np.std(forward_times) * 1000,
            'backward_mean': np.mean(backward_times) * 1000,
            'backward_std': np.std(backward_times) * 1000,
            'memory_before_backward_gb': memory_before_backward,
            'peak_memory_gb': peak_memory,
            'success': True
        }
        
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            'success': False,
            'error': 'OOM'
        }
    except Exception as e:
        torch.cuda.empty_cache()
        return {
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention at different scales")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations to time")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    args = parser.parse_args()
    
    # Configuration
    batch_size = args.batch_size
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    
    print(f"Benchmarking Attention Implementation")
    print(f"Batch Size: {batch_size}")
    print(f"Iterations: {args.iterations}")
    print(f"=" * 100)
    
    # Header
    print(f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<20} {'Backward (ms)':<20} {'Mem Before Bwd (GB)':<22} {'Peak Mem (GB)':<15} {'Status':<10}")
    print("=" * 100)
    
    results = []
    
    # Iterate through all combinations
    for d_model, seq_len in product(d_models, seq_lens):
        result = benchmark_attention(
            batch_size=batch_size,
            d_model=d_model,
            seq_len=seq_len,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup
        )
        
        result['d_model'] = d_model
        result['seq_len'] = seq_len
        result['batch_size'] = batch_size
        results.append(result)
        
        # Print results
        if result['success']:
            print(f"{d_model:<10} {seq_len:<10} "
                  f"{result['forward_mean']:.2f} ± {result['forward_std']:.2f}     "
                  f"{result['backward_mean']:.2f} ± {result['backward_std']:.2f}     "
                  f"{result['memory_before_backward_gb']:.4f}                "
                  f"{result['peak_memory_gb']:.4f}          "
                  f"OK")
        else:
            print(f"{d_model:<10} {seq_len:<10} {'N/A':<20} {'N/A':<20} {'N/A':<22} {'N/A':<15} {result['error']:<10}")
    
    print("=" * 100)
    
    # Analysis
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    
    # Find first OOM configuration
    oom_configs = [r for r in results if not r['success']]
    if oom_configs:
        first_oom = oom_configs[0]
        print(f"\nFirst OOM configuration:")
        print(f"  d_model={first_oom['d_model']}, seq_len={first_oom['seq_len']}")
        
        # Memory accounting for smallest OOM config
        print(f"\nMemory Accounting for d_model={first_oom['d_model']}, seq_len={first_oom['seq_len']}:")
        batch = batch_size
        d = first_oom['d_model']
        s = first_oom['seq_len']
        
        # Forward pass memory
        qkv_size = 3 * batch * s * d * 4 / (1024**3)  # Q, K, V in bytes
        attention_scores_size = batch * s * s * 4 / (1024**3)  # Attention scores
        attention_weights_size = batch * s * s * 4 / (1024**3)  # After softmax
        output_size = batch * s * d * 4 / (1024**3)  # Output
        
        print(f"  Q, K, V: 3 × ({batch} × {s} × {d}) × 4 bytes = {qkv_size:.4f} GB")
        print(f"  Attention scores: ({batch} × {s} × {s}) × 4 bytes = {attention_scores_size:.4f} GB")
        print(f"  Attention weights: ({batch} × {s} × {s}) × 4 bytes = {attention_weights_size:.4f} GB")
        print(f"  Output: ({batch} × {s} × {d}) × 4 bytes = {output_size:.4f} GB")
        print(f"  Total (forward activations): {qkv_size + attention_scores_size + attention_weights_size + output_size:.4f} GB")
        
        # Backward pass saves attention scores and weights for gradient computation
        print(f"\n  Backward pass saves:")
        print(f"    - Attention scores for softmax gradient: {attention_scores_size:.4f} GB")
        print(f"    - Attention weights for matmul gradient: {attention_weights_size:.4f} GB")
        print(f"    - Q, K, V for gradient computation: {qkv_size:.4f} GB")
        
    # Analyze memory scaling with sequence length
    print(f"\nMemory scaling with sequence length:")
    print(f"{'d_model':<10} {'seq_len':<10} {'Mem Before Bwd (GB)':<22} {'Memory Saved (GB)':<20}")
    
    for d_model in d_models:
        d_results = [r for r in results if r['d_model'] == d_model and r['success']]
        if d_results:
            for r in d_results:
                # Estimate memory saved for backward (primarily attention scores)
                batch = r['batch_size']
                s = r['seq_len']
                # Attention scores saved: batch × seq_len × seq_len × 4 bytes
                attention_memory = batch * s * s * 4 / (1024**3)
                print(f"{r['d_model']:<10} {r['seq_len']:<10} "
                      f"{r['memory_before_backward_gb']:.4f}                "
                      f"~{attention_memory:.4f} (attn scores)")
    
    print("\n" + "=" * 100)
    print("SOLUTIONS TO REDUCE MEMORY COST:")
    print("=" * 100)
    print("1. Gradient Checkpointing (Recomputation):")
    print("   - Don't save activations during forward pass")
    print("   - Recompute them during backward pass")
    print("   - Trades compute for memory (2x compute, much less memory)")
    print()
    print("2. Flash Attention:")
    print("   - Fused kernel that computes attention without materializing full attention matrix")
    print("   - Uses block-wise computation and on-chip SRAM")
    print("   - O(N) memory instead of O(N²) for sequence length N")
    print()
    print("3. Mixed Precision Training:")
    print("   - Use FP16/BF16 instead of FP32 (2x memory reduction)")
    print()
    print("4. Reduce Batch Size:")
    print("   - Linear reduction in memory usage")
    print()
    print("5. Sparse Attention Patterns:")
    print("   - Only compute attention for subset of positions")
    print("   - E.g., local attention, strided attention, etc.")
    print("=" * 100)


if __name__ == "__main__":
    main()
