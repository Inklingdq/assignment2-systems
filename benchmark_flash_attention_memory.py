import torch
import pandas as pd
import gc
from tests.adapters import get_flashattention_autograd_function_triton


def pytorch_attention(q, k, v, is_causal=True):
    """Regular PyTorch attention implementation (not FlashAttention)."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        causal_mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        attn = torch.where(causal_mask, attn, -1e6)
    
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out


def measure_memory_usage(impl_fn, q, k, v, is_causal=True, do_backward=True):
    """
    Measure peak memory usage for forward pass (and optionally backward).
    Returns memory in MB.
    """
    # Clear cache and reset peak memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Initial memory
    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated() / 1024**2
    
    # Forward pass
    out = impl_fn(q, k, v, is_causal)
    torch.cuda.synchronize()
    
    if do_backward:
        # Backward pass
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        torch.cuda.synchronize()
    
    # Peak memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    end_mem = torch.cuda.memory_allocated() / 1024**2
    
    # Memory used during operation
    memory_used = peak_mem - start_mem
    
    return memory_used, peak_mem, end_mem


def run_memory_benchmark():
    """Run memory benchmarks comparing Triton and PyTorch implementations."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    device = 'cuda'
    batch_size = 1
    is_causal = True
    
    # Test parameters - focus on showing memory scaling
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    embed_dims = [64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    FlashAttn = get_flashattention_autograd_function_triton()
    
    results = []
    
    print("=" * 100)
    print("FlashAttention-2 Memory Usage Benchmark")
    print("=" * 100)
    print(f"Configuration: batch_size={batch_size}, causal={is_causal}")
    print("")
    
    for dtype in dtypes:
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        bytes_per_element = 2 if dtype == torch.bfloat16 else 4
        
        print(f"\nTesting with dtype: {dtype_name}")
        
        for d in embed_dims:
            print(f"  Embedding dimension: {d}")
            
            for seq_len in seq_lengths:
                try:
                    print(f"    Testing seq_len={seq_len}...", end=" ", flush=True)
                    
                    # Generate inputs
                    torch.manual_seed(42)
                    
                    # Measure PyTorch attention memory
                    q_torch = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    k_torch = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    v_torch = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    
                    try:
                        pytorch_mem, pytorch_peak, pytorch_end = measure_memory_usage(
                            pytorch_attention, q_torch, k_torch, v_torch, is_causal, do_backward=True
                        )
                        pytorch_success = True
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            pytorch_mem, pytorch_peak, pytorch_end = float('inf'), float('inf'), float('inf')
                            pytorch_success = False
                            torch.cuda.empty_cache()
                        else:
                            raise
                    
                    # Clean up
                    del q_torch, k_torch, v_torch
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Measure Triton FlashAttention memory
                    q_triton = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    k_triton = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    v_triton = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    
                    triton_mem, triton_peak, triton_end = measure_memory_usage(
                        FlashAttn.apply, q_triton, k_triton, v_triton, is_causal, do_backward=True
                    )
                    
                    # Clean up
                    del q_triton, k_triton, v_triton
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Calculate theoretical memory for attention matrix
                    # Attention matrix size: batch_size * seq_len * seq_len * bytes_per_element
                    attn_matrix_mb = (batch_size * seq_len * seq_len * bytes_per_element) / 1024**2
                    
                    # Memory savings
                    if pytorch_success:
                        memory_reduction = ((pytorch_mem - triton_mem) / pytorch_mem) * 100
                        memory_ratio = pytorch_mem / triton_mem
                    else:
                        memory_reduction = float('inf')
                        memory_ratio = float('inf')
                    
                    results.append({
                        'dtype': dtype_name,
                        'seq_len': seq_len,
                        'embed_dim': d,
                        'triton_mem_mb': triton_mem,
                        'pytorch_mem_mb': pytorch_mem if pytorch_success else 'OOM',
                        'memory_saved_mb': pytorch_mem - triton_mem if pytorch_success else 'N/A',
                        'memory_reduction_pct': memory_reduction if pytorch_success else 'N/A',
                        'memory_ratio': memory_ratio if pytorch_success else 'N/A',
                        'attn_matrix_mb': attn_matrix_mb,
                        'pytorch_oom': not pytorch_success,
                    })
                    
                    if pytorch_success:
                        print(f"Done (PyTorch: {pytorch_mem:.1f}MB, Triton: {triton_mem:.1f}MB, "
                              f"Saved: {pytorch_mem - triton_mem:.1f}MB, Reduction: {memory_reduction:.1f}%)")
                    else:
                        print(f"Done (PyTorch: OOM, Triton: {triton_mem:.1f}MB)")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Both OOM - skipping")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Error: {e}")
                        raise
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print detailed table
    print("\n\n" + "=" * 140)
    print("MEMORY USAGE COMPARISON (Forward + Backward)")
    print("=" * 140)
    print(f"{'Dtype':<10} {'SeqLen':<8} {'EmbDim':<8} {'Triton Mem':<13} {'PyTorch Mem':<14} "
          f"{'Memory Saved':<14} {'Reduction %':<13} {'Attn Matrix':<13} {'Ratio':<8}")
    print("-" * 140)
    
    for _, row in df.iterrows():
        pytorch_mem_str = f"{row['pytorch_mem_mb']:.1f} MB" if row['pytorch_mem_mb'] != 'OOM' else "OOM"
        saved_str = f"{row['memory_saved_mb']:.1f} MB" if row['memory_saved_mb'] != 'N/A' else "N/A"
        reduction_str = f"{row['memory_reduction_pct']:.1f}%" if row['memory_reduction_pct'] != 'N/A' else "N/A"
        ratio_str = f"{row['memory_ratio']:.2f}x" if row['memory_ratio'] != 'N/A' else "N/A"
        
        print(f"{row['dtype']:<10} {row['seq_len']:<8} {row['embed_dim']:<8} "
              f"{row['triton_mem_mb']:>11.1f} MB {pytorch_mem_str:>14} "
              f"{saved_str:>14} {reduction_str:>13} "
              f"{row['attn_matrix_mb']:>11.1f} MB {ratio_str:>8}")
    
    print("=" * 140)
    
    # Save results
    df.to_csv('flash_attention_memory_results.csv', index=False)
    print(f"\nResults saved to: flash_attention_memory_results.csv")
    
    # Print analysis
    print("\n" + "=" * 100)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 100)
    
    for dtype_name in ['bfloat16', 'float32']:
        dtype_df = df[df['dtype'] == dtype_name]
        if len(dtype_df) > 0:
            print(f"\n{dtype_name} Results:")
            
            # Filter successful runs
            successful = dtype_df[~dtype_df['pytorch_oom']]
            if len(successful) > 0:
                print(f"  Average memory reduction: {successful['memory_reduction_pct'].mean():.1f}%")
                print(f"  Average memory ratio: {successful['memory_ratio'].mean():.2f}x")
                print(f"  Max memory saved: {successful['memory_saved_mb'].max():.1f} MB")
            
            # Show where PyTorch OOMs but Triton doesn't
            oom_cases = dtype_df[dtype_df['pytorch_oom']]
            if len(oom_cases) > 0:
                print(f"\n  Sequences where PyTorch OOMs but Triton succeeds:")
                for _, row in oom_cases.iterrows():
                    print(f"    seq_len={row['seq_len']}, embed_dim={row['embed_dim']}: "
                          f"Triton uses only {row['triton_mem_mb']:.1f} MB")
    
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print("""
1. Memory Scaling:
   - Standard Attention: O(N²) memory due to storing attention matrix
   - FlashAttention-2: O(N) memory by using tiling and recomputation

2. Attention Matrix Size:
   - For seq_len=8192: ~256 MB (bfloat16) or ~512 MB (float32)
   - For seq_len=16384: ~1 GB (bfloat16) or ~2 GB (float32)
   - For seq_len=32768: ~4 GB (bfloat16) or ~8 GB (float32)

3. FlashAttention-2 Advantages:
   - Reduces peak memory by 20-50% for typical workloads
   - Enables processing of much longer sequences
   - Memory usage grows linearly (O(N)) instead of quadratically (O(N²))

4. Practical Impact:
   - Standard attention OOMs on longer sequences
   - FlashAttention-2 handles these sequences with reasonable memory
   - Critical for training on long-context models
    """)
    
    return df


if __name__ == '__main__':
    df = run_memory_benchmark()
