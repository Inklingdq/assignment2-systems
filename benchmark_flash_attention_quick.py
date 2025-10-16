import torch
import triton
import pandas as pd
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


def run_quick_benchmark():
    """Run a quicker benchmark with fewer configurations."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    device = 'cuda'
    batch_size = 1
    is_causal = True
    
    # Reduced sweep parameters for faster benchmarking
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    embed_dims = [32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    FlashAttn = get_flashattention_autograd_function_triton()
    
    results = []
    
    print("Starting Quick FlashAttention-2 Benchmarks")
    print("=" * 100)
    print(f"Configuration: batch_size={batch_size}, causal={is_causal}")
    print("")
    
    for dtype in dtypes:
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        print(f"\nTesting with dtype: {dtype_name}")
        
        for d in embed_dims:
            print(f"  Embedding dimension: {d}")
            
            for seq_len in seq_lengths:
                try:
                    print(f"    Testing seq_len={seq_len}...", end=" ", flush=True)
                    
                    # Generate random inputs
                    torch.manual_seed(42)
                    q = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    grad_output = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype)
                    
                    # Warm up
                    for _ in range(5):
                        _ = FlashAttn.apply(q.clone().detach(), k.clone().detach(), v.clone().detach(), is_causal)
                        _ = pytorch_attention(q.clone().detach(), k.clone().detach(), v.clone().detach(), is_causal)
                    torch.cuda.synchronize()
                    
                    # Benchmark Triton forward
                    q_triton = q.clone().detach().requires_grad_(True)
                    k_triton = k.clone().detach().requires_grad_(True)
                    v_triton = v.clone().detach().requires_grad_(True)
                    
                    def triton_fwd():
                        return FlashAttn.apply(q_triton, k_triton, v_triton, is_causal)
                    
                    triton_fwd_ms = triton.testing.do_bench(triton_fwd, warmup=10, rep=50)
                    
                    # Benchmark Triton forward+backward
                    def triton_fwd_bwd():
                        q_triton.grad = None
                        k_triton.grad = None
                        v_triton.grad = None
                        out = FlashAttn.apply(q_triton, k_triton, v_triton, is_causal)
                        out.backward(grad_output)
                        torch.cuda.synchronize()
                    
                    triton_fwd_bwd_ms = triton.testing.do_bench(triton_fwd_bwd, warmup=10, rep=50)
                    triton_bwd_ms = triton_fwd_bwd_ms - triton_fwd_ms
                    
                    # Benchmark PyTorch forward
                    q_torch = q.clone().detach().requires_grad_(True)
                    k_torch = k.clone().detach().requires_grad_(True)
                    v_torch = v.clone().detach().requires_grad_(True)
                    
                    def torch_fwd():
                        return pytorch_attention(q_torch, k_torch, v_torch, is_causal)
                    
                    torch_fwd_ms = triton.testing.do_bench(torch_fwd, warmup=10, rep=50)
                    
                    # Benchmark PyTorch forward+backward
                    def torch_fwd_bwd():
                        q_torch.grad = None
                        k_torch.grad = None
                        v_torch.grad = None
                        out = pytorch_attention(q_torch, k_torch, v_torch, is_causal)
                        out.backward(grad_output)
                        torch.cuda.synchronize()
                    
                    torch_fwd_bwd_ms = triton.testing.do_bench(torch_fwd_bwd, warmup=10, rep=50)
                    torch_bwd_ms = torch_fwd_bwd_ms - torch_fwd_ms
                    
                    # Calculate speedups
                    speedup_fwd = torch_fwd_ms / triton_fwd_ms
                    speedup_bwd = torch_bwd_ms / triton_bwd_ms if triton_bwd_ms > 0 else 0
                    speedup_fwd_bwd = torch_fwd_bwd_ms / triton_fwd_bwd_ms
                    
                    results.append({
                        'dtype': dtype_name,
                        'seq_len': seq_len,
                        'embed_dim': d,
                        'triton_fwd_ms': triton_fwd_ms,
                        'triton_bwd_ms': triton_bwd_ms,
                        'triton_fwd_bwd_ms': triton_fwd_bwd_ms,
                        'pytorch_fwd_ms': torch_fwd_ms,
                        'pytorch_bwd_ms': torch_bwd_ms,
                        'pytorch_fwd_bwd_ms': torch_fwd_bwd_ms,
                        'speedup_fwd': speedup_fwd,
                        'speedup_bwd': speedup_bwd,
                        'speedup_fwd_bwd': speedup_fwd_bwd,
                    })
                    
                    print(f"Done (Speedup F+B: {speedup_fwd_bwd:.2f}x)")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM - skipping")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Error: {e}")
                        raise
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print detailed table
    print("\n\n" + "=" * 160)
    print("DETAILED BENCHMARK RESULTS")
    print("=" * 160)
    print(f"{'Dtype':<10} {'SeqLen':<8} {'EmbDim':<8} {'Triton Fwd':<12} {'Triton Bwd':<12} {'Triton F+B':<12} "
          f"{'PyTorch Fwd':<13} {'PyTorch Bwd':<13} {'PyTorch F+B':<13} {'Speedup Fwd':<12} {'Speedup Bwd':<12} {'Speedup F+B':<12}")
    print("-" * 160)
    
    for _, row in df.iterrows():
        print(f"{row['dtype']:<10} {row['seq_len']:<8} {row['embed_dim']:<8} "
              f"{row['triton_fwd_ms']:>10.3f} ms {row['triton_bwd_ms']:>10.3f} ms {row['triton_fwd_bwd_ms']:>10.3f} ms "
              f"{row['pytorch_fwd_ms']:>11.3f} ms {row['pytorch_bwd_ms']:>11.3f} ms {row['pytorch_fwd_bwd_ms']:>11.3f} ms "
              f"{row['speedup_fwd']:>10.2f}x {row['speedup_bwd']:>10.2f}x {row['speedup_fwd_bwd']:>10.2f}x")
    
    print("=" * 160)
    
    # Save results
    df.to_csv('flash_attention_benchmark_quick_results.csv', index=False)
    print(f"\nResults saved to: flash_attention_benchmark_quick_results.csv")
    
    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    print("\nAverage speedups by dtype:")
    for dtype_name in ['bfloat16', 'float32']:
        dtype_df = df[df['dtype'] == dtype_name]
        if len(dtype_df) > 0:
            print(f"  {dtype_name}:")
            print(f"    Forward:          {dtype_df['speedup_fwd'].mean():.2f}x (min: {dtype_df['speedup_fwd'].min():.2f}x, max: {dtype_df['speedup_fwd'].max():.2f}x)")
            print(f"    Backward:         {dtype_df['speedup_bwd'].mean():.2f}x (min: {dtype_df['speedup_bwd'].min():.2f}x, max: {dtype_df['speedup_bwd'].max():.2f}x)")
            print(f"    Forward+Backward: {dtype_df['speedup_fwd_bwd'].mean():.2f}x (min: {dtype_df['speedup_fwd_bwd'].min():.2f}x, max: {dtype_df['speedup_fwd_bwd'].max():.2f}x)")
    
    print("\nAverage speedups by sequence length:")
    for seq_len in sorted(df['seq_len'].unique()):
        seq_df = df[df['seq_len'] == seq_len]
        print(f"  seq_len={seq_len}:  F+B speedup: {seq_df['speedup_fwd_bwd'].mean():.2f}x")
    
    return df


if __name__ == '__main__':
    df = run_quick_benchmark()
