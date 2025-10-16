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


def benchmark_forward(impl_fn, q, k, v, is_causal=True):
    """Benchmark forward pass."""
    def fn():
        out = impl_fn(q, k, v, is_causal)
        return out
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


def benchmark_backward(impl_fn, q, k, v, grad_output, is_causal=True):
    """Benchmark backward pass."""
    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        out = impl_fn(q, k, v, is_causal)
        out.backward(grad_output)
        torch.cuda.synchronize()
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


def benchmark_forward_backward(impl_fn, q, k, v, grad_output, is_causal=True):
    """Benchmark end-to-end forward + backward pass."""
    def fn():
        q.grad = None
        k.grad = None
        v.grad = None
        out = impl_fn(q, k, v, is_causal)
        out.backward(grad_output)
        torch.cuda.synchronize()
    
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms


def get_tile_sizes(seq_len, d):
    """Adjust tile sizes based on input dimensions."""
    # For larger sequences or smaller dimensions, use smaller tiles
    if seq_len >= 16384 or d <= 32:
        return 32, 32
    elif seq_len >= 8192:
        return 64, 64
    else:
        return 128, 128


def run_benchmark():
    """Run comprehensive benchmarks comparing Triton and PyTorch implementations."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    device = 'cuda'
    batch_size = 1
    is_causal = True
    
    # Sweep parameters
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    embed_dims = [16, 32, 64, 128]
    dtypes = [torch.bfloat16, torch.float32]
    
    FlashAttn = get_flashattention_autograd_function_triton()
    
    results = []
    
    print("Starting FlashAttention-2 Benchmarks")
    print("=" * 100)
    
    for dtype in dtypes:
        dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float32"
        print(f"\nTesting with dtype: {dtype_name}")
        
        for d in embed_dims:
            print(f"  Embedding dimension: {d}")
            
            for seq_len in seq_lengths:
                # Skip if we might run out of memory
                if seq_len > 32768 and d >= 64:
                    print(f"    Skipping seq_len={seq_len} (memory constraints)")
                    continue
                
                try:
                    print(f"    Testing seq_len={seq_len}...", end=" ")
                    
                    # Generate random inputs
                    torch.manual_seed(42)
                    q = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype, requires_grad=True)
                    grad_output = torch.randn(batch_size, seq_len, d, device=device, dtype=dtype)
                    
                    # Warm up
                    _ = FlashAttn.apply(q.clone().detach(), k.clone().detach(), v.clone().detach(), is_causal)
                    _ = pytorch_attention(q.clone().detach(), k.clone().detach(), v.clone().detach(), is_causal)
                    torch.cuda.synchronize()
                    
                    # Benchmark Triton implementation
                    q_triton = q.clone().detach().requires_grad_(True)
                    k_triton = k.clone().detach().requires_grad_(True)
                    v_triton = v.clone().detach().requires_grad_(True)
                    
                    triton_fwd = benchmark_forward(FlashAttn.apply, q_triton, k_triton, v_triton, is_causal)
                    triton_fwd_bwd = benchmark_forward_backward(FlashAttn.apply, q_triton, k_triton, v_triton, grad_output, is_causal)
                    triton_bwd = triton_fwd_bwd - triton_fwd
                    
                    # Benchmark PyTorch implementation
                    q_torch = q.clone().detach().requires_grad_(True)
                    k_torch = k.clone().detach().requires_grad_(True)
                    v_torch = v.clone().detach().requires_grad_(True)
                    
                    torch_fwd = benchmark_forward(pytorch_attention, q_torch, k_torch, v_torch, is_causal)
                    torch_fwd_bwd = benchmark_forward_backward(pytorch_attention, q_torch, k_torch, v_torch, grad_output, is_causal)
                    torch_bwd = torch_fwd_bwd - torch_fwd
                    
                    # Calculate speedups
                    speedup_fwd = torch_fwd / triton_fwd
                    speedup_bwd = torch_bwd / triton_bwd
                    speedup_fwd_bwd = torch_fwd_bwd / triton_fwd_bwd
                    
                    results.append({
                        'dtype': dtype_name,
                        'seq_len': seq_len,
                        'embed_dim': d,
                        'triton_fwd_ms': triton_fwd,
                        'triton_bwd_ms': triton_bwd,
                        'triton_fwd_bwd_ms': triton_fwd_bwd,
                        'pytorch_fwd_ms': torch_fwd,
                        'pytorch_bwd_ms': torch_bwd,
                        'pytorch_fwd_bwd_ms': torch_fwd_bwd,
                        'speedup_fwd': speedup_fwd,
                        'speedup_bwd': speedup_bwd,
                        'speedup_fwd_bwd': speedup_fwd_bwd,
                    })
                    
                    print(f"Done (Triton fwd+bwd: {triton_fwd_bwd:.2f}ms, PyTorch fwd+bwd: {torch_fwd_bwd:.2f}ms, Speedup: {speedup_fwd_bwd:.2f}x)")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"OOM - skipping")
                        torch.cuda.empty_cache()
                    else:
                        print(f"Error: {e}")
                        raise
    
    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)
    print("\nForward Pass Latencies (ms):")
    print("-" * 100)
    pivot_fwd = df.pivot_table(
        values=['triton_fwd_ms', 'pytorch_fwd_ms', 'speedup_fwd'],
        index=['dtype', 'embed_dim'],
        columns='seq_len',
        aggfunc='first'
    )
    print(pivot_fwd.to_string())
    
    print("\n\nBackward Pass Latencies (ms):")
    print("-" * 100)
    pivot_bwd = df.pivot_table(
        values=['triton_bwd_ms', 'pytorch_bwd_ms', 'speedup_bwd'],
        index=['dtype', 'embed_dim'],
        columns='seq_len',
        aggfunc='first'
    )
    print(pivot_bwd.to_string())
    
    print("\n\nEnd-to-End Forward+Backward Latencies (ms):")
    print("-" * 100)
    pivot_fwd_bwd = df.pivot_table(
        values=['triton_fwd_bwd_ms', 'pytorch_fwd_bwd_ms', 'speedup_fwd_bwd'],
        index=['dtype', 'embed_dim'],
        columns='seq_len',
        aggfunc='first'
    )
    print(pivot_fwd_bwd.to_string())
    
    # Save detailed results
    df.to_csv('flash_attention_benchmark_results.csv', index=False)
    print("\n\nDetailed results saved to: flash_attention_benchmark_results.csv")
    
    # Print detailed table
    print("\n\nDETAILED RESULTS TABLE:")
    print("-" * 160)
    print(f"{'Dtype':<10} {'SeqLen':<8} {'EmbDim':<8} {'Triton Fwd':<12} {'Triton Bwd':<12} {'Triton F+B':<12} "
          f"{'PyTorch Fwd':<13} {'PyTorch Bwd':<13} {'PyTorch F+B':<13} {'Speedup Fwd':<12} {'Speedup Bwd':<12} {'Speedup F+B':<12}")
    print("-" * 160)
    
    for _, row in df.iterrows():
        print(f"{row['dtype']:<10} {row['seq_len']:<8} {row['embed_dim']:<8} "
              f"{row['triton_fwd_ms']:>10.3f} ms {row['triton_bwd_ms']:>10.3f} ms {row['triton_fwd_bwd_ms']:>10.3f} ms "
              f"{row['pytorch_fwd_ms']:>11.3f} ms {row['pytorch_bwd_ms']:>11.3f} ms {row['pytorch_fwd_bwd_ms']:>11.3f} ms "
              f"{row['speedup_fwd']:>10.2f}x {row['speedup_bwd']:>10.2f}x {row['speedup_fwd_bwd']:>10.2f}x")
    
    print("-" * 160)
    
    return df


if __name__ == '__main__':
    df = run_benchmark()
