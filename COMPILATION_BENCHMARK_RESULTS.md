# Compilation Benchmark Results

## Part (a): Compiled vs Uncompiled Attention

### Configuration
- Batch size: 8
- d_model (head embedding dimension): 64
- Sequence length: 1024
- Iterations: 100
- Warmup: 10

### Results Table

| Version | Forward (ms) | Backward (ms) | Speedup (Fwd) | Speedup (Bwd) |
|---------|--------------|---------------|---------------|---------------|
| Uncompiled | 0.329 ± 0.006 | 0.801 ± 0.024 | 1.00x | 1.00x |
| Compiled | 0.369 ± 0.029 | 0.558 ± 0.044 | 0.89x | **1.44x** |

### Analysis

**Forward Pass:**
- Compiled version is slightly slower (0.89x) than uncompiled
- Likely due to compilation overhead for this small configuration
- The forward pass is already highly optimized with efficient matmul kernels

**Backward Pass:**
- Compiled version is **1.44x faster** than uncompiled
- Significant speedup from kernel fusion in gradient computation
- torch.compile() can fuse the gradient operations for softmax and matrix multiplications

**Key Takeaway:** torch.compile() provides more benefit for the backward pass in attention, where it can fuse gradient computation operations. The forward pass is already well-optimized by PyTorch's cuBLAS kernels for matrix multiplication.

---

## Part (b): Compiled vs Uncompiled Transformer Model

### Configuration
- Model: d_model=512, d_ff=2048, num_layers=4, num_heads=8
- Data: batch_size=4, context_length=256, vocab_size=10000
- Iterations: 100
- Warmup: 10

### Results Table

| Version | Forward Pass (ms) | Full Training Step (ms) | Speedup (Fwd) | Speedup (Full) |
|---------|-------------------|------------------------|---------------|----------------|
| Uncompiled | 7.369 ± 0.287 | 21.082 ± 0.755 | 1.00x | 1.00x |
| Compiled | 7.264 ± 0.223 | 18.960 ± 0.596 | **1.01x** | **1.11x** |

### Analysis

**Forward Pass:**
- Compiled version is marginally faster (1.01x speedup)
- Minimal improvement because forward operations are already well-optimized
- PyTorch's built-in kernels (matmul, softmax, etc.) are already highly efficient

**Full Training Step (Forward + Backward + Optimizer):**
- Compiled version is **1.11x faster** (11% speedup)
- More significant improvement when including backward pass and optimizer
- Compilation benefits accumulate across multiple operations

**Performance Breakdown:**
```
Uncompiled: 21.08 ms = 7.37 ms (forward) + 13.71 ms (backward+optimizer)
Compiled:   18.96 ms = 7.26 ms (forward) + 11.70 ms (backward+optimizer)
```

The backward + optimizer portion sees a **1.17x speedup** (13.71 ms → 11.70 ms).

### Why Backward Pass Benefits More

1. **Kernel Fusion Opportunities:**
   - Forward pass: Already uses efficient fused kernels (cuBLAS, cuDNN)
   - Backward pass: More opportunities to fuse gradient operations (e.g., fusing softmax backward with matmul backward)

2. **Memory Traffic Reduction:**
   - Fusing gradient computations reduces intermediate tensor materialization
   - Less memory bandwidth usage

3. **Graph-Level Optimizations:**
   - torch.compile() can reorder and fuse operations across layer boundaries
   - Particularly beneficial for the backward pass with many gradient accumulations

### Recommendations

1. **When to Use torch.compile():**
   - Larger models (more layers, larger hidden dimensions)
   - Longer sequences (more opportunities for fusion)
   - Production training (worth the compilation overhead)

2. **When Overhead May Dominate:**
   - Very small models or quick experiments
   - Frequent model architecture changes (requires recompilation)
   - Single inference runs (compilation cost not amortized)

3. **Best Practices:**
   - Always warm up the compiled model before benchmarking
   - Use `torch.set_float32_matmul_precision('high')` for better TF32 performance
   - Consider using `mode="reduce-overhead"` for even better performance in production

---

## Expected Speedups for Larger Models

For the 2.7B model configuration (d_model=2560, d_ff=10240, num_layers=32), we would expect:

- **Forward pass:** 1.1-1.3x speedup
- **Full training step:** 1.2-1.5x speedup

Larger models benefit more from compilation because:
- More operations to fuse across layers
- Compilation overhead is amortized over larger compute
- More complex graph structures benefit from optimizations

---

## Conclusion

**torch.compile()** provides modest but meaningful speedups for Transformer models:
- ~1-11% speedup for this small model configuration
- Larger benefits expected for bigger models and longer sequences
- Primary benefits come from the backward pass (gradient computation)
- Production training workloads should use compilation for cumulative benefits

The relatively small speedups (~1.1x) are actually expected for this model size because PyTorch's eager mode execution is already highly optimized. The real value of torch.compile() becomes apparent with:
1. Larger models where fusion opportunities compound
2. Longer training runs where compilation cost is amortized
3. Custom operations that aren't as well-optimized in eager mode
